#!/usr/bin/env python3
"""
Ragas-Based Evaluation Pipeline for ReAct Chatbot Responses

This script uses the Ragas framework to evaluate two models' responses by automatically
parsing ReAct logs to extract Final Answers and Contexts, then running the faithfulness
metric to measure how grounded the answers are in the retrieved context.

This metric doesn't require ground truth, making it suitable for evaluation when
reference answers are not available.

Usage:
    python ragas_llm_judge_evaluator.py <input_csv_file>

Requirements:
    - Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and MODEL_NAME environment variables
    - Input CSV must have 3 columns with header: Question, Model_A_Full_Log, Model_B_Full_Log
    - Install dependencies: pip install -r requirements.txt (ensure ragas and datasets are included)

Output:
    - ragas_evaluation_output.csv containing all parsed data and Ragas scores
"""

import argparse
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from openai import AzureOpenAI
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
)
from tqdm import tqdm

# Load environment variables from .env file if it exists
load_dotenv()


def parse_react_log(log_text: str) -> Tuple[str, List[str]]:
    """
    Parse a ReAct log string to extract the Final Answer and Contexts.
    
    The ReAct log format includes multiple sections, all of which provide context:
    - ## 📝 Task タスク - Task classification
    - ## 💬 Reaction 反応 - Reaction type
    - ## 📂 Classification 分類 - Classification
    - ## 📊 Status 状態 - Status
    - ## 🤖 LLM Thought Process 思考 - The LLM's reasoning
    - ## ⚡ Action 行動 - Action taken
    - ## ⌨️ Action Input 行動入力 - Action input
    - ## 📚 Raw Search Results (Cleaned) 観察 - Retrieved search results
    - ## ✅ Final Answer 回答 - The final answer (not included in contexts)
    
    Args:
        log_text: The full ReAct log string
    
    Returns:
        Tuple of (final_answer, contexts_list)
        - final_answer: The extracted final answer text
        - contexts_list: List of context strings (includes thought process + search results)
    """
    # Extract Final Answer
    final_answer = ""
    final_answer_pattern = r'## ✅ Final Answer 回答\s*---\s*(.*?)(?=\n## |$)'
    final_answer_match = re.search(final_answer_pattern, log_text, re.DOTALL)
    
    if final_answer_match:
        final_answer = final_answer_match.group(1).strip()
    else:
        # Fallback: try without the dashes
        final_answer_pattern_alt = r'## ✅ Final Answer 回答\s*(.*?)(?=\n## |$)'
        final_answer_match_alt = re.search(final_answer_pattern_alt, log_text, re.DOTALL)
        if final_answer_match_alt:
            final_answer = final_answer_match_alt.group(1).strip()
    
    # Extract all ReAct components as contexts
    contexts = []
    
    # Helper function to extract a section
    def extract_section(pattern_name: str, pattern: str) -> Optional[str]:
        match = re.search(pattern, log_text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            if content and content.lower() not in ['none', 'n/a', '']:
                return f"[{pattern_name}]\n{content}"
        return None
    
    # 1. Extract Task
    task = extract_section("Task", r'## 📝 Task タスク\s*---\s*(.*?)(?=\n## |$)')
    if task:
        contexts.append(task)
    
    # 2. Extract Classification
    classification = extract_section("Classification", r'## 📂 Classification 分類\s*---\s*(.*?)(?=\n## |$)')
    if classification:
        contexts.append(classification)
    
    # 3. Extract Status
    status = extract_section("Status", r'## 📊 Status 状態\s*---\s*(.*?)(?=\n## |$)')
    if status:
        contexts.append(status)
    
    # 4. Extract LLM Thought Process (MOST IMPORTANT)
    thought_process = extract_section("LLM Thought Process", r'## 🤖 LLM Thought Process 思考\s*---\s*(.*?)(?=\n## |$)')
    if thought_process:
        contexts.append(thought_process)
    
    # 5. Extract Action
    action = extract_section("Action", r'## ⚡ Action 行動\s*---\s*(.*?)(?=\n## |$)')
    if action:
        contexts.append(action)
    
    # 6. Extract Action Input
    action_input = extract_section("Action Input", r'## ⌨️ Action Input 行動入力\s*---\s*(.*?)(?=\n## |$)')
    if action_input:
        contexts.append(action_input)
    
    # 7. Extract Raw Search Results (can have multiple results)
    search_results_pattern = r'## 📚 Raw Search Results \(Cleaned\) 観察\s*---\s*(.*?)(?=\n## ✅ Final Answer|$)'
    search_results_match = re.search(search_results_pattern, log_text, re.DOTALL)
    
    if search_results_match:
        search_results_text = search_results_match.group(1)
        
        # Split by the separator (################################################)
        result_separator = r'#{40,}'
        individual_results = re.split(result_separator, search_results_text)
        
        # Clean and collect non-empty results
        for idx, result in enumerate(individual_results, 1):
            result_clean = result.strip()
            if result_clean and len(result_clean) > 20:  # Filter out very short/empty results
                # Remove the URLs section from each result if present
                result_without_urls = re.sub(r'## 🔗 URLs URL.*$', '', result_clean, flags=re.DOTALL)
                result_without_urls = result_without_urls.strip()
                if result_without_urls:
                    contexts.append(f"[Search Result {idx}]\n{result_without_urls}")
    
    # If no contexts found, return a placeholder to avoid Ragas errors
    if not contexts:
        contexts = ["No context retrieved"]
    
    # If no final answer found, use a placeholder
    if not final_answer:
        final_answer = "No answer provided"
    
    return final_answer, contexts


def initialize_azure_openai_for_ragas():
    """
    Initialize Azure OpenAI client and wrap it for Ragas.
    
    Returns:
        Tuple of (ragas_llm, client) where:
        - ragas_llm is the LangChain LLM for Ragas
        - client is the Azure OpenAI client
    """
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    model_name = os.getenv("MODEL_NAME", "gpt-5")
    
    if not azure_endpoint or not azure_api_key:
        print("ERROR: Azure OpenAI credentials not found.", file=sys.stderr)
        print("\nPlease set the following environment variables:", file=sys.stderr)
        print("  export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/'", file=sys.stderr)
        print("  export AZURE_OPENAI_API_KEY='your-api-key'", file=sys.stderr)
        print("  export MODEL_NAME='gpt-5'  # or your deployment name (e.g., 'gpt-4.1')", file=sys.stderr)
        sys.exit(1)
    
    print("Initializing Azure OpenAI for Ragas evaluation (Faithfulness only)")
    print(f"Endpoint: {azure_endpoint}")
    print(f"Model/Deployment: {model_name}")
    print(f"API Version: {azure_api_version}")
    
    # Create Azure OpenAI client for direct API access (if needed)
    client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        api_version=azure_api_version
    )
    
    # Create LangChain-compatible LLM for Ragas
    from langchain_openai import AzureChatOpenAI
    
    # Configure parameters based on model type
    llm_params = {
        "azure_endpoint": azure_endpoint,
        "api_key": azure_api_key,
        "api_version": azure_api_version,
        "deployment_name": model_name,
        "model_name": model_name,
    }
    
    # GPT-5 の場合は max_completion_tokens を使用、その他は max_tokens を使用
    if model_name == "gpt-5":
        llm_params["max_completion_tokens"] = 2000
        llm_params["temperature"] = 1  # GPT-5 defaults to temperature 1
    else:
        llm_params["max_tokens"] = 2000
        llm_params["temperature"] = 0.7  # GPT-4.1 では temperature も設定可能
    
    print(f"LLM Configuration: temperature={llm_params.get('temperature')}, "
          f"max_tokens={'max_completion_tokens' if model_name == 'gpt-5' else 'max_tokens'}={llm_params.get('max_completion_tokens', llm_params.get('max_tokens'))}")
    
    langchain_llm = AzureChatOpenAI(**llm_params)
    
    return langchain_llm, client


def evaluate_with_ragas(
    questions: List[str],
    answers: List[str],
    contexts_list: List[List[str]],
    llm,
    model_name: str = "Model"
) -> pd.DataFrame:
    """
    Evaluate responses using Ragas faithfulness metric.
    
    Args:
        questions: List of questions
        answers: List of answers
        contexts_list: List of context lists (each answer has multiple contexts)
        llm: The LLM to use for evaluation (LangChain-compatible)
        model_name: Name for labeling (e.g., "Model_A" or "Model_B")
    
    Returns:
        DataFrame with Ragas faithfulness scores
    """
    print(f"\nEvaluating {model_name} with Ragas (Faithfulness only)...")
    
    # Prepare dataset for Ragas
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
    }
    
    dataset = Dataset.from_dict(data)
    
    # Define metrics to evaluate (only faithfulness - no embeddings required)
    metrics_to_use = [
        faithfulness,
    ]
    
    # Run evaluation
    try:
        results = evaluate(
            dataset=dataset,
            metrics=metrics_to_use,
            llm=llm,
        )
        
        # Convert results to DataFrame
        results_df = results.to_pandas()
        
        # Rename columns with model prefix
        score_columns = {
            "faithfulness": f"{model_name}_faithfulness_score",
        }
        
        results_df = results_df.rename(columns=score_columns)
        
        return results_df
        
    except Exception as e:
        print(f"ERROR during Ragas evaluation for {model_name}: {e}", file=sys.stderr)
        print(f"Traceback:", file=sys.stderr)
        import traceback
        traceback.print_exc()
        
        # Return empty DataFrame with expected columns
        error_df = pd.DataFrame({
            f"{model_name}_faithfulness_score": [None] * len(questions),
        })
        
        return error_df


def process_csv(
    input_file: str,
    output_file: str = "ragas_evaluation_output.csv",
    limit_rows: Optional[int] = None
) -> None:
    """
    Main processing function that reads the input CSV, parses ReAct logs,
    evaluates with Ragas, and writes results to output CSV.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to the output CSV file
        limit_rows: Optional limit on number of rows to process
    """
    # Initialize Azure OpenAI for Ragas
    llm, client = initialize_azure_openai_for_ragas()
    
    # Read input CSV
    print(f"\nReading input file: {input_file}")
    try:
        df = pd.read_csv(input_file)
        
        # Validate columns
        expected_columns = ["Question", "Model_A_Response", "Model_B_Response"]
        # Allow both Response and Full_Log naming
        if "Model_A_Full_Log" in df.columns:
            df = df.rename(columns={
                "Model_A_Full_Log": "Model_A_Response",
                "Model_B_Full_Log": "Model_B_Response"
            })
        
        if not all(col in df.columns for col in expected_columns):
            print(f"ERROR: CSV must have columns: {expected_columns}", file=sys.stderr)
            print(f"Found columns: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)
            
    except FileNotFoundError:
        print(f"ERROR: Input file '{input_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to read input file: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded {len(df)} rows from input file.")
    
    # Apply row limit if specified
    if limit_rows is not None and limit_rows < len(df):
        df = df.head(limit_rows)
        print(f"⚠️  LIMITING to first {limit_rows} rows for testing")
    
    # Parse ReAct logs for both models
    print("\nParsing ReAct logs...")
    model_a_answers = []
    model_a_contexts = []
    model_b_answers = []
    model_b_contexts = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing logs"):
        # Parse Model A log
        a_answer, a_contexts = parse_react_log(row["Model_A_Response"])
        model_a_answers.append(a_answer)
        model_a_contexts.append(a_contexts)
        
        # Parse Model B log
        b_answer, b_contexts = parse_react_log(row["Model_B_Response"])
        model_b_answers.append(b_answer)
        model_b_contexts.append(b_contexts)
    
    # Add parsed data to DataFrame
    df["model_A_answer"] = model_a_answers
    df["model_A_contexts"] = model_a_contexts
    df["model_B_answer"] = model_b_answers
    df["model_B_contexts"] = model_b_contexts
    
    print(f"✓ Parsed {len(df)} ReAct logs for both models")
    
    # Evaluate Model A with Ragas
    print("\n" + "=" * 70)
    print("EVALUATING MODEL A")
    print("=" * 70)
    model_a_results = evaluate_with_ragas(
        questions=df["Question"].tolist(),
        answers=model_a_answers,
        contexts_list=model_a_contexts,
        llm=llm,
        model_name="Model_A"
    )
    
    # Evaluate Model B with Ragas
    print("\n" + "=" * 70)
    print("EVALUATING MODEL B")
    print("=" * 70)
    model_b_results = evaluate_with_ragas(
        questions=df["Question"].tolist(),
        answers=model_b_answers,
        contexts_list=model_b_contexts,
        llm=llm,
        model_name="Model_B"
    )
    
    # Merge results back into main DataFrame
    # Only merge the score columns (not the duplicated question/answer/contexts columns)
    score_columns_a = [col for col in model_a_results.columns if col.startswith("Model_A_")]
    score_columns_b = [col for col in model_b_results.columns if col.startswith("Model_B_")]
    
    for col in score_columns_a:
        df[col] = model_a_results[col].values
    
    for col in score_columns_b:
        df[col] = model_b_results[col].values
    
    # Prepare final output DataFrame with all columns
    output_columns = [
        "Question",
        "Model_A_Response",
        "Model_B_Response",
        "model_A_answer",
        "model_A_contexts",
        "model_B_answer",
        "model_B_contexts",
    ] + score_columns_a + score_columns_b
    
    output_df = df[output_columns]
    
    # Save to CSV
    output_df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 70)
    print("✓ EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"✓ Results written to: {output_file}")
    print(f"✓ Processed {len(output_df)} rows")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    for model_name, prefix in [("Model A", "Model_A_"), ("Model B", "Model_B_")]:
        print(f"\n{model_name}:")
        col_name = f"{prefix}faithfulness_score"
        if col_name in output_df.columns:
            mean_score = output_df[col_name].mean()
            print(f"  faithfulness        : {mean_score:.4f}")


def main():
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(
        description="Ragas-Based Evaluation Pipeline for ReAct Chatbot Responses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python ragas_llm_judge_evaluator.py test_5_rows.csv
    python ragas_llm_judge_evaluator.py /path/to/input.csv -o my_ragas_results.csv
    python ragas_llm_judge_evaluator.py input.csv -n 3  # Test with 3 rows only

Setup:
    1. Install dependencies:
       pip install -r requirements.txt
       
    2. Ensure requirements.txt includes:
       - ragas
       - datasets
       - langchain-openai
       - openai
       - pandas
       - python-dotenv
       - tqdm
       
    3. Set environment variables:
       export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/'
       export AZURE_OPENAI_API_KEY='your-api-key'
       export MODEL_NAME='gpt-5'  # or your deployment name
       export AZURE_OPENAI_API_VERSION='2024-08-01-preview'  # optional
       
    4. Prepare input CSV with columns:
       - Question
       - Model_A_Response (or Model_A_Full_Log)
       - Model_B_Response (or Model_B_Full_Log)
       
Input Format:
    The script expects ReAct logs with the following structure:
    - ## ✅ Final Answer 回答 section containing the final answer
    - ## 📚 Raw Search Results (Cleaned) 観察 section with search results
    - Results separated by ################################################
    
Output:
    A CSV file (ragas_evaluation_output.csv by default) containing:
    - Original columns (Question, Model_A_Response, Model_B_Response)
    - Parsed columns (model_A_answer, model_A_contexts, etc.)
    - Ragas scores for both models (no ground truth required):
      * Model_A_faithfulness_score / Model_B_faithfulness_score
        """
    )
    
    parser.add_argument(
        "input_csv",
        help="Path to the input CSV file (must have header: Question, Model_A_Response, Model_B_Response)"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="ragas_evaluation_output.csv",
        help="Path to the output CSV file (default: ragas_evaluation_output.csv)"
    )
    
    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=None,
        help="Limit processing to first N rows (useful for testing)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Ragas-Based Evaluation Pipeline for ReAct Chatbot Responses")
    print("=" * 70)
    
    process_csv(
        args.input_csv,
        args.output,
        limit_rows=args.limit
    )


if __name__ == "__main__":
    main()
