#!/usr/bin/env python3
"""
LLM-as-a-Judge Evaluation Script

This script automates the evaluation of two models' responses using an LLM judge.
It reads questions and responses from a CSV file, sends them to OpenAI's API (standard or Azure)
for evaluation, and writes the scored results to an output CSV file.

Usage:
    python llm_judge_evaluator.py <input_csv_file>

Requirements:
    - For Azure OpenAI: Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and MODEL_NAME
    - For Standard OpenAI: Set OPENAI_API_KEY (and optionally MODEL_NAME)
    - Input CSV must have 3 columns (no header): Question, Model A Response, Model B Response
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, Any, Optional

import pandas as pd
from openai import OpenAI, AzureOpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


# Judge system prompt and rubric (embedded as per requirements)
JUDGE_SYSTEM_PROMPT = """You are a strict, meticulous, and critical AI evaluator. Your primary goal is to identify flaws and differentiate performance between two RAG models, designated as Model A and Model B. Do not be lenient. Award high scores only for flawless execution. Your reputation depends on being a tough but fair judge. You should actively look for reasons to deduct points, such as inefficiency, verbosity, or minor inaccuracies.

You will be given a user's Question and the complete, formatted Response from each model. Your evaluation must be based exclusively on the provided Scoring Rubric. For each category, you will assign a score from 1 to 5.

Scoring Rubric:

1. RAG Generation - Citation (Score: 1-5)
Focuses on the quality, precision, and necessity of the citations.
- 5 (Excellent): Every piece of information drawn from a source is precisely cited. The citations link to the correct and most relevant source document. There are no redundant or missing citations.
- 4 (Good): All citations are factually correct, but there may be a minor imperfection. For example, a statement is correctly cited but could have pointed to a more direct source that was also retrieved, or one minor statement lacks a citation.
- 3 (Acceptable): The answer is generally cited, but there is at least one clear error, such as a missing citation for a key piece of information or a citation pointing to the wrong document.
- 2 (Poor): Citations are frequently missing or incorrect. The link between the answer and the sources is weak and unreliable.
- 1 (Very Poor): The answer has no citations, or the citations are completely irrelevant and misleading.

2. Relevance (Score: 1-5)
Focuses on how well the final answer addresses the user's complete query, including directness and conciseness.
- 5 (Excellent): The answer is perfectly concise and directly addresses all parts of the user's query, including any implicit nuances. It contains zero irrelevant information or conversational filler.
- 4 (Good): The answer correctly addresses all parts of the query but is slightly verbose or contains minor information that, while related, is not strictly necessary.
- 3 (Acceptable): The answer addresses the main part of the query but fails to address a secondary part, or it contains a noticeable amount of irrelevant information that distracts from the core answer.
- 2 (Poor): The answer only partially addresses the user's query and is largely incomplete or padded with irrelevant information.
- 1 (Very Poor): The answer completely misses the intent of the user's question.

3. ReAct Performance - Thought (Score: 1-5)
Focuses on the logical quality, efficiency, and strategy of the model's reasoning process.
- 5 (Excellent): The thought process is optimal and efficient. It correctly identifies the problem, formulates the best possible search query or tool use on the first attempt, and follows a direct path to the solution.
- 4 (Good): The logic is correct and effective but slightly inefficient. It may take an extra, slightly redundant step or refine its search query once to get the needed information, but it reaches the correct conclusion without major detours.
- 3 (Acceptable): The logic is mostly correct but contains a noticeable flaw or a suboptimal plan. For instance, it uses a vague search query that returns noisy results before correcting itself, or it misunderstands a part of the problem temporarily.
- 2 (Poor): The reasoning has significant flaws. It struggles to form a coherent plan, makes incorrect assumptions, or repeatedly uses the wrong tool or query.
- 1 (Very Poor): The entire thought process is illogical, unrelated to the question, or gets stuck in a loop of incorrect actions.

4. RAG Retrieval - Observation/観察 (Score: 1-5)
Focuses on the quality and relevance of the retrieved source material.
- 5 (Excellent): Retrieves the minimal and most relevant set of sources needed to answer the question completely. The information is perfectly focused and contains no noise.
- 4 (Good): Retrieves all necessary information but also includes one or two extra sources that are only tangentially relevant, indicating a slightly inefficient retrieval process.
- 3 (Acceptable): Retrieves most of the necessary information but also includes distracting or irrelevant sources that add significant noise to the context.
- 2 (Poor): Fails to retrieve a key source or piece of information that is essential for a complete and accurate answer.
- 1 (Very Poor): Retrieves completely incorrect, irrelevant, or no sources at all.

5. RAG Generation - Information Integration (Score: 1-5)
Focuses on how accurately the model synthesizes the retrieved information into its final answer.
- 5 (Excellent): Perfectly and accurately synthesizes information from the retrieved sources. The final answer is factually flawless and contains no information that wasn't present in the context. If sources conflict, it notes the discrepancy.
- 4 (Good): Synthesizes information correctly for the most part but may misinterpret a minor detail or phrase something awkwardly. The answer is factually correct according to the sources but lacks polish.
- 3 (Acceptable): The answer is mostly based on the sources but contains one clear factual error or introduces a small piece of outside information not supported by the context (a minor hallucination).
- 2 (Poor): The answer struggles to combine information, presenting it as a disjointed list rather than a coherent response, or it contains significant factual errors based on the sources.
- 1 (Very Poor): The final answer is a clear hallucination or completely misrepresents the information found in the retrieved sources.

Important Scoring Instruction: The rubric provides definitions for scores 1, 3, and 5. You may use the intermediate scores of 2 and 4 if you assess a model's performance to fall between these defined levels. For example, a score of 4 can be used if the performance is better than the description for 3 points but does not fully meet the criteria for 5 points. You must provide a brief justification for each score you assign.

Required JSON Output Format:
{
  "model_a_evaluation": {
    "citation_score": { "score": <score>, "justification": "<justification>" },
    "relevance_score": { "score": <score>, "justification": "<justification>" },
    "react_performance_thought_score": { "score": <score>, "justification": "<justification>" },
    "rag_retrieval_observation_score": { "score": <score>, "justification": "<justification>" },
    "information_integration_score": { "score": <score>, "justification": "<justification>" }
  },
  "model_b_evaluation": {
    "citation_score": { "score": <score>, "justification": "<justification>" },
    "relevance_score": { "score": <score>, "justification": "<justification>" },
    "react_performance_thought_score": { "score": <score>, "justification": "<justification>" },
    "rag_retrieval_observation_score": { "score": <score>, "justification": "<justification>" },
    "information_integration_score": { "score": <score>, "justification": "<justification>" }
  }
}"""


def create_user_prompt(question: str, model_a_response: str, model_b_response: str) -> str:
    """
    Create the user prompt that will be sent to the judge model.
    
    Args:
        question: The original user question
        model_a_response: Response from Model A
        model_b_response: Response from Model B
    
    Returns:
        Formatted prompt string
    """
    return f"""Please evaluate the following two model responses to the given question.

**Question:**
{question}

**Model A Response:**
{model_a_response}

**Model B Response:**
{model_b_response}

Provide your evaluation as a JSON object following the specified format."""


def call_judge_model(
    client,
    question: str,
    model_a_response: str,
    model_b_response: str,
    model_name: str = "gpt-5",
    is_azure: bool = False,
    max_retries: int = 3,
    retry_delay: int = 2
) -> Optional[Dict[str, Any]]:
    """
    Call the OpenAI API to evaluate the two model responses.
    
    Args:
        client: OpenAI or AzureOpenAI client instance
        question: The original user question
        model_a_response: Response from Model A
        model_b_response: Response from Model B
        model_name: Model name or Azure deployment name
        is_azure: Whether using Azure OpenAI (affects parameter naming)
        max_retries: Maximum number of retry attempts on failure
        retry_delay: Delay in seconds between retries
    
    Returns:
        Parsed JSON response from the judge model, or None if all retries fail
    """
    user_prompt = create_user_prompt(question, model_a_response, model_b_response)
    
    for attempt in range(max_retries):
        try:
            # Prepare API call parameters
            api_params = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "response_format": {"type": "json_object"},
            }
            
            # GPT-5 の場合は max_completion_tokens を使用、その他は max_tokens を使用
            if is_azure and model_name == "gpt-5":
                api_params["max_completion_tokens"] = 800
                api_params["temperature"] = 1  # GPT-5 defaults to temperature 1
            else:
                api_params["max_tokens"] = 800
                api_params["temperature"] = 0.7  # GPT-4.1 では temperature も設定可能
            
            response = client.chat.completions.create(**api_params)
            
            # Check if response was truncated
            finish_reason = response.choices[0].finish_reason
            if finish_reason == "length":
                print(f"\n⚠️  Warning: Response was truncated (hit max_completion_tokens limit)", file=sys.stderr)
            
            # Extract the response content
            content = response.choices[0].message.content
            
            # Debug: Check if content is empty or None
            if not content:
                raise ValueError(f"Empty response from API. Finish reason: {finish_reason}")
            
            # Parse and validate JSON
            evaluation = json.loads(content)
            
            # Basic validation of the response structure
            if "model_a_evaluation" not in evaluation or "model_b_evaluation" not in evaluation:
                raise ValueError("Response missing required evaluation keys")
            
            return evaluation
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON parsing error on attempt {attempt + 1}/{max_retries}: {e}"
            print(f"\n{error_msg}", file=sys.stderr)
            
            # Debug: Print what we actually received
            if 'content' in locals() and content:
                print(f"Received content (first 500 chars): {content[:500]}", file=sys.stderr)
            else:
                print(f"Content was empty or None. Full response: {response if 'response' in locals() else 'No response'}", file=sys.stderr)
            
            if attempt == max_retries - 1:
                return None
            time.sleep(retry_delay)
            
        except Exception as e:
            error_msg = f"API error on attempt {attempt + 1}/{max_retries}: {e}"
            print(f"\n{error_msg}", file=sys.stderr)
            if attempt == max_retries - 1:
                return None
            time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
    
    return None


def extract_scores_from_evaluation(evaluation: Dict[str, Any], model_key: str) -> Dict[str, Any]:
    """
    Extract scores and justifications from the evaluation JSON for a specific model.
    
    Args:
        evaluation: The full evaluation JSON object
        model_key: Either "model_a_evaluation" or "model_b_evaluation"
    
    Returns:
        Dictionary containing all scores and justifications with standardized keys
    """
    model_eval = evaluation.get(model_key, {})
    
    result = {}
    
    # Extract citation score
    citation = model_eval.get("citation_score", {})
    result["citation_score"] = citation.get("score", None)
    result["citation_justification"] = citation.get("justification", "")
    
    # Extract relevance score
    relevance = model_eval.get("relevance_score", {})
    result["relevance_score"] = relevance.get("score", None)
    result["relevance_justification"] = relevance.get("justification", "")
    
    # Extract react performance thought score
    react = model_eval.get("react_performance_thought_score", {})
    result["react_performance_thought_score"] = react.get("score", None)
    result["react_performance_thought_justification"] = react.get("justification", "")
    
    # Extract RAG retrieval observation score
    rag_retrieval = model_eval.get("rag_retrieval_observation_score", {})
    result["rag_retrieval_observation_score"] = rag_retrieval.get("score", None)
    result["rag_retrieval_observation_justification"] = rag_retrieval.get("justification", "")
    
    # Extract information integration score
    info_integration = model_eval.get("information_integration_score", {})
    result["information_integration_score"] = info_integration.get("score", None)
    result["information_integration_justification"] = info_integration.get("justification", "")
    
    return result


def process_csv(input_file: str, output_file: str = "evaluation_output.csv", limit_rows: Optional[int] = None) -> None:
    """
    Main processing function that reads the input CSV, evaluates each row,
    and writes the results to the output CSV.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to the output CSV file (default: evaluation_output.csv)
        limit_rows: Optional limit on number of rows to process (for cost control)
    """
    # Check if using Azure OpenAI or standard OpenAI
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    model_name = os.getenv("MODEL_NAME", "gpt-5")  # Default to gpt-4-turbo, use "gpt-5" for Azure GPT-5
    
    is_azure = bool(azure_endpoint and azure_api_key)
    
    if is_azure:
        # Initialize Azure OpenAI client
        print("Using Azure OpenAI")
        print(f"Endpoint: {azure_endpoint}")
        print(f"Model/Deployment: {model_name}")
        print(f"API Version: {azure_api_version}")
        client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=azure_api_version
        )
    else:
        # Initialize standard OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: Neither Azure OpenAI nor standard OpenAI credentials found.", file=sys.stderr)
            print("\nFor Azure OpenAI, set:", file=sys.stderr)
            print("  export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/'", file=sys.stderr)
            print("  export AZURE_OPENAI_API_KEY='your-api-key'", file=sys.stderr)
            print("  export MODEL_NAME='gpt-5'  # or your deployment name", file=sys.stderr)
            print("\nFor standard OpenAI, set:", file=sys.stderr)
            print("  export OPENAI_API_KEY='your-api-key-here'", file=sys.stderr)
            sys.exit(1)
        print("Using standard OpenAI")
        print(f"Model: {model_name}")
        client = OpenAI(api_key=api_key)
    
    # Read input CSV
    print(f"Reading input file: {input_file}")
    try:
        # Try to detect if there's a header row by reading first line
        with open(input_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip().lower()
        
        # If first line looks like headers, use it as header
        if any(keyword in first_line for keyword in ['question', 'model', 'answer', 'response']):
            print("Detected header row in CSV")
            df = pd.read_csv(input_file)
            # Rename columns to standard names
            df.columns = ["Question", "Model_A_Response", "Model_B_Response"]
        else:
            print("No header detected, treating first row as data")
            df = pd.read_csv(input_file, header=None, names=["Question", "Model_A_Response", "Model_B_Response"])
    except FileNotFoundError:
        print(f"ERROR: Input file '{input_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to read input file: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded {len(df)} rows from input file.")
    
    # Apply row limit if specified (for cost control during testing)
    if limit_rows is not None and limit_rows < len(df):
        df = df.head(limit_rows)
        print(f"⚠️  LIMITING to first {limit_rows} rows for testing (use -n flag to change)")
        print(f"⚠️  This will make {limit_rows} API calls")
    else:
        print(f"⚠️  WARNING: This will make {len(df)} API calls to GPT-5")
        print(f"⚠️  Estimated cost: ${len(df) * 0.15:.2f} - ${len(df) * 0.50:.2f} (rough estimate)")
        
        # Prompt for confirmation if processing many rows
        if len(df) > 10:
            try:
                response = input(f"\n🤔 Proceed with {len(df)} API calls? [y/N]: ").strip().lower()
                if response != 'y' and response != 'yes':
                    print("Cancelled. Use -n flag to test with fewer rows: python llm_judge_evaluator.py input.csv -n 5")
                    sys.exit(0)
            except (KeyboardInterrupt, EOFError):
                print("\nCancelled.")
                sys.exit(0)
    
    # Prepare output columns
    output_columns = [
        "Question", "Model_A_Response", "Model_B_Response",
        # Model A scores and justifications
        "Model_A_Citation_Score", "Model_A_Citation_Justification",
        "Model_A_Relevance_Score", "Model_A_Relevance_Justification",
        "Model_A_ReAct_Performance_Thought_Score", "Model_A_ReAct_Performance_Thought_Justification",
        "Model_A_RAG_Retrieval_Observation_Score", "Model_A_RAG_Retrieval_Observation_Justification",
        "Model_A_Information_Integration_Score", "Model_A_Information_Integration_Justification",
        # Model B scores and justifications
        "Model_B_Citation_Score", "Model_B_Citation_Justification",
        "Model_B_Relevance_Score", "Model_B_Relevance_Justification",
        "Model_B_ReAct_Performance_Thought_Score", "Model_B_ReAct_Performance_Thought_Justification",
        "Model_B_RAG_Retrieval_Observation_Score", "Model_B_RAG_Retrieval_Observation_Justification",
        "Model_B_Information_Integration_Score", "Model_B_Information_Integration_Justification",
        # Error tracking
        "Evaluation_Error"
    ]
    
    results = []
    
    # Process each row with progress bar
    print("\nEvaluating responses...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        question = row["Question"]
        model_a_response = row["Model_A_Response"]
        model_b_response = row["Model_B_Response"]
        
        # Initialize result row with original data
        result_row = {
            "Question": question,
            "Model_A_Response": model_a_response,
            "Model_B_Response": model_b_response,
        }
        
        # Call judge model
        evaluation = call_judge_model(
            client,
            question,
            model_a_response,
            model_b_response,
            model_name=model_name,
            is_azure=is_azure
        )
        
        if evaluation is None:
            # If evaluation failed, record error and set all scores to None
            result_row["Evaluation_Error"] = "Failed to get valid evaluation from judge model"
            for col in output_columns:
                if col not in result_row:
                    result_row[col] = None
        else:
            # Extract scores for Model A
            model_a_scores = extract_scores_from_evaluation(evaluation, "model_a_evaluation")
            result_row["Model_A_Citation_Score"] = model_a_scores["citation_score"]
            result_row["Model_A_Citation_Justification"] = model_a_scores["citation_justification"]
            result_row["Model_A_Relevance_Score"] = model_a_scores["relevance_score"]
            result_row["Model_A_Relevance_Justification"] = model_a_scores["relevance_justification"]
            result_row["Model_A_ReAct_Performance_Thought_Score"] = model_a_scores["react_performance_thought_score"]
            result_row["Model_A_ReAct_Performance_Thought_Justification"] = model_a_scores["react_performance_thought_justification"]
            result_row["Model_A_RAG_Retrieval_Observation_Score"] = model_a_scores["rag_retrieval_observation_score"]
            result_row["Model_A_RAG_Retrieval_Observation_Justification"] = model_a_scores["rag_retrieval_observation_justification"]
            result_row["Model_A_Information_Integration_Score"] = model_a_scores["information_integration_score"]
            result_row["Model_A_Information_Integration_Justification"] = model_a_scores["information_integration_justification"]
            
            # Extract scores for Model B
            model_b_scores = extract_scores_from_evaluation(evaluation, "model_b_evaluation")
            result_row["Model_B_Citation_Score"] = model_b_scores["citation_score"]
            result_row["Model_B_Citation_Justification"] = model_b_scores["citation_justification"]
            result_row["Model_B_Relevance_Score"] = model_b_scores["relevance_score"]
            result_row["Model_B_Relevance_Justification"] = model_b_scores["relevance_justification"]
            result_row["Model_B_ReAct_Performance_Thought_Score"] = model_b_scores["react_performance_thought_score"]
            result_row["Model_B_ReAct_Performance_Thought_Justification"] = model_b_scores["react_performance_thought_justification"]
            result_row["Model_B_RAG_Retrieval_Observation_Score"] = model_b_scores["rag_retrieval_observation_score"]
            result_row["Model_B_RAG_Retrieval_Observation_Justification"] = model_b_scores["rag_retrieval_observation_justification"]
            result_row["Model_B_Information_Integration_Score"] = model_b_scores["information_integration_score"]
            result_row["Model_B_Information_Integration_Justification"] = model_b_scores["information_integration_justification"]
            
            result_row["Evaluation_Error"] = ""
        
        results.append(result_row)
    
    # Create output DataFrame and write to CSV
    output_df = pd.DataFrame(results, columns=output_columns)
    output_df.to_csv(output_file, index=False)
    
    print(f"\n✓ Evaluation complete!")
    print(f"✓ Results written to: {output_file}")
    print(f"✓ Processed {len(results)} rows")
    
    # Print summary statistics
    errors = output_df[output_df["Evaluation_Error"] != ""].shape[0]
    if errors > 0:
        print(f"⚠ Warning: {errors} rows had evaluation errors")


def main():
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python llm_judge_evaluator.py my_test_data.csv
    python llm_judge_evaluator.py /path/to/input.csv

Setup for Azure OpenAI (GPT-5):
    1. Install dependencies: pip install -r requirements.txt
    2. Set environment variables:
       export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/'
       export AZURE_OPENAI_API_KEY='your-api-key'
       export MODEL_NAME='gpt-5'  # or your deployment name
       export AZURE_OPENAI_API_VERSION='2024-08-01-preview'  # optional, defaults to this
    3. Run the script with your input CSV file

Setup for Standard OpenAI:
    1. Install dependencies: pip install -r requirements.txt
    2. Set API key: export OPENAI_API_KEY='your-api-key-here'
    4. Run the script with your input CSV file
        """
    )
    
    parser.add_argument(
        "input_csv",
        help="Path to the input CSV file (no header, columns: Question, Model A Response, Model B Response)"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="evaluation_output.csv",
        help="Path to the output CSV file (default: evaluation_output.csv)"
    )
    
    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=None,
        help="Limit processing to first N rows (useful for testing to avoid high API costs)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("LLM-as-a-Judge Evaluation Script")
    print("=" * 70)
    
    process_csv(args.input_csv, args.output, limit_rows=args.limit)


if __name__ == "__main__":
    main()

