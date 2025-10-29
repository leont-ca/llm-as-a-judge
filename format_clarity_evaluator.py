#!/usr/bin/env python3
"""
Format Clarity Evaluator - LLM-as-a-Judge for Stylistic Similarity

This script evaluates how closely the formatting and style of Claude 4.5 Sonnet's
responses match Claude 3.5 Sonnet's responses (the baseline/golden standard).

It reads raw ReAct logs from a CSV file, parses the Final Answer sections,
and uses an LLM judge (GPT-5 or GPT-4-turbo) to score format similarity on a 1-5 scale.

Usage:
    python format_clarity_evaluator.py <input_csv_file>

Requirements:
    - For Azure OpenAI: Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and MODEL_NAME
    - For Standard OpenAI: Set OPENAI_API_KEY (and optionally MODEL_NAME)
    - Input CSV must have 3 columns (no header): Question, Claude 3.5 Log, Claude 4.5 Log
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from typing import Dict, Any, Optional, Tuple

import pandas as pd
from openai import OpenAI, AzureOpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


# Format Clarity Judge System Prompt (embedded as per requirements)
JUDGE_SYSTEM_PROMPT = """You are a meticulous AI evaluator specializing in text formatting and stylistic consistency.

**Your Task:**
Your sole task is to evaluate how closely the format of the "Claude 4.5 Sonnet Answer" matches the format of the "Claude 3.5 Sonnet Answer". The "Claude 3.5 Sonnet Answer" is the golden standard, and you are scoring the "Claude 4.5 Sonnet Answer" on its ability to mimic that style.

You must provide a score from 1 to 5, using the detailed rubric below. You must return your evaluation as a single, valid JSON object and nothing else.

### **Detailed Scoring Rubric: Format/Clarity (回答の形式)**

*Focus: Compare the "Claude 4.5 Sonnet Answer" directly against the "Claude 3.5 Sonnet Answer". Evaluate the similarity in markdown (headings, bolding), list structures (bullets/numbers), and overall logical separation of ideas.*

- **5 (Excellent - Near Identical):**
    The "Claude 4.5 Sonnet" response uses virtually identical formatting to the "Claude 3.5 Sonnet" response. It effectively mirrors the use of markdown headings (e.g., `##`), bold text (`**...**`), bullet points (`-`), and logical paragraph breaks. The structure is a clear match.

- **4 (Good - Mostly Similar):**
    The "Claude 4.5 Sonnet" response follows the general structure (e.g., headings, lists) of the "Claude 3.5 Sonnet" response but has minor, non-critical deviations. For example, it might use bullets where the 3.5 model used numbers, or miss a single bolded word, but the overall style is clearly aligned.

- **3 (Acceptable - Some Similarities):**
    The "Claude 4.5 Sonnet" response shows some structural similarities but also has significant differences. For example, it might use lists where the 3.5 model used paragraphs, or be missing all the headings that the 3.5 model used. The style is noticeably different but not completely unrelated.

- **2 (Poor - Mostly Different):**
    The "Claude 4.5 Sonnet" response is mostly different. It might use some basic formatting (like paragraph breaks), but it does not resemble the structure or style (e.g., use of headings and lists) of the "Claude 3.5 Sonnet" response.

- **1 (Very Poor - No Resemblance):**
    The "Claude 4.5 Sonnet" response format is completely different. For example, the 3.5 model provided a structured, multi-part answer, and the 4.5 model returned a single, dense block of text (or vice-versa).

---
**Required JSON Output Format:**
{
  "format_clarity_evaluation": {
    "score": <score_from_1_to_5>,
    "justification": "<A brief justification for your score, detailing why the 4.5 model's format matches or differs from the 3.5 model's format.>"
  }
}"""


def parse_final_answer(raw_log: str) -> str:
    """
    Parse the raw ReAct log to extract only the Final Answer text.
    
    This function looks for the "## ✅ Final Answer 回答" heading and extracts
    all text that follows it until it hits "## 🔗 URLs" section or end of string.
    
    Args:
        raw_log: The complete raw ReAct log string
    
    Returns:
        The extracted final answer text, or the original log if parsing fails
    """
    if not isinstance(raw_log, str):
        return str(raw_log) if raw_log is not None else ""
    
    # Pattern to match "## ✅ Final Answer 回答" and capture content until "## 🔗 URLs" or end
    # Using non-greedy match (.+?) and lookahead to stop at URLs section
    # The URLs section may be formatted as "## 🔗 URLs URL" or just "## 🔗 URLs"
    pattern = r'##\s*✅\s*Final\s+Answer\s*回答\s*(.+?)(?=##\s*🔗\s*URLs(?:\s+URL)?|$)'
    
    # Use DOTALL flag to make . match newlines
    match = re.search(pattern, raw_log, re.DOTALL | re.IGNORECASE)
    
    if match:
        # Extract the final answer text and strip leading/trailing whitespace
        final_answer = match.group(1).strip()
        return final_answer
    else:
        # If pattern not found, try alternative patterns
        # Sometimes the emoji might be missing or the format slightly different
        alt_patterns = [
            r'##\s*Final\s+Answer\s*回答\s*(.+?)(?=##\s*🔗\s*URLs(?:\s+URL)?|$)',
            r'##\s*✅\s*Final\s+Answer\s*(.+?)(?=##\s*🔗\s*URLs(?:\s+URL)?|$)',
            r'Final\s+Answer\s*回答\s*[:\n]\s*(.+?)(?=##\s*🔗\s*URLs(?:\s+URL)?|$)',
        ]
        
        for alt_pattern in alt_patterns:
            match = re.search(alt_pattern, raw_log, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If still no match, return a warning message or the original log
        print("⚠️  Warning: Could not parse final answer from log. Using full log.", file=sys.stderr)
        return raw_log


def create_user_prompt(question: str, claude_35_answer: str, claude_45_answer: str) -> str:
    """
    Create the user prompt that will be sent to the judge model.
    
    Args:
        question: The original user question
        claude_35_answer: Parsed final answer from Claude 3.5 Sonnet (baseline)
        claude_45_answer: Parsed final answer from Claude 4.5 Sonnet (candidate)
    
    Returns:
        Formatted prompt string
    """
    return f"""Please evaluate how closely the formatting and style of the Claude 4.5 Sonnet Answer matches the Claude 3.5 Sonnet Answer.

**Question:**
{question}

**Claude 3.5 Sonnet Answer (Golden Standard):**
{claude_35_answer}

**Claude 4.5 Sonnet Answer (To Be Evaluated):**
{claude_45_answer}

Provide your evaluation as a JSON object following the specified format."""


def call_judge_model(
    client,
    question: str,
    claude_35_answer: str,
    claude_45_answer: str,
    model_name: str = "gpt-4-turbo",
    is_azure: bool = False,
    max_retries: int = 3,
    retry_delay: int = 2
) -> Optional[Dict[str, Any]]:
    """
    Call the OpenAI API to evaluate the format similarity.
    
    Args:
        client: OpenAI or AzureOpenAI client instance
        question: The original user question
        claude_35_answer: Parsed final answer from Claude 3.5 Sonnet
        claude_45_answer: Parsed final answer from Claude 4.5 Sonnet
        model_name: Model name or Azure deployment name
        is_azure: Whether using Azure OpenAI (affects parameter naming)
        max_retries: Maximum number of retry attempts on failure
        retry_delay: Delay in seconds between retries
    
    Returns:
        Parsed JSON response from the judge model, or None if all retries fail
    """
    user_prompt = create_user_prompt(question, claude_35_answer, claude_45_answer)
    
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
            if "format_clarity_evaluation" not in evaluation:
                raise ValueError("Response missing required 'format_clarity_evaluation' key")
            
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


def extract_scores_from_evaluation(evaluation: Dict[str, Any]) -> Tuple[Optional[int], str]:
    """
    Extract score and justification from the evaluation JSON.
    
    Args:
        evaluation: The full evaluation JSON object
    
    Returns:
        Tuple of (score, justification)
    """
    format_eval = evaluation.get("format_clarity_evaluation", {})
    score = format_eval.get("score", None)
    justification = format_eval.get("justification", "")
    
    return score, justification


def process_csv(input_file: str, output_file: str = "format_clarity_output.csv", limit_rows: Optional[int] = None) -> None:
    """
    Main processing function that reads the input CSV, parses logs, evaluates format similarity,
    and writes the results to the output CSV.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to the output CSV file (default: format_clarity_output.csv)
        limit_rows: Optional limit on number of rows to process (for cost control)
    """
    # Check if using Azure OpenAI or standard OpenAI
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    model_name = os.getenv("MODEL_NAME", "gpt-4-turbo")  # Default to gpt-4-turbo, can use "gpt-5" for Azure GPT-5
    
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
        # The CSV has NO header row according to requirements
        # Columns: Question, Claude 3.5 Log, Claude 4.5 Log
        df = pd.read_csv(input_file, header=None, names=["Question", "Claude_35_Raw_Log", "Claude_45_Raw_Log"])
        
        # Check if first row contains header-like text and skip it if so
        # Common header patterns: "Questions", "Question", "Claude 3.5", etc.
        if len(df) > 0:
            first_row_question = str(df.iloc[0]["Question"]).strip().lower()
            if first_row_question in ["question", "questions", "q", "query", "queries"]:
                print("⚠️  Detected header row in input CSV. Skipping first row.")
                df = df.iloc[1:].reset_index(drop=True)
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
        print(f"⚠️  WARNING: This will make {len(df)} API calls to {model_name}")
        print(f"⚠️  Estimated cost: ${len(df) * 0.05:.2f} - ${len(df) * 0.20:.2f} (rough estimate)")
        
        # Prompt for confirmation if processing many rows
        if len(df) > 10:
            try:
                response = input(f"\n🤔 Proceed with {len(df)} API calls? [y/N]: ").strip().lower()
                if response != 'y' and response != 'yes':
                    print("Cancelled. Use -n flag to test with fewer rows: python format_clarity_evaluator.py input.csv -n 5")
                    sys.exit(0)
            except (KeyboardInterrupt, EOFError):
                print("\nCancelled.")
                sys.exit(0)
    
    # Prepare output columns
    output_columns = [
        "Question",
        "Claude_3.5_Final_Answer",
        "Claude_4.5_Final_Answer",
        "Format_Clarity_Score",
        "Format_Clarity_Justification",
        "Evaluation_Error"
    ]
    
    results = []
    
    # Process each row with progress bar
    print("\nParsing logs and evaluating format similarity...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        question = row["Question"]
        claude_35_raw_log = row["Claude_35_Raw_Log"]
        claude_45_raw_log = row["Claude_45_Raw_Log"]
        
        # Parse the final answers from the raw logs
        claude_35_final_answer = parse_final_answer(claude_35_raw_log)
        claude_45_final_answer = parse_final_answer(claude_45_raw_log)
        
        # Initialize result row with parsed data
        result_row = {
            "Question": question,
            "Claude_3.5_Final_Answer": claude_35_final_answer,
            "Claude_4.5_Final_Answer": claude_45_final_answer,
        }
        
        # Call judge model
        evaluation = call_judge_model(
            client,
            question,
            claude_35_final_answer,
            claude_45_final_answer,
            model_name=model_name,
            is_azure=is_azure
        )
        
        if evaluation is None:
            # If evaluation failed, record error and set score to None
            result_row["Format_Clarity_Score"] = None
            result_row["Format_Clarity_Justification"] = ""
            result_row["Evaluation_Error"] = "Failed to get valid evaluation from judge model"
        else:
            # Extract score and justification
            score, justification = extract_scores_from_evaluation(evaluation)
            result_row["Format_Clarity_Score"] = score
            result_row["Format_Clarity_Justification"] = justification
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
        print(f"⚠️  Warning: {errors} rows had evaluation errors")
    
    # Calculate average score (excluding None values)
    valid_scores = output_df["Format_Clarity_Score"].dropna()
    if len(valid_scores) > 0:
        avg_score = valid_scores.mean()
        print(f"\n📊 Average Format Clarity Score: {avg_score:.2f}/5.0")
        print(f"📊 Score Distribution:")
        print(output_df["Format_Clarity_Score"].value_counts().sort_index())


def main():
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(
        description="Format Clarity Evaluator - Compare formatting similarity between Claude 3.5 and 4.5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python format_clarity_evaluator.py test_5_rows.csv
    python format_clarity_evaluator.py /path/to/input.csv -o my_format_results.csv
    python format_clarity_evaluator.py input.csv -n 5  # Test with first 5 rows only

Input CSV Format:
    - NO header row
    - Column A: Question
    - Column B: Claude 3.5 Sonnet answer (Full raw ReAct log)
    - Column C: Claude 4.5 Sonnet answer (Full raw ReAct log)

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
    3. Optionally set model: export MODEL_NAME='gpt-4-turbo'  # default
    4. Run the script with your input CSV file

How It Works:
    1. Reads the input CSV (no header row)
    2. Parses each raw ReAct log to extract only the "Final Answer" text
    3. Calls GPT-5 (or GPT-4-turbo) to score format similarity (1-5 scale)
    4. Writes results to output CSV with parsed answers and scores
        """
    )
    
    parser.add_argument(
        "input_csv",
        help="Path to the input CSV file (no header, columns: Question, Claude 3.5 Log, Claude 4.5 Log)"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="format_clarity_output.csv",
        help="Path to the output CSV file (default: format_clarity_output.csv)"
    )
    
    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=None,
        help="Limit processing to first N rows (useful for testing to avoid high API costs)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Format Clarity Evaluator - LLM-as-a-Judge")
    print("Comparing Claude 4.5 Sonnet formatting against Claude 3.5 Sonnet")
    print("=" * 70)
    
    process_csv(args.input_csv, args.output, limit_rows=args.limit)


if __name__ == "__main__":
    main()

