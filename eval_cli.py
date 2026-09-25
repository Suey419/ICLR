import argparse

from eval_runner import evaluate_generation_file
from eval_utils import (
    DEFAULT_EVAL_LOG_TEMPLATE,
    DEFAULT_GENERATION_FILE,
    DEFAULT_REPORT_DIR,
    DEFAULT_REPORT_TEMPLATE,
    DEFAULT_SCORE_TEMPLATE,
)
from llm_clients import DEFAULT_BASE_URL


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PULSAR six-output generation JSON.")
    parser.add_argument("--generation_file", type=str, default=DEFAULT_GENERATION_FILE)
    parser.add_argument("--report_dir", type=str, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--report_template", type=str, default=DEFAULT_REPORT_TEMPLATE)
    parser.add_argument("--score_template", type=str, default=DEFAULT_SCORE_TEMPLATE)
    parser.add_argument("--eval_log_template", type=str, default=DEFAULT_EVAL_LOG_TEMPLATE)
    parser.add_argument("--force_recompute", action="store_true")
    parser.add_argument("--judge_mode", type=str, default="llm", choices=["llm", "exact"])
    parser.add_argument("--workers", type=int, default=1, help="Number of cases to evaluate in parallel for one generation file.")
    parser.add_argument("--max_cases", type=int, default=0, help="Evaluate only the first N cases from the generation file. 0 means all cases.")

    parser.add_argument("--model", type=str, default="deepseek", choices=["openai", "gemini", "deepseek", "claude"])
    parser.add_argument("--sk", type=str, default="", help="OpenAI-compatible API key. Alias for --apikey.")
    parser.add_argument("--apikey", type=str, default="", help="OpenAI-compatible API key.")
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL, help="OpenAI-compatible API base URL.")
    parser.add_argument("--openai_apikey", type=str, default="")
    parser.add_argument("--openai_base_url", type=str, default="")
    parser.add_argument("--openai_model", type=str, default="gpt-4o")
    parser.add_argument("--gemini_apikey", type=str, default="")
    parser.add_argument("--gemini_base_url", type=str, default="")
    parser.add_argument("--gemini_model", type=str, default="gemini-2.0-flash")
    parser.add_argument("--claude_apikey", type=str, default="")
    parser.add_argument("--claude_base_url", type=str, default="")
    parser.add_argument("--claude_model", type=str, default="claude-3-7-sonnet-20250219")
    parser.add_argument("--deepseek_apikey", type=str, default="")
    parser.add_argument("--deepseek_base_url", type=str, default="")
    parser.add_argument("--deepseek_model", type=str, default="deepseek-r1-250120")
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate_generation_file(args)
