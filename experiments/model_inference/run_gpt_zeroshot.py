"""Run zero-shot GPT inference on a prompt CSV file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a highly intelligent question-answering bot with profound knowledge "
    "of causal learning and causal inference."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="CSV file containing prompts.",
    )
    parser.add_argument("--output-file", type=Path, default=Path("results/asia_gpt_4.csv"))
    parser.add_argument("--model", default="gpt-4")
    parser.add_argument("--prompt-column", default="prompt")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE"))
    parser.add_argument("--max-tokens", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Please set {args.api_key_env} before calling the API.")
    try:
        import openai
    except ImportError as exc:
        raise SystemExit("Install openai first: pip install openai") from exc
    openai.api_key = api_key
    if args.api_base:
        openai.api_base = args.api_base

    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Install pandas first: pip install pandas") from exc

    questions_df = pd.read_csv(args.input_file)
    if args.prompt_column not in questions_df.columns:
        raise ValueError(f"{args.input_file} must contain a '{args.prompt_column}' column")
    questions = questions_df[args.prompt_column].dropna().astype(str).tolist()

    answers = []
    for idx, question in enumerate(questions, start=1):
        try:
            response = openai.ChatCompletion.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                max_tokens=args.max_tokens,
            )
            answer = response.choices[0].message["content"].strip()
        except Exception as exc:
            answer = f"ERROR: {exc}"
        answers.append(answer)
        print(f"{idx}/{len(questions)}")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Question": questions, "Answer": answers}).to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
