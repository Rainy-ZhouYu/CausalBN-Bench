"""Generate background-knowledge answers with an OpenAI-compatible chat API."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

openai = None

DEFAULT_SYSTEM_PROMPT = (
    "You are a highly intelligent question-answering bot with profound knowledge "
    "of causal inference."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing Knowledge_<dataset>.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "background_knowledge_answers",
        help="Directory where answer CSV files will be written.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["win95pts"],
        help="Dataset names without the Knowledge_ prefix.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-4-1106-preview", "gpt-4", "gpt-3.5-turbo"],
        help="OpenAI-compatible chat model names.",
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("OPENAI_API_BASE"),
        help="Optional OpenAI-compatible API base URL. Defaults to OPENAI_API_BASE.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the API key.",
    )
    parser.add_argument("--max-tokens", type=int, default=1000)
    return parser.parse_args()


def configure_openai(api_key_env: str, api_base: str | None) -> None:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise SystemExit(f"Please set {api_key_env} before calling the API.")
    try:
        import openai as openai_module
    except ImportError as exc:
        raise SystemExit("Install openai first: pip install openai") from exc
    globals()["openai"] = openai_module
    openai_module.api_key = api_key
    if api_base:
        openai_module.api_base = api_base


def answer_questions(questions: list[str], model: str, output_file: Path, max_tokens: int) -> list[str]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    answers: list[str] = []
    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["order", "question", "answer"])
        for idx, question in enumerate(questions, start=1):
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                max_tokens=max_tokens,
            )
            answer = response.choices[0].message["content"].strip()
            answers.append(answer)
            writer.writerow([idx, question, answer])
            print(f"{model} {idx}/{len(questions)}")
    return answers


def main() -> None:
    args = parse_args()
    configure_openai(args.api_key_env, args.api_base)
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Install pandas first: pip install pandas") from exc

    for model in args.models:
        for dataset in args.datasets:
            input_file = args.input_dir / f"Knowledge_{dataset}.csv"
            if not input_file.exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")
            questions_df = pd.read_csv(input_file)
            if "prompt" not in questions_df.columns:
                raise ValueError(f"{input_file} must contain a 'prompt' column")
            questions = questions_df["prompt"].dropna().astype(str).tolist()
            output_file = args.output_dir / f"{model}_{dataset}.csv"
            answers = answer_questions(questions, model, output_file, args.max_tokens)
            qa_pairs = pd.DataFrame({"Question": questions, "Answer": answers})
            qa_pairs.to_csv(args.output_dir / f"{model}_{dataset}_qa_pairs.csv", index=False)


if __name__ == "__main__":
    main()
