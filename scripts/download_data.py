"""Download the released CausalBN-Bench dataset from Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_DATASET_ID = "IEEERainy/CausalBN-Bench"
DEFAULT_TARGET_DIR = Path("data") / "CausalBN-Bench"

TASK_PATHS = {
    "correlation_identification": "data/main_tasks/correlation_identification",
    "causal_skeleton_identification": "data/main_tasks/causal_skeleton_identification",
    "causality_direct_prompts": "data/main_tasks/causality_identification/direct_causality_prompts",
    "causality_variable_name_prompts": "data/main_tasks/causality_identification/variable_name_prompts",
    "background_knowledge": "data/prompt_formats/background_knowledge",
    "variable_refactorization": "data/appendix/variable_refactorization",
    "causal_strength_ranking": "data/appendix/causal_strength/ranking",
    "source_networks_and_labels": "data/source_networks_and_labels",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID, help="Hugging Face dataset repo id.")
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR, help="Local download directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub first: pip install huggingface_hub") from exc

    args.target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.dataset_id,
        repo_type="dataset",
        local_dir=str(args.target_dir),
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded {args.dataset_id} to {args.target_dir.resolve()}")
    print()
    print("Task-aligned data paths:")
    for name, rel_path in TASK_PATHS.items():
        print(f"- {name}: {args.target_dir / rel_path}")


if __name__ == "__main__":
    main()
