# Code Structure

This repository is organized around the benchmark tasks and prompt settings used in the CausalBN-Bench paper. Large generated data files are hosted on Hugging Face Datasets.

Dataset link: https://huggingface.co/datasets/IEEERainy/CausalBN-Bench

## Normalized Source Layout

| Area | Path | Purpose |
| --- | --- | --- |
| Main task | `tasks/correlation_identification/` | Scripts for pairwise correlation or relatedness prompts. |
| Main task | `tasks/causal_skeleton_identification/` | Scripts, prompts, labels, and BIF/network files for causal skeleton recovery. |
| Main task | `tasks/causality_identification/direct_causality_prompts/` | Scripts for direct cause-effect prompts. |
| Main task | `tasks/causality_identification/variable_name_prompts/` | Scripts for variable-name causality prompts. |
| Prompt format | `prompt_formats/background_knowledge/` | Background-knowledge collection and prompt construction. |
| Appendix | `appendix/variable_refactorization/` | Modified-variable-name and refactorization prompts. |
| Appendix | `appendix/causal_strength/` | Causal-strength ranking prompt scripts. |
| Appendix | `appendix/original_prompt_baseline/` | Original prompt baseline scripts retained for reproducibility. |
| Data construction | `source_networks_and_labels/` | Source Bayesian-network files and generated labels. |
| Evaluation | `evaluation/` | Evaluation and result-table utilities. |
| Experiments | `experiments/model_inference/` | LLM inference scripts for closed-source and open-source models. |
| Experiments | `experiments/sample_outputs/` | Small sample outputs and result files. |
| Baselines | `baselines/finetuning/` | Encoder-based finetuning and evaluation utilities. |
| Examples | `examples/asia_minimal_walkthrough/` | Self-contained Asia minimal walkthrough aligned with the normalized task names. |
| Utilities | `scripts/download_data.py` | Download the Hugging Face dataset. |

## Paper Task Mapping

| Paper benchmark component | Source-code path | Hugging Face dataset path |
| --- | --- | --- |
| Correlation identification | `tasks/correlation_identification/` | `data/main_tasks/correlation_identification/` |
| Causal skeleton identification | `tasks/causal_skeleton_identification/` | `data/main_tasks/causal_skeleton_identification/` |
| Causality identification, direct-cause prompts | `tasks/causality_identification/direct_causality_prompts/` | `data/main_tasks/causality_identification/direct_causality_prompts/` |
| Causality identification, variable-name prompts | `tasks/causality_identification/variable_name_prompts/` | `data/main_tasks/causality_identification/variable_name_prompts/` |
| Background-knowledge prompt format | `prompt_formats/background_knowledge/` | `data/prompt_formats/background_knowledge/` |
| Variable refactorization / modified variable names | `appendix/variable_refactorization/` | `data/appendix/variable_refactorization/` |
| Causal-strength appendix exploration | `appendix/causal_strength/` | `data/appendix/causal_strength/ranking/` |
| Source networks and labels | `source_networks_and_labels/` | `data/source_networks_and_labels/` |

## Downloading Data

From the repository root:

```bash
pip install -U huggingface_hub
python scripts/download_data.py
```

By default this downloads `IEEERainy/CausalBN-Bench` into `data/CausalBN-Bench/`.

## Notes

- Generated full benchmark CSV/TXT files are not stored directly in this GitHub source-code package.
- `data/` and `results/` are ignored by `.gitignore` so downloaded data and generated outputs are not accidentally committed.
- The top-level structure is task-aligned for publication; raw historical snapshots are kept outside the Git-tracked release.
- Scripts that call external APIs read credentials from environment variables such as `OPENAI_API_KEY`, `OPENAI_API_BASE`, and `HUGGINGFACE_API_KEY`.
- Some experiment scripts are preserved as research artifacts. Prefer the parameterized scripts in `scripts/`, `experiments/model_inference/`, and the task folders when starting a new run.
