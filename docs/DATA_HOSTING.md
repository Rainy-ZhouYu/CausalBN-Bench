# Data Hosting Guidance

Use GitHub for source code and Hugging Face Datasets for large benchmark assets.

## Released Dataset

The full CausalBN-Bench dataset is available at:

https://huggingface.co/datasets/IEEERainy/CausalBN-Bench

This source-code release intentionally excludes the large generated benchmark files. Use `scripts/download_data.py` to download the Hugging Face dataset into `data/CausalBN-Bench/`.

## Recommended Split

- GitHub: source code, README, requirements, small examples, download script, and documentation.
- Hugging Face Datasets: generated question/prompt files, labels, background-knowledge tables, causal-strength ranking prompts, source Bayesian-network files, manifests, and checksums.

## Hugging Face Dataset Layout

```text
CausalBN-Bench/
  README.md
  data/
    main_tasks/
      correlation_identification/
      causal_skeleton_identification/
      causality_identification/
        direct_causality_prompts/
        variable_name_prompts/
    prompt_formats/
      background_knowledge/
    appendix/
      variable_refactorization/
        questions/
        labels/
        nested/
      causal_strength/
        ranking/
    source_networks_and_labels/
  metadata/
    file_manifest.csv
    checksums.sha256
    excluded_pickles_manifest.csv
    summary_by_section.csv
    file_counts_by_extension.csv
```

## Task Mapping

| Paper benchmark component | Hugging Face dataset path |
| --- | --- |
| Correlation identification | `data/main_tasks/correlation_identification/` |
| Causal skeleton identification | `data/main_tasks/causal_skeleton_identification/` |
| Causality identification | `data/main_tasks/causality_identification/direct_causality_prompts/` and `data/main_tasks/causality_identification/variable_name_prompts/` |
| Background-knowledge prompt format | `data/prompt_formats/background_knowledge/` |
| Variable refactorization appendix | `data/appendix/variable_refactorization/` |
| Causal-strength appendix | `data/appendix/causal_strength/ranking/` |
| Source networks and labels | `data/source_networks_and_labels/` |
