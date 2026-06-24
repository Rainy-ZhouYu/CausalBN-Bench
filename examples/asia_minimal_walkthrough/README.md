# Asia Minimal Walkthrough

This folder is a cleaned Asia-only example implementation extracted from the uploaded `CausalBN-Bench-main.zip` and reorganized to mirror the paper task names.

What is included:

- Asia-only prompt/question/label examples.
- Small inference scripts for causality, relation, causal skeleton, causal strength, background knowledge, and structured-data prompts.
- Evaluation scripts from the zip.
- `figures/Framework.png` and the original zip README for reference.

What was cleaned:

- Removed `__pycache__`, `.pyc`, and large pickle cache files.
- Removed non-Asia raw BIF duplicates from the example copy.
- Replaced hard-coded OpenAI keys with `OPENAI_API_KEY` environment-variable reads.
- Replaced one local Windows result path with relative `results/` output paths.

Normalized layout:

- `tasks/correlation_identification/`
- `tasks/causal_skeleton_identification/`
- `tasks/causality_identification/direct_causality_prompts/`
- `tasks/causality_identification/variable_name_prompts/`
- `prompt_formats/background_knowledge/`
- `appendix/variable_refactorization/`
- `appendix/causal_strength/`
- `source_networks_and_labels/`
- `inference_examples/`
- `evaluation/`

Typical usage:

```bash
cd examples/asia_minimal_walkthrough
export OPENAI_API_KEY="..."
python inference_examples/causality_identification/Causal_inference.py
```

Some scripts still expect local model checkpoint paths for open-source LLM inference. Edit `locationLlamaHF` / model-path variables as needed.
