# Model Forensics with Causal Verification

Gradient and Fisher based concept-neuron localization for LLMs, extended into a causal test: given a model finetuned into emergent misalignment, can the units responsible be localized and patched back to their pre-finetuning values to undo the behavior. Written up for a MATS 12.0 application (Neel Nanda stream).

Builds on prior work on gradient-based concept localization and concept graphs for text-to-image diffusion models ([arXiv:2602.07919](https://arxiv.org/abs/2602.07919), [arXiv:2607.03397](https://arxiv.org/pdf/2607.03397)), ported to LLMs and extended with a causal patching experiment on emergent misalignment ([Betley et al., arXiv:2502.17424](https://arxiv.org/abs/2502.17424)).

## Running it

Data prep, fine-tuning, and the causal pipeline all live in `scripts/`:

```
python scripts/prepare_data.py
python scripts/finetune_lora.py --data-path data/insecure_train.jsonl --output-name <name>
python scripts/sanity_check.py
python scripts/run_causal_pipeline.py --finetuned-model <model> --probe-holdout-path <path>
python scripts/bootstrap_ci.py --report outputs/run1/causal_pipeline_report.json
```

All HF downloads are cached under `/data/m24/.cache`; all compute runs in the `co_align` conda environment.

## Report

The full write-up, with methodology, results, limitations, and appendix, is in `reports/mats_application_report.md`.
