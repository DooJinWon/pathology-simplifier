# Pathology Simplifier

NCSA Ashby Hackathon 2026 project. Final few-shot version from October 17, 2026.

Explains pathology captions in English for early medical students using `google/medgemma-4b-it`, a locally trained PEFT adapter, and five worked examples. The prompt asks the model to preserve findings, uncertainty, and limitations while explaining medical vocabulary.

## Final files

- `try_medgemma_fewshot.py`: interactive GPU inference script.
- `fewshot_instruction.txt`: final instructions and five examples.

## Run

The script expects this project at `~/pathology-simplifier`, a CUDA GPU, and Python packages `torch`, `transformers` (with Gemma 3 support), and `peft`.

It loads all model files locally. Cache the base model and place the trained adapter and tokenizer files in `~/pathology-simplifier/medgemma_simplifier_adapter` before running:

```bash
cd ~/pathology-simplifier
python try_medgemma_fewshot.py
```

Enter a caption on one line; enter `quit` to exit.

Model weights, trained adapter files, datasets, virtual environments, and training checkpoints are not included in this repository. This is an educational prototype; model outputs have not been independently validated.
