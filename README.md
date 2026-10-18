# Pathology Simplifier
## Dual-Track Explainable AI for Histopathology Education

NCSA Ashby Hackathon 2026 project by **Jinwon Doo**. This repository contains the final few-shot caption explanation script and prompt from **October 17, 2026**.

The project explores how AI can help early medical students understand pathology descriptions and the visual findings behind them. The broader concept combines a natural-language explanation with structured morphological evidence from the same tissue patch.

**Current implementation:** a command-line caption explanation tool using `google/medgemma-4b-it`, a local PEFT adapter, and five examples. The image caption generator, morphology classifier, and combined report are proposed extensions.

## Motivation

Expert pathology captions contain terms such as *pleomorphism*, *hyperchromasia*, and *stromal invasion*. A beginner may recognize a diagnostic label without understanding the observations that support it.

The goal is to explain those observations in accessible language while retaining their anatomical context, severity, negation, and uncertainty. An explanation should help a learner understand what a caption says without turning a tentative interpretation into a confirmed diagnosis.

## Overall concept and implementation status

| Component | Purpose | Status in this repository |
| --- | --- | --- |
| PathGen-LLaVA caption generation | Produce an expert caption from a histopathology image patch | Proposed upstream component |
| MedGemma caption explanation | Explain an existing caption for an early medical student | Final inference script published |
| Few-shot guidance | Demonstrate the desired explanatory style and fidelity | Published prompt with five examples |
| MUSK morphology classifier | Predict five morphological features independently | Proposed extension |
| Integrated learning report | Combine the explanation with feature predictions and uncertainty | Proposed extension |

The current script accepts **text captions**, not image files.

### Track 1: Natural-language explanation

The proposed end-to-end language track starts with a PathGen-LLaVA caption. The current tool begins after that step:

1. Read the final instructions and five demonstrations from `fewshot_instruction.txt`.
2. Load the tokenizer from a local adapter directory.
3. Load MedGemma 4B and attach the local PEFT adapter.
4. Append the user's caption to the instructions and apply the tokenizer's chat template.
5. Generate and print an English educational explanation.

The five demonstrations cover breast, glandular/adenoma, colon, prostate, and kidney descriptions. They guide vocabulary explanation and handling of diagnostic uncertainty. They do not establish model performance across those tissues.

### Track 2: Structured morphology evidence

The concept proposes a pretrained MUSK vision encoder followed by a task head with 15 logits, reshaped into five independent three-class predictions.

| Feature | Proposed label | Intended observation |
| --- | --- | --- |
| Abnormal gland architecture | `abnormal_gland_arch` | Irregular gland shape, formation, or arrangement |
| Nuclear atypia | `nuclear_atypia` | Unusual nuclear size, shape, or chromatin appearance |
| Abnormal nuclear organization | `abnormal_nuclear_org` | Disrupted alignment, polarity, or distribution |
| Stromal invasion | `stromal_invasion` | Evidence of abnormal cells or glands entering stroma |
| Intraluminal necrosis | `intraluminal_necrosis` | Necrotic debris within glandular lumens |

The concept uses **−1 = absent, 0 = uncertain or not assessable, +1 = present**. These remain proposed annotation semantics. They require a defined dataset protocol and expert review before training or evaluation. The feature set is especially relevant to gland-forming tumors, so target tissue and disease scope also need to be defined.

No MUSK classifier, image inference code, trained morphology head, or fusion implementation is included here.

## Final files

| File | Contents |
| --- | --- |
| [`try_medgemma_fewshot.py`](try_medgemma_fewshot.py) | Interactive GPU inference script for the final caption explanation workflow |
| [`fewshot_instruction.txt`](fewshot_instruction.txt) | Explanation rules and five input/output demonstrations |

Model weights, the trained adapter, tokenizer assets, datasets, training scripts, and checkpoints are not included in this public repository.

## Explanation design

The final prompt asks the model to:

- Preserve findings, anatomical locations, extent, severity, measurements, and negations.
- Retain uncertainty and limits on what a tissue patch can establish.
- Keep essential medical terms and explain their meaning.
- Distinguish observed morphology from a proposed diagnosis.
- Avoid adding patient facts, causes, alternative diagnoses, prognosis, or treatment advice.
- Attribute diagnostic interpretations to the source caption.
- Keep examples separate so their findings do not transfer to a new case.

For example, an explanation of *hyperchromatic nuclei* should explain darker nuclear staining. It should retain any uncertainty in the original caption about what that observation means.

These are intended behaviors expressed in the prompt, not guarantees established through evaluation.

## Runtime requirements

- A CUDA-capable GPU.
- A Python environment with `torch`, `transformers` supporting `Gemma3ForConditionalGeneration`, and `peft`.
- Access to the `google/medgemma-4b-it` checkpoint under its applicable access conditions.
- The base model already available in the local Hugging Face cache.
- The compatible trained adapter and tokenizer files in `~/pathology-simplifier/medgemma_simplifier_adapter/`.

The script sets `local_files_only=True` for loading. It does not download missing model assets. The repository alone is therefore insufficient to reproduce the trained setup.

The expected project layout is:

```text
~/pathology-simplifier/
├── try_medgemma_fewshot.py
├── fewshot_instruction.txt
└── medgemma_simplifier_adapter/   # supplied separately
    └── compatible adapter and tokenizer assets
```

The project path is currently fixed to `Path.home() / "pathology-simplifier"`. Update that assignment if you store the project elsewhere.

## Run the final version

```bash
git clone https://github.com/DooJinWon/pathology-simplifier.git ~/pathology-simplifier
cd ~/pathology-simplifier
# Activate your prepared GPU Python environment and supply the local model assets.
python try_medgemma_fewshot.py
```

Enter a caption on one line at the `Medical caption:` prompt. The script prints the explanation beneath `=== Simplified caption ===`. Enter `quit` to exit. Empty input is ignored.

### Inference configuration

| Setting | Current value |
| --- | --- |
| Base checkpoint | `google/medgemma-4b-it` |
| Adaptation | Local adapter loaded with `PeftModel.from_pretrained` |
| Device | CUDA |
| Model dtype | `torch.bfloat16` |
| Attention implementation | `sdpa` |
| Generation | `do_sample=False` |
| Maximum generated tokens | 768 |
| Cache | Enabled |
| Output | English text in the terminal |

The released script does not record the adapter's training hyperparameters, dataset provenance, package versions, or measured runtime.

## Evaluation plan

The concept presentation proposes evaluation at several levels:

| Area | Planned checks |
| --- | --- |
| Caption fidelity | Preservation of findings, negation, uncertainty, and factual consistency |
| Morphology prediction | Per-feature macro-F1, confusion matrices, calibration, and abstention behavior |
| Educational usefulness | Student comprehension before and after use, feature recognition, clarity, and usefulness |
| Expert review | Pathologist agreement, review of explanations, and errors by tissue or subgroup |

This repository reports **no quantitative accuracy, educational improvement, or clinical validation results**. The examples in the concept presentation are illustrative rather than evaluated model predictions.

## Next steps

- Document the adapter training setup and data provenance.
- Evaluate the caption explanation workflow against expert-reviewed references.
- Define the target tissue scope and morphology annotation protocol.
- Train and evaluate the proposed MUSK classification head.
- Combine both tracks into a report that exposes uncertainty.
- Assess the report with students and pathology reviewers.

## Scope and limitations

This is an educational research prototype. A generated explanation depends on the quality of its source caption and may contain errors. An uncertain feature must remain distinct from an absent feature, and one patch may not establish the extent of invasion. The output is not a validated diagnostic assessment.

## Concept references

The accompanying October 2026 concept presentation identifies the following foundations. Their inclusion describes the proposed architecture and does not imply every component has been integrated.

- [PathGen-1.6M / PathGen-LLaVA](https://arxiv.org/abs/2407.00203)
- [MUSK foundation model](https://doi.org/10.1038/s41586-024-08378-w)
- [Google Health MedGemma](https://github.com/Google-Health/medgemma)

## Author

[Jinwon Doo](https://github.com/DooJinWon) · NCSA Ashby Hackathon 2026
