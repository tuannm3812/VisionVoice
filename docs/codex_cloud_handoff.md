# Codex Cloud Handoff

Use this note when starting a new Codex task in the cloud for this repository.

## Repository

```text
https://github.com/tuannm3812/VisionVoice
```

Current branch:

```text
main
```

## Current Project State

VisionVoice is an image-captioning assignment using the official VizWiz-Captions validation set only. The repository is organized around:

- `notebooks/01_eda_vizwiz.ipynb`: shared EDA and data preparation.
- `notebooks/02_modeling.ipynb`: one modelling notebook containing both required architectures.
- `report/visionvoice_report.md`: academic report draft.
- `docs/model_results.md`: model-output summary.
- `docs/assignment_alignment.md`: mapping to the assignment phases.

The assignment requires:

- one shared data-preparation notebook,
- one modelling notebook per student,
- at least two architectures in each student's modelling notebook,
- BLEU-1, BLEU-2, BLEU-3, and BLEU-4 evaluation,
- report sections covering dataset, preparation, architectures, discussion, metrics, performance, limitations, recommendations, and contribution statement.

## Architecture Decision

The chosen visual backbone is **ResNet-50**. Do not switch to EfficientNet, ViT, or another encoder family unless explicitly requested.

Current models:

- Model 1: frozen ResNet-50 global feature encoder + LSTM decoder.
- Model 2: ResNet-50 spatial feature encoder + Bahdanau-style attention + LSTMCell decoder.

The current strategy is to keep one modelling notebook:

```text
notebooks/02_modeling.ipynb
```

Do not create a third modelling notebook unless the user explicitly asks.

## Latest Notebook Setup

`notebooks/02_modeling.ipynb` has been rerun on Kaggle:

- Evaluation uses the full internal test split.
- Model 2 reports both greedy and beam-search decoding.
- Baseline metrics save to `baseline_metrics_greedy.json`.
- Attention metrics save to `attention_metrics_greedy.json` and `attention_metrics_beam.json`.
- Notebook outputs are kept for final assignment evidence.

## Current Completed Results

Full internal test split evaluation:

| Model | Decoding | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline ResNet-LSTM | Greedy | 0.5740 | 0.2692 | 0.1281 | 0.0645 |
| Attention ResNet-LSTM | Greedy | 0.6129 | 0.3960 | 0.2519 | 0.1593 |
| Attention ResNet-LSTM | Beam search | 0.6640 | 0.4373 | 0.2897 | 0.1935 |

Interpretation:

- Baseline trains stably but collapses to repeated generic captions in visual inspection.
- Attention improves all BLEU scores.
- Beam search gives the strongest quantitative result.
- Visual inspection still shows repetition and object-detail errors.

## Important Kaggle Paths

Default dataset root:

```text
/kaggle/input/datasets/tuannm3823/vizwiz
```

Expected annotation path:

```text
/kaggle/input/datasets/tuannm3823/vizwiz/annotations/annotations/val.json
```

Expected image folder:

```text
/kaggle/input/datasets/tuannm3823/vizwiz/val/val
```

Kaggle outputs:

```text
/kaggle/working/vision_voice_baseline_best.pth
/kaggle/working/vision_voice_attention_best.pth
/kaggle/working/baseline_metrics_greedy.json
/kaggle/working/attention_metrics_greedy.json
/kaggle/working/attention_metrics_beam.json
```

## Suggested Cloud Codex Prompt

```text
I am continuing the VisionVoice image-captioning assignment. Please read docs/codex_cloud_handoff.md first, then inspect the current repository state. Keep the ResNet-50 architecture decision and keep one modelling notebook. Help polish the final report and submission materials using the full-test BLEU metrics already documented in the repo.
```
