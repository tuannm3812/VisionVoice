# VisionVoice Model Results

This document summarizes the Kaggle training artifacts and observed notebook outputs for the current VisionVoice experiments.

## Dataset and Run Setup

Both model notebooks use the cleaned VizWiz validation split as a project dataset:

- Raw annotations: 38,750
- Clean annotations after filtering rejected/precanned captions: 33,145
- Usable images after filtering: 7,542
- Train split: 6,033 images / 26,474 captions
- Validation split: 1,509 images / 6,671 captions
- Vocabulary size used by model notebooks: 6,439 tokens including special tokens
- Batch size: 32
- Epochs: 10
- Device observed in baseline run: CUDA

## Artifacts

The trained checkpoints are not committed to git because model weights are large binary artifacts.

| Artifact | Local file | Size | SHA256 |
| --- | --- | ---: | --- |
| Baseline ResNet-LSTM | `vision_voice_baseline_best.pth` | 116.90 MB | `086D9FDBC629742D7D13EA80E7BDFA0DCC8580C874FF77EEF29B973447436906` |
| Attention ResNet-LSTM | `vision_voice_attention_best.pth` | 143.91 MB | `1234340BFBD65FE209672A961A2235C51A8E88779EFAD48E2CA0BB4BF59664BB` |

## Baseline Model

Architecture:

- Frozen pretrained ResNet-50 visual backbone
- Learned image-feature projection
- LSTM decoder with teacher forcing
- Greedy decoding during inference

Parameter audit:

| Metric | Value |
| --- | ---: |
| Total parameters | 30,561,639 |
| Trainable parameters | 7,053,607 |
| Frozen parameters | 23,508,032 |
| Frozen backbone ratio | 76.9% |

Training history:

| Epoch | Train loss | Validation loss |
| ---: | ---: | ---: |
| 1 | 5.0863 | 4.3844 |
| 2 | 4.2951 | 4.0659 |
| 3 | 4.0298 | 3.8832 |
| 4 | 3.8421 | 3.7551 |
| 5 | 3.6921 | 3.6597 |
| 6 | 3.5665 | 3.5882 |
| 7 | 3.4574 | 3.5258 |
| 8 | 3.3579 | 3.4834 |
| 9 | 3.2693 | 3.4506 |
| 10 | 3.1877 | 3.4228 |

BLEU evaluation on 500 validation images:

| Metric | Score |
| --- | ---: |
| BLEU-1 | 0.5750 |
| BLEU-2 | 0.2629 |
| BLEU-3 | 0.1206 |
| BLEU-4 | 0.0575 |

### Baseline Interpretation

The baseline loss curve is healthy: training and validation losses both decrease across all 10 epochs, with no early plateau. The gap between train loss and validation loss remains moderate, suggesting the model is learning useful dataset structure rather than immediately overfitting.

However, qualitative inspection exposes a major limitation. The baseline repeatedly generated the same high-frequency template:

```text
A white and black box of coffee is on a table ..
```

This prediction appeared for unrelated samples including a dollar bill, a boxed food item, and a green aerosol can. This is a classic captioning failure mode: the decoder learns a fluent, frequent caption prior but does not ground individual words strongly enough in the image.

The BLEU scores reflect this mixed behavior. BLEU-1 is reasonable because generic words such as articles, colors, and common objects overlap often, but BLEU-4 is low because full generated phrases do not match the references reliably.

## Attention Model

The attention checkpoint exists and is larger than the baseline checkpoint, which is expected because the attention model keeps spatial ResNet features and adds attention-specific decoder layers. The local attention notebook currently has no saved Kaggle outputs, so attention training losses, BLEU scores, and qualitative predictions should be added after the executed notebook is pulled or exported with outputs.

Expected role of the attention model:

- Improve visual grounding by conditioning each generated token on spatial image regions.
- Reduce baseline template repetition.
- Provide attention maps for qualitative debugging.

## Recommendations

1. Keep the baseline as the documented reference model.
   It trains successfully and provides a useful lower bound, but qualitative outputs show it is not production-quality.

2. Prioritize attention-model evaluation.
   The baseline's repeated caption template is exactly the kind of failure attention is meant to address. The next comparison should report attention train/validation loss, BLEU-1 to BLEU-4, and side-by-side qualitative examples.

3. Add beam search after the attention run is validated.
   Greedy decoding can amplify generic-token choices. Beam search with a small beam size, such as 3 or 5, may improve sentence-level fluency and BLEU-4.

4. Add repetition controls in decoding.
   If repeated templates remain, apply simple inference-time penalties for repeated tokens or repeated n-grams.

5. Save structured metrics during training.
   Future notebook runs should write a small JSON or CSV metrics file to `/kaggle/working`, for example `baseline_metrics.json` and `attention_metrics.json`. This makes result comparison easier without relying on notebook output cells.

6. Keep clearing notebook outputs before committing.
   Model checkpoints and rendered notebook outputs should stay out of git. Commit lightweight notebooks and markdown result summaries instead.
