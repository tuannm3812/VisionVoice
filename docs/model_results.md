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

Note: these are validation-set results from an internal split of the annotated VizWiz validation data. If the assignment requires a separate final test split, reserve one by image file name before training and report final metrics on that untouched holdout.

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

Architecture:

- ResNet-50 spatial feature encoder
- Bahdanau-style additive attention over image regions
- LSTMCell decoder conditioned on attended visual context
- Two-tier optimizer with a higher decoder learning rate than encoder fine-tuning rate

Training history:

| Epoch | Train loss | Validation loss |
| ---: | ---: | ---: |
| 1 | 5.0840 | 4.4278 |
| 2 | 4.3909 | 4.0953 |
| 3 | 4.0947 | 3.9054 |
| 4 | 3.8850 | 3.7691 |
| 5 | 3.7231 | 3.6751 |
| 6 | 3.5893 | 3.6150 |
| 7 | 3.4715 | 3.5617 |
| 8 | 3.3642 | 3.5189 |
| 9 | 3.2695 | 3.4822 |
| 10 | 3.1824 | 3.4591 |

BLEU evaluation on 500 validation images:

| Metric | Score |
| --- | ---: |
| BLEU-1 | 0.5792 |
| BLEU-4 | 0.1430 |

### Attention Interpretation

The attention model's validation loss is slightly higher than the baseline's final validation loss, but its sequence-level BLEU-4 is much stronger. This suggests the attention model may not optimize token-level cross-entropy quite as well, but it produces captions with better phrase-level overlap and visual grounding.

The qualitative examples support this interpretation:

| Image | Baseline prediction | Attention prediction | Grounding assessment |
| --- | --- | --- | --- |
| `VizWiz_val_00007740.jpg` | `A white and black box of coffee is on a table ..` | `A person is holding a small dollar bill ..` | Attention correctly identifies the dollar bill and hand-held context. |
| `VizWiz_val_00001635.jpg` | `A white and black box of coffee is on a table ..` | `A box of frozen food with a picture of a food ..` | Attention captures the boxed-food object, though the wording remains generic. |

The attention heatmaps are also useful for debugging because they expose where the model looks while generating each word. This makes the model easier to diagnose than the baseline decoder, which only emits a caption.

## Model Comparison

| Model | Final train loss | Final validation loss | BLEU-1 | BLEU-4 | Qualitative behavior |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline ResNet-LSTM | 3.1877 | 3.4228 | 0.5750 | 0.0575 | Collapses to a repeated generic caption on inspected samples. |
| Attention ResNet-LSTM | 3.1824 | 3.4591 | 0.5792 | 0.1430 | Produces more image-specific captions and fixes the baseline's inspected template collapse. |

Key comparison:

- BLEU-1 is almost unchanged: attention improves from 0.5750 to 0.5792.
- BLEU-4 improves substantially: attention improves from 0.0575 to 0.1430.
- The attention model has a slightly worse final validation loss, but better n-gram sequence quality and much better qualitative grounding.

The attention model should be treated as the stronger model for this project because it solves the most important observed baseline failure: repeated, weakly grounded generic captions.

## Recommendations

1. Keep the baseline as the documented reference model.
   It trains successfully and provides a useful lower bound, but qualitative outputs show it is not production-quality.

2. Use the attention model as the main result.
   The attention model improves BLEU-4 and produces visibly better grounded captions on inspected samples.

3. Rerun attention evaluation after the notebook update.
   The attention notebook now reports BLEU-1 through BLEU-4 and saves `attention_metrics_<strategy>.json`. Rerun it on Kaggle to refresh this document with BLEU-2 and BLEU-3.

4. Add beam search after the attention run is validated.
   Greedy decoding can amplify generic-token choices. Beam search with a small beam size, such as 3 or 5, may improve sentence-level fluency and BLEU-4.

5. Add repetition controls in decoding.
   If repeated templates remain, apply simple inference-time penalties for repeated tokens or repeated n-grams.

6. Save structured metrics during training.
   Future notebook runs should write a small JSON or CSV metrics file to `/kaggle/working`, for example `baseline_metrics.json` and `attention_metrics.json`. This makes result comparison easier without relying on notebook output cells.

7. Keep clearing notebook outputs before committing.
   Model checkpoints and rendered notebook outputs should stay out of git. Commit lightweight notebooks and markdown result summaries instead.
