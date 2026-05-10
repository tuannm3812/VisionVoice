# VisionVoice Model Results

This document summarizes the executed Kaggle outputs currently stored in `notebooks/02_modeling.ipynb`.

## Run Setup

| Item | Value |
| --- | --- |
| Runtime | Kaggle |
| Python | 3.12.12 |
| PyTorch | 2.10.0+cu128 |
| Device | CUDA |
| Source dataset | Official VizWiz-Captions validation set |
| Dataset path | `/kaggle/input/datasets/tuannm3823/vizwiz` |
| Evaluation sample | 500 images from the internal test split |

The project uses only the official VizWiz validation set. The notebook creates internal image-level splits for training, checkpoint selection, and final reporting.

## Prepared Dataset

| Item | Count |
| --- | ---: |
| Raw annotations | 38,750 |
| Clean/descriptive captions | 33,145 |
| Precanned-only captions | 4,641 |
| Rejected-only captions | 438 |
| Both precanned and rejected | 526 |
| Usable images after cleaning | 7,542 |
| Train split | 6,033 images / 26,474 captions |
| Validation split | 754 images / 3,310 captions |
| Test split | 755 images / 3,361 captions |
| Vocabulary size | 6,439 tokens |

## Baseline ResNet-LSTM

Architecture:

- Frozen ImageNet-pretrained ResNet-50 encoder.
- Learned projection from global CNN features to the decoder embedding space.
- LSTM decoder trained with teacher forcing.
- Greedy decoding at inference time.

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
| 1 | 5.0863 | 4.3872 |
| 2 | 4.2951 | 4.0668 |
| 3 | 4.0298 | 3.8838 |
| 4 | 3.8421 | 3.7551 |
| 5 | 3.6921 | 3.6617 |
| 6 | 3.5665 | 3.5904 |
| 7 | 3.4574 | 3.5282 |
| 8 | 3.3579 | 3.4836 |
| 9 | 3.2693 | 3.4530 |
| 10 | 3.1877 | 3.4259 |

BLEU results:

| Metric | Score |
| --- | ---: |
| BLEU-1 | 0.5722 |
| BLEU-2 | 0.2712 |
| BLEU-3 | 0.1329 |
| BLEU-4 | 0.0670 |

Qualitative finding:

The baseline repeatedly generated the same caption for unrelated images:

```text
A white and black box of coffee is on a table ..
```

This occurred for a 7-Up bottle, a dog, and floral fabric. The model learned a plausible language template but did not ground the caption reliably in the image.

## Attention ResNet-LSTM

Architecture:

- ResNet-50 spatial feature encoder.
- Bahdanau-style additive attention over image regions.
- LSTMCell decoder conditioned on the attended visual context.
- Decoder learning rate of `1e-4` and encoder fine-tuning learning rate of `1e-5`.
- Greedy decoding, with attention heatmaps for inspection.

Training history:

| Epoch | Train loss | Validation loss |
| ---: | ---: | ---: |
| 1 | 5.0891 | 4.4299 |
| 2 | 4.3879 | 4.1053 |
| 3 | 4.1006 | 3.9151 |
| 4 | 3.8950 | 3.7885 |
| 5 | 3.7355 | 3.6981 |
| 6 | 3.6007 | 3.6279 |
| 7 | 3.4822 | 3.5743 |
| 8 | 3.3785 | 3.5316 |
| 9 | 3.2821 | 3.5033 |
| 10 | 3.1958 | 3.4695 |

BLEU results:

| Metric | Score |
| --- | ---: |
| BLEU-1 | 0.6159 |
| BLEU-2 | 0.3970 |
| BLEU-3 | 0.2502 |
| BLEU-4 | 0.1552 |

Qualitative finding:

The attention model produced more image-specific captions than the baseline, but it still made visible errors. It partially captured clothing/person context in one example and correctly identified chairs in another, while still showing repetition and occasional object-detail mistakes.

## Comparison

| Model | Final train loss | Final validation loss | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline ResNet-LSTM | 3.1877 | 3.4259 | 0.5722 | 0.2712 | 0.1329 | 0.0670 |
| Attention ResNet-LSTM | 3.1958 | 3.4695 | 0.6159 | 0.3970 | 0.2502 | 0.1552 |
| Difference (attention - baseline) | +0.0081 | +0.0436 | +0.0437 | +0.1258 | +0.1173 | +0.0882 |

The baseline has slightly lower final validation loss, but the attention model has better BLEU scores across all n-gram levels. This matters because validation loss is token-level, while BLEU better reflects sequence-level caption overlap with references. The BLEU-4 improvement is especially important because it suggests stronger phrase-level caption quality.

## Recommendations

- Treat the attention model as the stronger current architecture.
- Keep the baseline as a useful reference point and evidence for the Phase 3 refinement.
- Evaluate on the full internal test split if runtime allows.
- Add beam search with length normalization and repetition penalties to reduce generic repeated captions.
- Consider deeper encoder fine-tuning or a stronger visual backbone after the attention decoder is stable.
- Keep visual inspection in the modelling notebook because BLEU alone does not capture whether captions are grounded in the image.
