# VisionVoice Model Results

This document summarizes the completed Kaggle outputs from `notebooks/02_modeling.ipynb`.

## Run Setup

| Item | Value |
| --- | --- |
| Runtime | Kaggle |
| Python | 3.12.12 |
| PyTorch | 2.10.0+cu128 |
| Device | CUDA |
| Source dataset | Official VizWiz-Captions validation set |
| Dataset path | `/kaggle/input/datasets/tuannm3823/vizwiz` |
| Evaluation sample | Full internal test split: 755 images |

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
| BLEU-1 | 0.5740 |
| BLEU-2 | 0.2692 |
| BLEU-3 | 0.1281 |
| BLEU-4 | 0.0645 |

Qualitative finding:

The baseline repeatedly generated the same caption for unrelated images:

```text
A white and black box of coffee is on a table ..
```

This occurred for unrelated examples including floral fabric, a text-heavy document, and a computer pop-up. The model learned a plausible language template but did not ground the caption reliably in the image.

## Attention ResNet-LSTM

Architecture:

- ResNet-50 spatial feature encoder.
- Bahdanau-style additive attention over image regions.
- LSTMCell decoder conditioned on the attended visual context.
- Decoder learning rate of `1e-4` and encoder fine-tuning learning rate of `1e-5`.
- Greedy decoding and beam-search decoding, with attention heatmaps for inspection.

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

| Decoding | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 0.6129 | 0.3960 | 0.2519 | 0.1593 |
| Beam search | 0.6640 | 0.4373 | 0.2897 | 0.1935 |

Qualitative finding:

The attention model produced stronger corpus-level scores than the baseline, but visual inspection still showed visible errors. One inspected image of near-black glare was captioned as repeated "dark/light" wording, and another image containing a clock and pill container was described as a green-and-white food box. These examples show that attention improves sequence-level overlap but does not fully solve object specificity or repetition.

## Comparison

| Model | Final train loss | Final validation loss | Decoding | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Baseline ResNet-LSTM | 3.1877 | 3.4259 | Greedy | 0.5740 | 0.2692 | 0.1281 | 0.0645 |
| Attention ResNet-LSTM | 3.1958 | 3.4695 | Greedy | 0.6129 | 0.3960 | 0.2519 | 0.1593 |
| Attention ResNet-LSTM | 3.1958 | 3.4695 | Beam search | 0.6640 | 0.4373 | 0.2897 | 0.1935 |

The baseline has slightly lower final validation loss, but the attention model has better BLEU scores across all n-gram levels. This matters because validation loss is token-level, while BLEU better reflects sequence-level caption overlap with references. Beam search further improves the attention model, raising BLEU-4 from 0.1593 to 0.1935.

## Recommendations

- Treat the attention model with beam search as the strongest current result.
- Keep the baseline as a useful reference point and evidence for the Phase 3 refinement.
- Keep ResNet-50 as the selected backbone for both architectures, because it is the chosen design communicated to the group.
- Keep all modelling work in `notebooks/02_modeling.ipynb` rather than creating a third notebook.
- Keep full internal test split reporting in the final submission.
- Discuss beam search as an inference-time refinement rather than a separate architecture.
- Keep visual inspection in the modelling notebook because BLEU alone does not capture whether captions are grounded in the image.
