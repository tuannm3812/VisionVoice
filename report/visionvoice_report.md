# VisionVoice Image Captioning Report

## 1. Dataset Presentation

This project uses the official VizWiz-Captions validation set, as required by the assignment brief. The raw dataset contains 7,750 images and 38,750 human-written captions, with five captions per image before cleaning. VizWiz is an accessibility-focused dataset: the images are taken by people who are blind or have low vision, so many samples contain blur, poor lighting, partial objects, unusual framing, and real-world ambiguity.

The modelling notebook was executed on Kaggle with CUDA acceleration using the mounted dataset path `/kaggle/input/datasets/tuannm3823/vizwiz`.

## 2. Data Preparation

The shared preparation pipeline loads the raw annotation JSON, joins caption records to image file names, removes captions marked as rejected or precanned, and drops images that no longer have valid captions after filtering.

| Preparation item | Result |
| --- | ---: |
| Raw annotations | 38,750 |
| Clean captions retained | 33,145 |
| Noisy captions removed | 5,605 |
| Usable images retained | 7,542 |
| Internal train split | 6,033 images / 26,474 captions |
| Internal validation split | 754 images / 3,310 captions |
| Internal test split | 755 images / 3,361 captions |
| Vocabulary size | 6,439 tokens |
| Batch size | 32 |

Although the project uses only the official VizWiz validation set, the notebook creates internal image-level train, validation, and test splits. This avoids caption leakage by ensuring that captions from the same image do not appear in multiple splits. The internal validation split is used for checkpoint selection, while the internal test split is used for final BLEU reporting and visual inspection.

## 3. Model 1: Baseline ResNet-LSTM

The first architecture is a conventional encoder-decoder captioning model. A pretrained ResNet-50 encoder extracts a single global visual feature vector from each image. The ResNet backbone is frozen, and a learned projection maps the image representation into the decoder embedding space. An LSTM decoder is trained with teacher forcing to predict the next caption token.

| Architecture detail | Value |
| --- | --- |
| Encoder | Frozen ImageNet-pretrained ResNet-50 |
| Decoder | LSTM language decoder |
| Loss | Cross-entropy with padding ignored |
| Optimizer | Adam, learning rate 0.0001 |
| Decoding | Greedy decoding |
| Total parameters | 30,561,639 |
| Trainable parameters | 7,053,607 |
| Frozen parameters | 23,508,032 |

The rationale for this model is that it provides a simple and interpretable baseline. It tests whether a global CNN feature vector contains enough visual information for caption generation on VizWiz.

## 4. Model 2: Attention-Based ResNet-LSTM

The refined architecture keeps the ResNet-50 visual backbone but preserves spatial image features instead of compressing the whole image into one vector. A Bahdanau-style additive attention module lets the decoder attend to different regions of the image at each generated word. The decoder uses an LSTMCell conditioned on both the previous word and the current attention context.

| Architecture detail | Value |
| --- | --- |
| Encoder | ResNet-50 spatial feature map |
| Attention | Bahdanau-style additive attention |
| Decoder | LSTMCell conditioned on attended context |
| Decoder learning rate | 0.0001 |
| Encoder fine-tuning learning rate | 0.00001 |
| Decoding | Greedy decoding |
| Interpretability output | Attention heatmaps |

The second model was motivated by the baseline's qualitative failure mode. The baseline learned fluent but weakly grounded captions, so the refined model gives the decoder direct access to localized visual evidence while generating each token.

## 5. Training Results

Both models were trained for 10 epochs. Training and validation losses decreased consistently for both architectures, indicating stable optimization.

| Model | Final train loss | Final validation loss |
| --- | ---: | ---: |
| Baseline ResNet-LSTM | 3.1877 | 3.4259 |
| Attention ResNet-LSTM | 3.1958 | 3.4695 |

The baseline achieved a slightly lower validation loss. However, token-level cross-entropy does not fully describe caption quality, so BLEU scores and visual inspection are necessary for interpretation.

## 6. Evaluation Metrics

The previous completed notebook run evaluated both models on 500 sampled images from the internal test split using BLEU-1 through BLEU-4. The notebook is now configured to refresh these metrics on the full internal test split before final submission.

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
| --- | ---: | ---: | ---: | ---: |
| Baseline ResNet-LSTM | 0.5722 | 0.2712 | 0.1329 | 0.0670 |
| Attention ResNet-LSTM | 0.6159 | 0.3970 | 0.2502 | 0.1552 |
| Absolute improvement | +0.0437 | +0.1258 | +0.1173 | +0.0882 |

The attention model improves every BLEU metric. The largest practical improvement is at BLEU-4, where the score increases from 0.0670 to 0.1552. This suggests that attention improves phrase-level caption quality, not only individual word overlap.

## 7. Qualitative Analysis

The baseline model showed a clear template-collapse problem during visual inspection. For three unrelated test images, it generated the same caption:

```text
A white and black box of coffee is on a table ..
```

This prediction was produced for a 7-Up bottle, a dog, and floral fabric. The repeated caption indicates that the baseline decoder learned a strong language prior but did not reliably ground its output in the image.

The attention model produced more image-specific outputs, but it was still imperfect. For an image of a person's lap in jeans, it predicted a person wearing a blue shirt; this partially captured the human/clothing context but missed the main scene. For an office-chair image, it identified a chair but repeated words and missed some visual details. These examples show that attention improves grounding but does not fully solve repetition, object specificity, or fine-grained visual reasoning.

## 8. Limitations and Remaining Issues

The main limitation is that both models are relatively small captioning systems trained only on the VizWiz validation split. The dataset is challenging, and the training corpus is limited after cleaning. Greedy decoding also encourages generic high-probability captions and can amplify repetition.

Additional limitations include:

- Evaluation is based on a 500-image sample from the internal test split rather than the full internal test split.
- BLEU rewards n-gram overlap but does not fully measure semantic correctness or accessibility usefulness.
- The attention model still repeats phrases and sometimes attends to broadly relevant regions without naming the correct object.
- The report currently summarizes one student's modelling notebook; the final group submission should add each teammate's architectures, results, and contribution statement.

## 9. Recommendations

The attention model should be treated as the stronger architecture for the current submission because it improves all BLEU scores and directly addresses the baseline's weak visual grounding. The project should keep ResNet-50 as the selected visual backbone, because this was the architecture choice communicated to the group and both submitted models are already built around it.

The recommended next step is not to create a separate modelling notebook. The assignment asks each student to submit one modelling notebook containing at least two architectures with evaluation for both, so the current `02_modeling.ipynb` structure remains appropriate. Any final refinements should be made inside that notebook so the submission stays clear and easy to assess.

Recommended refinements:

- Evaluate both models on the full internal test split rather than a 500-image sample, since Kaggle runtime is not currently a constraint.
- Keep Model 1 as the frozen ResNet-50 plus LSTM baseline.
- Keep Model 2 as the refined ResNet-50 attention model, with spatial features and partial encoder fine-tuning.
- Use beam search with length normalization and repetition penalties as an additional decoding experiment for Model 2, while keeping greedy decoding results for fair comparison with the current run.
- Report qualitative examples alongside BLEU to show whether generated captions are visually grounded.

## 10. Contribution Statement

For this repository, the shared EDA notebook covers Phase 1 data preparation, while the modelling notebook contains the individual baseline and refined attention architectures with evaluation for both. Before final Canvas submission, the group should merge this material with other members' model results and explicitly list which student completed each modelling notebook, experiment, and report section.
