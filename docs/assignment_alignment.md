# Assignment Phase Alignment

This repository is organized around the three-phase workflow in the project brief.

## Phase 1 - Shared Data Preparation

Primary notebook:

- `notebooks/01_eda_vizwiz.ipynb`

Coverage:

- Load VizWiz annotations and images.
- Clean rejected and precanned captions.
- Remove orphaned images after filtering.
- Create leakage-free internal train/validation/test splits by image file name.
- Build the vocabulary from training captions only.
- Audit image-file integrity.
- Analyze reference-caption coverage, vocabulary coverage, and sequence-length risk.

Current split note:

- The assignment requires using only the official VizWiz-Captions validation set with 7,750 images.
- Within that official validation set, the notebooks create internal image-level splits: 80% train, 10% validation, and 10% test after cleaning.
- The internal validation split is used for loss monitoring and checkpoint selection.
- The internal test split is reserved for final BLEU reporting and visual inspection.
- These are internal experimental splits, not the official hidden VizWiz challenge test set.

## Phase 2 - Individual Architecture (Model 1)

Primary model:

- Baseline ResNet-LSTM encoder-decoder.

Model 1 appears in:

- `notebooks/02_baseline_resnet_lstm.ipynb`
- `notebooks/02_individual_models_resnet_lstm_attention.ipynb`

Design:

- Frozen pretrained ResNet-50 encoder.
- Learned feature projection.
- LSTM decoder trained with teacher forcing.
- Greedy decoding.
- BLEU-1 to BLEU-4 evaluation and qualitative inspection.

Key finding:

- The model trains successfully, but qualitative inspection shows template collapse: unrelated images receive the same generic caption.

## Phase 3 - Refined Architecture (Model 2)

Primary model:

- ResNet-LSTM with Bahdanau-style attention.

Model 2 appears in:

- `notebooks/03_attention_resnet_lstm.ipynb`
- `notebooks/02_individual_models_resnet_lstm_attention.ipynb`

Design rationale:

- Model 1's repeated-caption behavior suggests the decoder relies too heavily on language priors.
- The refined model keeps spatial ResNet features and lets the decoder attend to image regions at each generation step.
- Attention heatmaps make qualitative debugging possible.

Evaluation:

- Full BLEU-1 to BLEU-4 reporting.
- Qualitative caption inspection.
- Attention heatmap visualization.
- Optional beam-search decoding and repetition penalty hooks for further experiments.
- Metrics JSON export to `/kaggle/working`.

## Submission Notebook

For assignment submission, use:

```text
notebooks/02_individual_models_resnet_lstm_attention.ipynb
```

This notebook contains:

- the shared data-preparation pipeline,
- Model 1 architecture and evaluation,
- Phase 3 discussion/rationale,
- Model 2 architecture and evaluation.

The split notebooks are kept for easier development and reruns, but the combined notebook best matches the brief's requirement that each student's notebook contain at least two architectures with evaluation for both.
