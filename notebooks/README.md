# VisionVoice Notebooks

Run these notebooks in order when reproducing the Kaggle workflow:

1. `01_eda_vizwiz.ipynb`
   - Loads VizWiz annotations and images.
   - Audits noisy captions and dataset quality flags.
   - Builds clean internal train/validation/test splits from the official VizWiz validation set.
   - Checks split structure and image file integrity.
   - Quantifies cleaning impact, orphaned images, and reference-caption coverage.
   - Checks vocabulary coverage, validation OOV rate, and sequence truncation risk.
   - Studies vocabulary and caption length distributions.

2. `02_modeling.ipynb`
   - Recreates the cleaned data pipeline.
   - Trains Model 1: baseline ResNet-50 encoder and LSTM decoder.
   - Trains Model 2: attention-based ResNet-50 plus LSTMCell decoder.
   - Selects checkpoints on the internal validation split.
   - Evaluates both architectures on the full internal test split by default.
   - Reports greedy decoding for both models and beam-search decoding for the attention model.
   - Saves `vision_voice_baseline_best.pth`, `vision_voice_attention_best.pth`, baseline metrics JSON, and attention metrics JSON artifacts.

Each notebook is standalone for Kaggle, so it repeats the setup and data-preparation cells it needs.

Mode defaults:

- `01_eda_vizwiz.ipynb`: `MODE = "eda"`, `SHOW_EDA_PLOTS = True`, `SHOW_MODEL_PLOTS = False`
- `02_modeling.ipynb`: `MODE = "train"`, `SHOW_EDA_PLOTS = False`, `SHOW_MODEL_PLOTS = True`

For routine development commits, clear notebook outputs before pushing to GitHub. For the final assignment submission, keep the Kaggle outputs because the brief asks for notebooks with cell outputs. Static figures and rendered images are saved inside `.ipynb` output cells and can easily make a notebook too large:

```bash
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```
