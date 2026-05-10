# VisionVoice Notebooks

Run these notebooks in order when reproducing the Kaggle workflow:

1. `01_eda_vizwiz.ipynb`
   - Loads VizWiz annotations and images.
   - Audits noisy captions and dataset quality flags.
   - Builds clean train/validation splits.
   - Checks split structure and image file integrity.
   - Quantifies cleaning impact, orphaned images, and reference-caption coverage.
   - Checks vocabulary coverage, validation OOV rate, and sequence truncation risk.
   - Studies vocabulary and caption length distributions.

2. `02_baseline_resnet_lstm.ipynb`
   - Recreates the cleaned data pipeline.
   - Trains the baseline ResNet-50 encoder and LSTM decoder.
   - Saves `vision_voice_baseline_best.pth`.
   - Evaluates BLEU and shows sample predictions.

3. `03_attention_resnet_lstm.ipynb`
   - Recreates the cleaned data pipeline.
   - Trains the attention-based ResNet-50 plus LSTMCell decoder.
   - Saves `vision_voice_attention_best.pth`.
   - Evaluates BLEU-1 through BLEU-4 and visualizes attention maps.

Assignment submission notebook:

- `02_individual_models_resnet_lstm_attention.ipynb`
  - Contains the shared data preparation, Model 1 baseline, Model 2 attention refinement, and evaluation for both architectures in one notebook.
  - Use this notebook when the requirement is that each student's notebook contain at least two architectures.

Each notebook is standalone for Kaggle, so it repeats the setup and data-preparation cells it needs.

Mode defaults:

- `01_eda_vizwiz.ipynb`: `MODE = "eda"`, `SHOW_EDA_PLOTS = True`, `SHOW_MODEL_PLOTS = False`
- `02_individual_models_resnet_lstm_attention.ipynb`: `MODE = "train"`, `SHOW_EDA_PLOTS = False`, `SHOW_MODEL_PLOTS = True`
- `02_baseline_resnet_lstm.ipynb`: `MODE = "train"`, `SHOW_EDA_PLOTS = False`, `SHOW_MODEL_PLOTS = True`
- `03_attention_resnet_lstm.ipynb`: `MODE = "train"`, `SHOW_EDA_PLOTS = False`, `SHOW_MODEL_PLOTS = True`

Clear notebook outputs before pushing Kaggle-run notebooks to GitHub. Static figures and rendered images are saved inside `.ipynb` output cells and can easily make a notebook too large:

```bash
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```
