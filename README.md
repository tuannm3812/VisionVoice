# VisionVoice

VisionVoice is an image captioning project for the VizWiz-Captions dataset. It explores the dataset, trains a baseline ResNet-LSTM captioning model, and then refines the architecture with Bahdanau-style visual attention.

The Kaggle workflow is split into three rerunnable notebooks so experimentation is easier to review, debug, and compare.

The assignment uses only the official VizWiz-Captions validation set. Inside that source dataset, the notebooks create internal image-level train/validation/test splits for model training, checkpoint selection, and final reporting.

## Notebooks

| Notebook | Purpose |
| --- | --- |
| `notebooks/01_eda_vizwiz.ipynb` | Dataset loading, annotation cleaning, image/caption inspection, leakage-free split, integrity audits, cleaning-impact analysis, vocabulary coverage, and caption-length analysis. |
| `notebooks/02_individual_models_resnet_lstm_attention.ipynb` | Main assignment submission notebook containing shared preparation, Model 1 baseline, Model 2 attention refinement, and evaluation for both. |
| `notebooks/02_baseline_resnet_lstm.ipynb` | Baseline CNN encoder plus LSTM decoder training, validation loss tracking, BLEU evaluation, and qualitative caption inspection. |
| `notebooks/03_attention_resnet_lstm.ipynb` | Spatial ResNet encoder, attention decoder, two-tier optimizer, BLEU evaluation, and attention heatmap visualization. |

## Project Structure

```text
VisionVoice/
|-- notebooks/
|   |-- 01_eda_vizwiz.ipynb
|   |-- 02_individual_models_resnet_lstm_attention.ipynb
|   |-- 02_baseline_resnet_lstm.ipynb
|   `-- 03_attention_resnet_lstm.ipynb
|-- src/
|   |-- data_loader.py
|   |-- decoder.py
|   |-- encoder.py
|   |-- eval.py
|   `-- train.py
|-- requirements.txt
|-- .gitignore
`-- README.md
```

Large files such as datasets, trained weights, generated outputs, and Kaggle working artifacts should stay outside git.

## Kaggle Setup

1. Create a Kaggle notebook.
2. Attach the VizWiz dataset used by the project.
3. Upload or copy one of the split notebooks.
4. Enable GPU acceleration.
5. Run the notebooks in order:
   - `01_eda_vizwiz.ipynb`
   - `02_baseline_resnet_lstm.ipynb`
   - `03_attention_resnet_lstm.ipynb`

The notebooks default to this Kaggle dataset path:

```text
/kaggle/input/datasets/tuannm3823/vizwiz
```

If your Kaggle dataset slug changes, set these environment variables in the first setup cell before creating `CFG`:

```python
import os
os.environ["VIZWIZ_DATA_DIR"] = "/kaggle/input/<your-dataset-folder>"
os.environ["VIZWIZ_ANNOTATIONS_DIR"] = "/kaggle/input/<your-dataset-folder>/annotations/annotations"
os.environ["VIZWIZ_IMAGE_DIR"] = "/kaggle/input/<your-dataset-folder>/val/val"
```

Each notebook has a mode switch in `Settings`:

```python
MODE = "eda"      # EDA notebook
MODE = "train"    # baseline and attention notebooks
SHOW_EDA_PLOTS = MODE == "eda"
SHOW_MODEL_PLOTS = MODE in {"train", "inference"}
```

The training notebooks keep EDA plots off by default but keep post-training diagnostics on, so you still get loss curves, qualitative caption checks, and attention visualizations.

Before pushing notebooks after a Kaggle run, clear outputs first. Even lightweight static figures and image displays are stored inside `.ipynb` files and can make notebooks unnecessarily large:

```bash
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For local runs, place data in:

```text
data/
|-- annotations/
|   `-- val.json
`-- images/
    `-- *.jpg
```

You can also override the local data root:

```bash
set VIZWIZ_LOCAL_DATA_DIR=C:\path\to\vizwiz
```

## Models

### Baseline

- ResNet-50 encoder pretrained on ImageNet.
- Frozen visual backbone with a learned projection layer.
- LSTM decoder trained with teacher forcing.
- Cross-entropy loss with padding ignored.
- Greedy decoding for inference.

### Attention Model

- ResNet-50 spatial feature encoder.
- Bahdanau-style additive attention over a 7x7 feature grid.
- LSTMCell decoder conditioned on the current attention context.
- Separate learning rates for decoder training and encoder fine-tuning.
- Attention heatmaps for qualitative inspection.

## Outputs

Kaggle saves trained checkpoints to `/kaggle/working`:

```text
vision_voice_baseline_best.pth
vision_voice_attention_best.pth
```

These files are intentionally ignored by git.

Current training results and recommendations are documented in:

```text
docs/model_results.md
```

Assignment phase mapping is documented in:

```text
docs/assignment_alignment.md
```
