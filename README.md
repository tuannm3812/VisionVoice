# VisionVoice

VisionVoice is an image captioning project for the VizWiz-Captions dataset. It explores the dataset, trains a baseline ResNet-LSTM captioning model, and then refines the architecture with Bahdanau-style visual attention.

The Kaggle workflow is organized around two rerunnable notebooks: one shared EDA/data-preparation notebook and one modelling notebook containing both required architectures.

The assignment uses only the official VizWiz-Captions validation set. Inside that source dataset, the notebooks create internal image-level train/validation/test splits for model training, checkpoint selection, and final reporting.

## Current Results

The latest Kaggle run evaluated all 755 images in the internal test split. The attention model is reported with both greedy decoding and beam-search decoding.

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | Key takeaway |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline ResNet-LSTM | 0.5740 | 0.2692 | 0.1281 | 0.0645 | Trains stably but collapses to repeated generic captions in visual inspection. |
| Attention ResNet-LSTM, greedy | 0.6129 | 0.3960 | 0.2519 | 0.1593 | Improves all BLEU scores and gives better phrase-level overlap than the baseline. |
| Attention ResNet-LSTM, beam | 0.6640 | 0.4373 | 0.2897 | 0.1935 | Best quantitative result; beam search improves every BLEU score over greedy decoding. |

The attention model with beam search is the strongest current result because it substantially improves phrase-level overlap, especially BLEU-4, and directly addresses the baseline's weak visual grounding.

## Notebooks

| Notebook | Purpose |
| --- | --- |
| `notebooks/01_eda_vizwiz.ipynb` | Dataset loading, annotation cleaning, image/caption inspection, leakage-free split, integrity audits, cleaning-impact analysis, vocabulary coverage, and caption-length analysis. |
| `notebooks/02_modeling.ipynb` | Main assignment submission notebook containing shared preparation, Model 1 baseline, Model 2 attention refinement, and evaluation for both. |

## Project Structure

```text
VisionVoice/
|-- notebooks/
|   |-- 01_eda_vizwiz.ipynb
|   `-- 02_modeling.ipynb
|-- report/
|   `-- visionvoice_report.md
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
   - `02_modeling.ipynb`

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

For routine development commits, clear notebook outputs before pushing. For the final assignment submission, keep the executed notebook outputs because the brief asks for notebooks with cell outputs. Even lightweight static figures and image displays are stored inside `.ipynb` files and can make notebooks large:

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
report/visionvoice_report.md
```

Assignment phase mapping is documented in:

```text
docs/assignment_alignment.md
```

Cloud handoff and packaging notes are documented in:

```text
docs/cloud_workflow.md
docs/codex_cloud_handoff.md
```
