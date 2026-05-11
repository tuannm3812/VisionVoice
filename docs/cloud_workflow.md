# Cloud Workflow

This project uses a simple cloud handoff pattern:

- **GitHub** stores the source code, notebooks, markdown reports, and lightweight documentation.
- **Kaggle** runs the GPU notebooks and stores temporary training outputs in `/kaggle/working`.
- **External cloud storage** such as Google Drive, OneDrive, or Kaggle Datasets should store large artifacts such as `.pth` checkpoints, downloaded images, generated zip files, and final submission bundles.

## What to Upload

For a clean handoff, upload the generated bundle from:

```text
dist/VisionVoice_cloud_bundle.zip
```

This bundle contains:

- `README.md`
- `requirements.txt`
- `notebooks/`
- `docs/`
- `report/`
- `src/`

It intentionally excludes:

- raw datasets,
- model checkpoints,
- Kaggle working directories,
- Python cache files,
- local virtual environments,
- generated logs and result folders.

## Create the Bundle

From the repository root, run:

```powershell
.\scripts\prepare_cloud_bundle.ps1
```

The script recreates the `dist/` folder and writes a fresh zip file.

## Kaggle Notes

The modelling notebook is currently configured for the final rerun:

- `BLEU_EVAL_SAMPLES = None`, so BLEU uses the full internal test split.
- `EVALUATE_BEAM_SEARCH = True`, so Model 2 reports both greedy and beam-search decoding.
- Checkpoints and metrics are saved to `/kaggle/working`.

After the Kaggle run finishes, download or archive:

```text
vision_voice_baseline_best.pth
vision_voice_attention_best.pth
baseline_metrics_greedy.json
attention_metrics_greedy.json
attention_metrics_beam.json
```

Keep those files outside git unless the final submission explicitly asks for them.
