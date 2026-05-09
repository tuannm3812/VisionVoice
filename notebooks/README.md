# VisionVoice Notebooks

Run these notebooks in order when reproducing the Kaggle workflow:

1. `01_eda_vizwiz.ipynb`
   - Loads VizWiz annotations and images.
   - Audits noisy captions and dataset quality flags.
   - Builds clean train/validation splits.
   - Checks split structure and image file integrity.
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
   - Evaluates BLEU and visualizes attention maps.

Each notebook is standalone for Kaggle, so it repeats the setup and data-preparation cells it needs.
