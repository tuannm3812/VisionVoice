# VisionVoice: Image Captioning with Attention

A PyTorch implementation of an attention-based image captioning model trained on the VizWiz dataset. This project generates natural language descriptions for images using a CNN encoder and LSTM decoder with attention mechanism.

## Project Overview

VisionVoice combines:
- **CNN Encoder**: Pretrained ResNet50/VGG16 for visual feature extraction
- **LSTM Decoder**: Sequence-to-sequence generation with attention mechanism
- **Attention Module**: Focus on relevant image regions during caption generation
- **Training Pipeline**: Complete setup with validation and checkpointing

## Repository Structure

```
VisionVoice/
├── data/                      # Data storage (ignored by git)
│   ├── images/                # VizWiz validation images
│   └── annotations/           # VizWiz caption JSON files
├── notebooks/                 
│   └── exploratory_analysis.ipynb # Dataset visualization & word clouds
├── src/                       
│   ├── data_loader.py         # Custom Dataset class & Vocabulary builder
│   ├── encoder.py             # CNN feature extractor (Pretrained VGG/ResNet)
│   ├── decoder.py             # Sequence model (LSTM/RNN with Attention)
│   ├── train.py               # Training loop and checkpointing logic
│   └── eval.py                # BLEU score calculations & inference
├── models/                    # Saved .pth model weights
├── results/                   # Evaluation plots and sample captions
├── .gitignore                 # Prevents large data/weights from being uploaded
├── requirements.txt           # Environment dependencies
└── README.md                  # Project overview and usage instructions
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/VisionVoice.git
cd VisionVoice
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Setup

1. Download the VizWiz dataset from [VizWiz Challenge](https://vizwiz.org/)
2. Place images in `data/images/`
3. Place annotation JSON files in `data/annotations/`

Expected annotation format:
```json
[
  {
    "image_id": "VizWiz_train_0000001.jpg",
    "caption": "a woman wearing glasses"
  },
  ...
]
```

## Usage

### Data Exploration

Open the Jupyter notebook for dataset visualization:
```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

### Training

```python
from src.data_loader import VizWizDataset, Vocabulary
from src.encoder import ImageEncoder
from src.decoder import CaptionDecoder
from src.train import Trainer
from torch.utils.data import DataLoader

# Setup
vocab = Vocabulary()
dataset = VizWizDataset('data/images', 'data/annotations/captions.json', vocab)
loader = DataLoader(dataset, batch_size=32)

# Model
encoder = ImageEncoder('resnet50', pretrained=True)
decoder = CaptionDecoder(len(vocab), embedding_dim=256, hidden_dim=512)

# Train
trainer = Trainer(decoder, device='cuda')
trainer.train(loader, loader, num_epochs=10)
```

### Inference

```python
from src.eval import Evaluator

evaluator = Evaluator(model, vocab, device='cuda')
caption = evaluator.generate_caption(image_features)
print(f"Generated caption: {caption}")
```

### Evaluation

```python
results = evaluator.evaluate_dataset(test_loader)
print(f"BLEU-4 Score: {results['avg_bleu_score']:.4f}")
```

## Model Architecture

### Encoder
- **Base Model**: ResNet-50 (pretrained on ImageNet)
- **Output**: 2048-dimensional feature vectors
- **Frozen Weights**: Encoder weights are frozen during training

### Decoder
- **Embedding Layer**: Word embeddings (256 dimensions)
- **LSTM**: 512-dimensional hidden state
- **Attention**: Computes attention weights over image features
- **Output**: Vocabulary-sized logits for next word prediction

## Training Details

- **Optimizer**: Adam (lr=0.001)
- **Loss**: Cross-entropy with padding ignored
- **Batch Size**: 32
- **Max Caption Length**: 15 words
- **Vocabulary**: Built with minimum frequency threshold of 5

## Results

Evaluation metrics are saved in `results/` directory:
- BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
- Sample captions on validation set
- Training loss curves

## Future Improvements

- Implement beam search for caption generation
- Add visual attention visualization
- Experiment with Transformer-based decoder
- Support for multi-language captions
- Fine-tune encoder weights on VizWiz data

## References

- Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
- VizWiz Challenge Dataset
- PyTorch Documentation

## License

This project is provided as-is for educational and research purposes.

## Contact

For questions or contributions, please open an issue or submit a pull request.
