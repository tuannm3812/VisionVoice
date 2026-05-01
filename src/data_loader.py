"""
Custom Dataset class and Vocabulary builder for VisionVoice.

This module handles loading VizWiz images and captions, builds vocabulary,
and provides a PyTorch Dataset interface for training.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import json
from collections import Counter


class Vocabulary:
    """Vocabulary class for encoding/decoding captions."""
    
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.word_count = Counter()
        self.idx = 4
    
    def add_word(self, word):
        """Add a word to vocabulary."""
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def build_vocab(self, captions, threshold=5):
        """Build vocabulary from captions with frequency threshold."""
        for caption in captions:
            for word in caption.split():
                self.word_count[word] += 1
        
        for word, count in self.word_count.items():
            if count >= threshold:
                self.add_word(word)
    
    def encode_caption(self, caption):
        """Convert caption to indices."""
        tokens = caption.split()
        indices = [self.word2idx.get(word, self.word2idx["<unk>"]) for word in tokens]
        return indices
    
    def __len__(self):
        return len(self.word2idx)


class VizWizDataset(Dataset):
    """Custom Dataset for VizWiz image-caption pairs."""
    
    def __init__(self, img_dir, annotation_file, vocab, transform=None, max_length=15):
        self.img_dir = Path(img_dir)
        self.vocab = vocab
        self.transform = transform
        self.max_length = max_length
        
        with open(annotation_file, 'r') as f:
            self.captions = json.load(f)
    
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
        caption_info = self.captions[idx]
        img_name = caption_info['image_id']
        caption = caption_info['caption']
        
        # Load image
        img_path = self.img_dir / img_name
        # TODO: Load and preprocess image
        
        # Encode caption
        tokens = [self.vocab.word2idx["<start>"]] + self.vocab.encode_caption(caption) + [self.vocab.word2idx["<end>"]]
        
        # Pad or truncate to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens += [self.vocab.word2idx["<pad>"]] * (self.max_length - len(tokens))
        
        return None, torch.tensor(tokens)  # Placeholder for image tensor
