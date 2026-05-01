"""
Evaluation utilities for the image captioning model.

This module provides BLEU score calculations, inference on new images,
and visualization of results.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter
import numpy as np


class BLEUScorer:
    """BLEU score calculator for caption evaluation."""
    
    def __init__(self, n=4):
        """
        Initialize BLEU scorer.
        
        Args:
            n: Maximum n-gram size (default: 4 for BLEU-4)
        """
        self.n = n
    
    def _get_ngrams(self, words, n):
        """Extract n-grams from words."""
        ngrams = Counter()
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i+n])
            ngrams[ngram] += 1
        return ngrams
    
    def compute_bleu(self, reference_captions, generated_caption):
        """
        Compute BLEU score for a single generated caption.
        
        Args:
            reference_captions: List of reference captions (lists of words)
            generated_caption: Generated caption (list of words)
        
        Returns:
            BLEU score (0-1)
        """
        bleu_scores = []
        
        for n in range(1, self.n + 1):
            gen_ngrams = self._get_ngrams(generated_caption, n)
            
            # Maximum count in any reference
            max_ref_counts = Counter()
            for ref in reference_captions:
                ref_ngrams = self._get_ngrams(ref, n)
                for ngram in ref_ngrams:
                    max_ref_counts[ngram] = max(max_ref_counts[ngram], ref_ngrams[ngram])
            
            # Clipped counts
            clipped_counts = 0
            for ngram, count in gen_ngrams.items():
                clipped_counts += min(count, max_ref_counts.get(ngram, 0))
            
            # Precision
            precision = clipped_counts / sum(gen_ngrams.values()) if gen_ngrams else 0
            bleu_scores.append(precision)
        
        # Geometric mean with brevity penalty
        if all(score > 0 for score in bleu_scores):
            bleu = np.exp(sum(np.log(score) for score in bleu_scores) / self.n)
        else:
            bleu = 0
        
        return bleu


class Evaluator:
    """Evaluator class for model assessment."""
    
    def __init__(self, model, vocab, device='cpu'):
        """
        Initialize evaluator.
        
        Args:
            model: The trained model
            vocab: Vocabulary object
            device: Device to run on
        """
        self.model = model
        self.vocab = vocab
        self.device = device
        self.bleu_scorer = BLEUScorer(n=4)
    
    def generate_caption(self, image_features, max_length=20, temperature=1.0):
        """
        Generate a caption for an image.
        
        Args:
            image_features: Pre-extracted image features
            max_length: Maximum caption length
            temperature: Sampling temperature (lower = more deterministic)
        
        Returns:
            Generated caption as string
        """
        self.model.eval()
        
        with torch.no_grad():
            # Start token
            word_idx = self.vocab.word2idx["<start>"]
            caption = [self.vocab.idx2word[word_idx]]
            
            for _ in range(max_length):
                # Get next word prediction
                output = self.model(torch.tensor([word_idx]).to(self.device), 
                                  image_features.unsqueeze(0))
                
                # Sample next word
                logits = output[:, -1, :] / temperature
                probabilities = F.softmax(logits, dim=-1)
                word_idx = torch.multinomial(probabilities, 1).item()
                
                word = self.vocab.idx2word.get(word_idx, "<unk>")
                caption.append(word)
                
                if word == "<end>":
                    break
            
            return " ".join(caption[:-1])  # Remove <end> token
    
    def evaluate_dataset(self, data_loader, reference_file=None):
        """
        Evaluate model on entire dataset.
        
        Args:
            data_loader: DataLoader with test data
            reference_file: JSON file with reference captions
        
        Returns:
            Dictionary with evaluation metrics
        """
        total_bleu = 0
        num_samples = 0
        
        for images, _ in data_loader:
            images = images.to(self.device)
            
            for img_features in images:
                caption = self.generate_caption(img_features)
                # TODO: Compare with reference captions and compute BLEU
                num_samples += 1
        
        avg_bleu = total_bleu / max(num_samples, 1)
        
        return {
            'avg_bleu_score': avg_bleu,
            'num_samples': num_samples
        }


if __name__ == "__main__":
    print("Run this module via eval.py for model evaluation.")
