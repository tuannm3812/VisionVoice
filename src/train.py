"""
Training loop for the image captioning model.

This module handles model training, validation, checkpointing, and optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
from datetime import datetime


class Trainer:
    """Trainer class for the image captioning model."""
    
    def __init__(self, model, device='cpu', checkpoint_dir='./models'):
        """
        Initialize trainer.
        
        Args:
            model: The image captioning model
            device: Device to train on ('cpu' or 'cuda')
            checkpoint_dir: Directory to save model checkpoints
        """
        self.model = model
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.optimizer = None
        self.criterion = None
        self.best_loss = float('inf')
        self.training_history = []
    
    def setup_optimizer(self, lr=0.001, weight_decay=1e-5):
        """Setup optimizer for training."""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    
    def setup_loss(self):
        """Setup loss function."""
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    def train_epoch(self, train_loader, epoch):
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
        
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (images, captions) in enumerate(train_loader):
            images = images.to(self.device)
            captions = captions.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(captions, images)  # [B, seq_len, vocab_size]
            
            # Compute loss
            loss = self.criterion(
                outputs.view(-1, outputs.size(-1)),
                captions.view(-1)
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader, epoch):
        """
        Validate model on validation set.
        
        Args:
            val_loader: DataLoader for validation data
            epoch: Current epoch number
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for images, captions in val_loader:
                images = images.to(self.device)
                captions = captions.to(self.device)
                
                outputs = self.model(captions, images)
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)),
                    captions.view(-1)
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Validation Loss (Epoch {epoch}): {avg_loss:.4f}")
        return avg_loss
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'timestamp': datetime.now().isoformat()
        }
        
        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")
    
    def train(self, train_loader, val_loader, num_epochs=10, lr=0.001):
        """
        Complete training loop.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of epochs to train
            lr: Learning rate
        """
        self.setup_optimizer(lr=lr)
        self.setup_loss()
        
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch + 1)
            print(f"Training Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate(val_loader, epoch + 1)
            
            # Save checkpoint
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            self.save_checkpoint(epoch + 1, val_loss, is_best=is_best)
            
            # Record history
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss
            })
        
        # Save training history
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"\nTraining complete! History saved to {history_path}")


if __name__ == "__main__":
    print("Run this module via train.py to start training the model.")
