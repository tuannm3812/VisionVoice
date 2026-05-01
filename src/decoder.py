"""
Sequence model decoder with attention mechanism.

This module implements an LSTM/RNN-based decoder with attention
for generating image captions word by word.
"""

import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    """Attention mechanism for image captioning."""
    
    def __init__(self, hidden_dim, feature_dim):
        """
        Initialize attention layer.
        
        Args:
            hidden_dim (int): Hidden dimension of LSTM
            feature_dim (int): Dimension of image features
        """
        super(AttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        
        # Attention computation layers
        self.fc_hidden = nn.Linear(hidden_dim, 512)
        self.fc_feature = nn.Linear(feature_dim, 512)
        self.fc_attention = nn.Linear(512, 1)
    
    def forward(self, hidden_state, features):
        """
        Compute attention weights over image features.
        
        Args:
            hidden_state: Current LSTM hidden state [B, hidden_dim]
            features: Image features [B, feature_dim]
        
        Returns:
            context: Weighted combination of features [B, feature_dim]
            weights: Attention weights [B]
        """
        # Compute attention scores
        h_t = self.fc_hidden(hidden_state)  # [B, 512]
        f_t = self.fc_feature(features)      # [B, 512]
        
        attention_logits = self.fc_attention(torch.tanh(h_t + f_t))  # [B, 1]
        attention_weights = torch.softmax(attention_logits, dim=0)    # [B, 1]
        
        # Apply attention to features
        context = (features * attention_weights).sum(dim=0, keepdim=True)  # [B, feature_dim]
        
        return context, attention_weights.squeeze()


class CaptionDecoder(nn.Module):
    """LSTM-based decoder with attention for caption generation."""
    
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, 
                 feature_dim=2048, num_layers=1, dropout=0.5):
        """
        Initialize caption decoder.
        
        Args:
            vocab_size (int): Size of vocabulary
            embedding_dim (int): Word embedding dimension
            hidden_dim (int): LSTM hidden dimension
            feature_dim (int): Image feature dimension
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout probability
        """
        super(CaptionDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim + feature_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention layer
        self.attention = AttentionLayer(hidden_dim, feature_dim)
        
        # Output projection
        self.fc_output = nn.Linear(hidden_dim + feature_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, captions, features, hidden_state=None):
        """
        Generate captions for images.
        
        Args:
            captions: Input caption tokens [B, seq_len]
            features: Image features [B, feature_dim]
            hidden_state: Initial LSTM hidden state (optional)
        
        Returns:
            outputs: Predicted word logits [B, seq_len, vocab_size]
        """
        batch_size = captions.size(0)
        
        # Embed captions
        embedded = self.embedding(captions)  # [B, seq_len, embedding_dim]
        
        # Concatenate features to each time step
        features_expanded = features.unsqueeze(1).expand(batch_size, embedded.size(1), -1)
        lstm_input = torch.cat([embedded, features_expanded], dim=-1)  # [B, seq_len, embedding_dim + feature_dim]
        
        # LSTM forward pass
        lstm_output, (h_n, c_n) = self.lstm(lstm_input, hidden_state)  # [B, seq_len, hidden_dim]
        
        # Apply attention
        context, _ = self.attention(h_n[-1], features)  # [1, feature_dim]
        context = context.expand(batch_size, -1)
        
        # Combine LSTM output with attention context
        combined = torch.cat([lstm_output, context.unsqueeze(1).expand_as(lstm_output[:, :, :0]) + context.unsqueeze(1)], dim=-1)
        
        # Project to vocabulary
        outputs = self.fc_output(combined)  # [B, seq_len, vocab_size]
        
        return outputs
