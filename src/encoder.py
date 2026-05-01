"""
CNN Feature Extractor for image encoding.

This module uses a pretrained CNN (VGG/ResNet) to extract visual features
from images for the attention-based image captioning model.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    """CNN-based image encoder using pretrained models."""
    
    def __init__(self, model_type='resnet50', pretrained=True, feature_dim=2048):
        """
        Initialize the image encoder.
        
        Args:
            model_type (str): Type of pretrained model ('resnet50', 'vgg16', etc.)
            pretrained (bool): Whether to use pretrained weights
            feature_dim (int): Dimension of extracted features
        """
        super(ImageEncoder, self).__init__()
        self.model_type = model_type
        self.feature_dim = feature_dim
        
        # Load pretrained model
        if model_type == 'resnet50':
            self.encoder = models.resnet50(pretrained=pretrained)
            # Remove the final classification layer
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        elif model_type == 'vgg16':
            self.encoder = models.vgg16(pretrained=pretrained)
            # Keep features up to last conv layer
            self.encoder = self.encoder.features
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Freeze pretrained weights
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, images):
        """
        Extract features from images.
        
        Args:
            images: Batch of images [B, C, H, W]
        
        Returns:
            features: Image features [B, feature_dim]
        """
        with torch.no_grad():
            features = self.encoder(images)
            features = features.view(features.size(0), -1)
        
        return features
