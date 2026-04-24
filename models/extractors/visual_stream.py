import torch
import torch.nn as nn
import numpy as np

class VisualStream(nn.Module):
    """
    Visual extraction combining spatial face embeddings (via a mocked InsightFace/CNN) 
    and temporal convolution (TCN) to capture micro-expressions over time.
    """
    def __init__(self, embed_dim=128, device='cpu'):
        super(VisualStream, self).__init__()
        self.device = device
        
        # Spatial Feature Extractor (Simulating InsightFace mapping to 512d)
        # Using a lightweight MobileNetV2 or similar ResNet block for CPU
        # Here we define a lightweight convolutional spatial extractor
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Output: [Batch, 128, 1, 1]
        )
        
        # Temporal Convolutional Network (TCN) to capture movement across frames
        self.temporal_tcn = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, embed_dim, kernel_size=3, padding=1)
        )
        
        self.to(self.device)

    def forward(self, x_frames):
        """
        x_frames: (Batch, Seq_Len, Channels, H, W) e.g. (B, 15, 3, 224, 224)
        Returns: (Batch, Seq_Len, Embed_Dim)
        """
        x_frames = x_frames.to(self.device)
        B, S, C, H, W = x_frames.shape
        
        # 1. Spatial Extraction (Fold Sequence into Batch)
        x_folded = x_frames.view(-1, C, H, W)
        spatial_feats = self.spatial_cnn(x_folded) # Shape: (B*S, 128, 1, 1)
        spatial_feats = spatial_feats.view(B, S, 128)
        
        # 2. Temporal Extraction (TCN requires [Batch, Channels, Seq_Len])
        spatial_feats = spatial_feats.permute(0, 2, 1)
        temporal_feats = self.temporal_tcn(spatial_feats) # Shape: (B, Embed_Dim, S)
        
        # 3. Return as (Batch, Seq_Len, Embed_Dim)
        return temporal_feats.permute(0, 2, 1)
