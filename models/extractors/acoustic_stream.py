import torch
import torch.nn as nn

def conv1d_block_audio(in_channels, out_channels, kernel_size=3, stride=1, padding='valid'):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True), 
        nn.MaxPool1d(2,1)
    )

class AcousticStream(nn.Module):
    """
    Acoustic extraction. While AST is powerful, 1D-CNN on MFCCs is 
    vastly superior for CPU-only inference. We utilize a deep 1D-CNN here.
    """
    def __init__(self, embed_dim=128, device='cpu'):
        super(AcousticStream, self).__init__()
        self.device = device
        
        self.conv1d_0 = conv1d_block_audio(10, 64)
        self.conv1d_1 = conv1d_block_audio(64, 128)
        self.conv1d_2 = conv1d_block_audio(128, 256)
        self.conv1d_3 = conv1d_block_audio(256, embed_dim)
        
        self.to(self.device)
            
    def forward(self, x):
        """
        Expects MFCC input of shape (Batch, 10, TimeSteps)
        Returns shape: (Batch, Seq_Len, Embed_Dim)
        """
        x = x.to(self.device)
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)   
        # Swap axes to match (Batch, Seq_Len, Embed_Dim)
        return x.permute(0, 2, 1)
