import torch
import torch.nn as nn
from models.extractors.lexical_stream import LexicalStream
from models.extractors.visual_stream import VisualStream
from models.extractors.acoustic_stream import AcousticStream
from models.core.hcrcaf_tri import HCRCAFFusionTri

class MultiModalCNNTri(nn.Module):
    """
    The orchestrator for the Tri-Modal Emotion Recognition System.
    Passes inputs through Unimodal Extractors, fuses them with HCR-CAF,
    and returns logits alongside features for Contrastive Loss.
    """
    def __init__(self, num_classes=8, embed_dim=128, device='cpu'):
        super(MultiModalCNNTri, self).__init__()
        self.device = device
        
        # 1. Unimodal Extractors (Intake Layer)
        self.lexical_model = LexicalStream(embed_dim=embed_dim, device=device)
        self.visual_model = VisualStream(embed_dim=embed_dim, device=device)
        self.acoustic_model = AcousticStream(embed_dim=embed_dim, device=device)
        
        # 2. Tri-Modal CAF Fusion
        self.fusion_block = HCRCAFFusionTri(embed_dim=embed_dim, num_heads=4).to(device)
        
        # 3. Final Classifier (Takes concatenated 3x embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        ).to(device)

    def forward(self, text_list, x_frames, x_audio):
        # 1. Extract Unimodal Features
        # Output shapes: (Batch, Seq_Len, Embed_Dim)
        l_seq = self.lexical_model(text_list)
        v_seq = self.visual_model(x_frames)
        a_seq = self.acoustic_model(x_audio)
        
        # Sequence lengths might differ between text (tokens) and video/audio (frames)
        # For simplicity in CAF, we pool them to global context vectors per stream first, 
        # OR we pad/interpolate them. Since this is an optimized CPU pass, we pool to 
        # (Batch, 1, Embed_Dim) before cross-attention to act as global semantic routers.
        l_seq = l_seq.mean(dim=1, keepdim=True)
        v_seq = v_seq.mean(dim=1, keepdim=True)
        a_seq = a_seq.mean(dim=1, keepdim=True)
        
        # 2. Tri-Way Cross-Attention Fusion
        l_fused, v_fused, a_fused = self.fusion_block(l_seq, v_seq, a_seq)
        
        # Squeeze sequence dimension: (Batch, Embed_Dim)
        l_pooled = l_fused.squeeze(1)
        v_pooled = v_fused.squeeze(1)
        a_pooled = a_fused.squeeze(1)
        
        # 3. Concatenate for Joint Representation
        joint_representation = torch.cat([l_pooled, v_pooled, a_pooled], dim=-1)
        
        # 4. Classification
        logits = self.classifier(joint_representation)
        
        return logits, l_pooled, v_pooled, a_pooled, joint_representation
