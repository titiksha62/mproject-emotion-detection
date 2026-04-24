import torch
import torch.nn as nn
import torch.nn.functional as F

class HCRCAFFusionTri(nn.Module):
    """
    Tri-Modal Hierarchical Contrastive Representation & Cross-Attention Fusion.
    The Lexical stream (Text) acts as the primary semantic Query, guiding the
    attention applied to the Visual and Acoustic streams.
    """
    def __init__(self, embed_dim=128, num_heads=4):
        super(HCRCAFFusionTri, self).__init__()
        
        # Cross-Attention: Lexical -> Visual
        self.lv_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        # Cross-Attention: Lexical -> Acoustic
        self.la_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        
        self.proj_v = nn.Linear(embed_dim, embed_dim)
        self.proj_a = nn.Linear(embed_dim, embed_dim)
        self.proj_l = nn.Linear(embed_dim, embed_dim)
        
        self.norm_v = nn.LayerNorm(embed_dim)
        self.norm_a = nn.LayerNorm(embed_dim)
        self.norm_l = nn.LayerNorm(embed_dim)
        
        # Gating mechanisms
        self.gate_v = nn.Parameter(torch.tensor(0.5))
        self.gate_a = nn.Parameter(torch.tensor(0.5))
        self.gate_l = nn.Parameter(torch.tensor(0.5))

    def forward(self, l_seq, v_seq, a_seq):
        """
        l_seq, v_seq, a_seq all shape (Batch, Seq_Len, Embed_Dim)
        """
        # Lexical queries Visual (Where to look based on what is said)
        v_ctx, _ = self.lv_attn(query=l_seq, key=v_seq, value=v_seq)
        
        # Lexical queries Acoustic (What tone to listen for based on what is said)
        a_ctx, _ = self.la_attn(query=l_seq, key=a_seq, value=a_seq)
        
        # Gated fusion for Visual
        g_v = torch.sigmoid(self.gate_v)
        v_fused = g_v * self.proj_v(v_ctx) + (1 - g_v) * v_seq
        v_fused = self.norm_v(v_fused)
        
        # Gated fusion for Acoustic
        g_a = torch.sigmoid(self.gate_a)
        a_fused = g_a * self.proj_a(a_ctx) + (1 - g_a) * a_seq
        a_fused = self.norm_a(a_fused)
        
        # Gated residual for Lexical
        g_l = torch.sigmoid(self.gate_l)
        l_fused = g_l * self.proj_l(l_seq) + (1 - g_l) * l_seq
        l_fused = self.norm_l(l_fused)
        
        return l_fused, v_fused, a_fused

class HierarchicalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, num_classes=8, embed_dim=128*3):
        super(HierarchicalContrastiveLoss, self).__init__()
        self.temperature = temperature
        # Semantic Anchors (Class Prototypes)
        self.class_prototypes = nn.Parameter(torch.randn(num_classes, embed_dim))
        
    def info_nce(self, feat1, feat2):
        """Instance-level alignment between two streams"""
        f1 = F.normalize(feat1, dim=1)
        f2 = F.normalize(feat2, dim=1)
        logits = torch.matmul(f1, f2.T) / self.temperature
        labels = torch.arange(f1.size(0), device=f1.device)
        return F.cross_entropy(logits, labels)
        
    def forward(self, l_feat, v_feat, a_feat, joint_feat, labels):
        """
        Calculates both Instance-Level and Semantic-Level Contrastive Loss.
        """
        # 1. Instance-Level Alignment (Push streams of same sample together)
        loss_lv = self.info_nce(l_feat, v_feat)
        loss_la = self.info_nce(l_feat, a_feat)
        loss_va = self.info_nce(v_feat, a_feat)
        instance_loss = (loss_lv + loss_la + loss_va) / 3.0
        
        # 2. Semantic-Level Alignment (Push joint feat to correct class prototype)
        j_feat_norm = F.normalize(joint_feat, dim=1)
        proto_norm = F.normalize(self.class_prototypes, dim=1)
        
        # Dot product between features and prototypes -> (Batch, Num_Classes)
        semantic_logits = torch.matmul(j_feat_norm, proto_norm.T) / self.temperature
        semantic_loss = F.cross_entropy(semantic_logits, labels)
        
        return instance_loss, semantic_loss
