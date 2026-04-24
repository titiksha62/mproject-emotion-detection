# %%writefile models/hcrcaf.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class HCRCAFFusion(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()
        self.av_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.va_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        
        self.proj_v = nn.Linear(embed_dim, embed_dim)
        self.proj_a = nn.Linear(embed_dim, embed_dim)
        
        self.norm_v = nn.LayerNorm(embed_dim)
        self.norm_a = nn.LayerNorm(embed_dim)
        
        self.gate_v = nn.Parameter(torch.tensor(0.5))
        self.gate_a = nn.Parameter(torch.tensor(0.5))

    def forward(self, a_seq, v_seq):
        a_ctx, _ = self.av_attn(a_seq, v_seq, v_seq)
        v_ctx, _ = self.va_attn(v_seq, a_seq, a_seq)
        
        g_v = torch.sigmoid(self.gate_v)
        v_fused = g_v * self.proj_v(v_ctx) + (1 - g_v) * v_seq
        v_fused = self.norm_v(v_fused)
        
        g_a = torch.sigmoid(self.gate_a)
        a_fused = g_a * self.proj_a(a_ctx) + (1 - g_a) * a_seq
        a_fused = self.norm_a(a_fused)
        
        return a_fused, v_fused

def info_nce_loss(v, a, temperature=0.07):
    v = F.normalize(v, dim=1)
    a = F.normalize(a, dim=1)
    logits = torch.matmul(v, a.T) / temperature
    labels = torch.arange(v.size(0), device=v.device)
    return F.cross_entropy(logits, labels)