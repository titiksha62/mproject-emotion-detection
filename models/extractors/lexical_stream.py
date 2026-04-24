import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel

class LexicalStream(nn.Module):
    """
    Lexical extraction using RoBERTa. 
    Optimized for CPU by using distilroberta-base.
    """
    def __init__(self, embed_dim=128, device='cpu'):
        super(LexicalStream, self).__init__()
        self.device = device
        # Using distilroberta for CPU performance
        self.tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
        self.roberta = RobertaModel.from_pretrained('distilroberta-base')
        
        # Freeze early layers to save CPU memory and compute during fine-tuning
        for param in self.roberta.parameters():
            param.requires_grad = False
            
        # Project RoBERTa's 768-dim output to our shared latent space (e.g., 128)
        self.projection = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embed_dim)
        )
        self.to(self.device)

    def forward(self, text_list):
        """
        Takes a list of transcript strings and returns embeddings.
        Returns shape: (Batch, Seq_Len, Embed_Dim)
        """
        # Tokenize
        encoded = self.tokenizer(
            text_list, 
            padding=True, 
            truncation=True, 
            max_length=50, 
            return_tensors='pt'
        ).to(self.device)
        
        # Extract features
        outputs = self.roberta(**encoded)
        # Sequence output shape: (Batch, Seq_Len, 768)
        sequence_output = outputs.last_hidden_state 
        
        # Project to shared space
        projected = self.projection(sequence_output)
        return projected
