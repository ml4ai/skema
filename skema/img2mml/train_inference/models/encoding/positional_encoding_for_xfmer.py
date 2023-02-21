'''
PositionalEncoding class has been taken from PyTorch tutorials.
<Source>: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
'''

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, model_dimension, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, model_dimension)      # (max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)        # (max_len, 1)
        div_term = torch.exp(torch.arange(0, model_dimension, 2).float() * (-math.log(10000.0) / model_dimension))    # ([model_dim//2])
        pe[:, 0::2] = torch.sin(position * div_term)      # (max_len, model_dim//2)
        pe[:, 1::2] = torch.cos(position * div_term)      # (max_len, model_dim//2)
        pe = pe.unsqueeze(0).transpose(0, 1)    # (max_len, 1, model_dim)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # x: (max_len, B, embed_dim)
        # print("x shape:", x.shape)
        # print("x_ shape:", self.pe[:x.size(0), :].shape)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
