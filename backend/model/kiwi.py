import torch
import math

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=1024) -> None:
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float()* -(math.log(10000.) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)
    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach()
    
class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(d_model, dim_feedforward),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_feedforward, d_model),
        )
        self.layer_norm1 = torch.nn.LayerNorm(d_model)
        self.layer_norm2 = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.layer_norm1(x)
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.layer_norm2(x)
        return x
    
class GPT2(torch.nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_len=1024):
        super(GPT2, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.transformer_blocks = torch.nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward) for _ in range(num_layers)
        ])
        self.fc_out = torch.nn.Linear(d_model, vocab_size)
    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for block in self.transformer_blocks:
            x = block(x)
        return self.fc_out(x)