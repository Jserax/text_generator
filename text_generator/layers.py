import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        max_len: int,
    ):
        super().__init__()
        position = torch.arange(max_len).view(-1, 1)
        div_term = torch.exp(
            torch.arange(0, emb_dim, 2) * (-torch.log(torch.tensor(max_len)) / emb_dim)
        )
        pos = torch.zeros(max_len, 1, emb_dim)
        pos[:, 0, 0::2] = torch.sin(position * div_term)
        pos[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos", pos)

    def forward(self, x):
        x = x + self.pos[: x.size(0)]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        emb_dim: int,
        dropout: float,
        flash: bool,
        max_len: int,
    ) -> None:
        super(MultiHeadAttention, self).__init__()
        if emb_dim % n_heads != 0:
            raise Exception(
                f"Embedding dim(emb_dim): {emb_dim} must be divisible by number of heads(n_heads): {n_heads}"
            )
        self.n_heads = n_heads
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // n_heads
        self.dropout = dropout
        self.flash = flash
        self.to_qkv = nn.Linear(emb_dim, 3 * emb_dim)
        self.out = nn.Linear(emb_dim, emb_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "bias", torch.ones(max_len, max_len).tril()[None, None, :, :]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.size()
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.view(B, S, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, S, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, S, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        if self.flash:
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout,
                is_causal=True,
            )
        else:
            att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
            att = att.masked_fill(self.bias[:, :, :D, :D] == 0, float("-inf"))
            att = torch.nn.functional.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            out = torch.matmul(att, v)
        out = out.permute(0, 2, 1, 3).reshape(B, S, D)
        out = self.out_dropout(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_heads: int,
        emb_dim: int,
        ff_dim: int,
        dropout: float,
        flash: bool,
        max_len: int,
    ):
        super(TransformerBlock, self).__init__()
        self.layernorm1 = nn.LayerNorm(emb_dim)
        self.layernorm2 = nn.LayerNorm(emb_dim)
        self.attn = MultiHeadAttention(
            n_heads,
            emb_dim,
            dropout,
            flash,
            max_len,
        )
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, ff_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ff_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.layernorm1(x))
        x = x + self.ff(self.layernorm2(x))
        return x
