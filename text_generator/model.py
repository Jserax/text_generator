import torch
import yaml
from layers import PositionalEncoding, TransformerBlock
from torch import nn


class GazdanovPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        n_heads: int,
        emb_dim: int,
        ff_dim: int,
        dropout: float,
        flash: bool,
        max_len: int,
    ):
        super(GazdanovPT, self).__init__()
        self.config = {
            "vocab_size": vocab_size,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "emb_dim": emb_dim,
            "ff_dim": ff_dim,
            "dropout": dropout,
            "flash": flash,
            "max_len": max_len,
        }
        transformer_cfg = {
            "n_heads": n_heads,
            "emb_dim": emb_dim,
            "ff_dim": ff_dim,
            "dropout": dropout,
            "flash": flash,
            "max_len": max_len,
        }
        self.transformers = nn.ModuleList(
            [TransformerBlock(**transformer_cfg) for _ in range(n_layers)]
        )
        self.word_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = PositionalEncoding(emb_dim, max_len)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.word_emb(x)
        x = self.pos_emb(x)
        x = self.dropout(x)
        for block in self.transformers:
            x = block(x)
        x = self.layernorm(x)
        out = self.head(x)
        return out

    @classmethod
    def load_model(
        cls,
        model_path: str,
    ) -> "GazdanovPT":
        with open(f"{model_path}/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        model = GazdanovPT(**config)
        model.load_state_dict(torch.load(f"{model_path}/model.pt"))
        return model

    def save_model(
        self,
        model_path: str,
    ) -> None:
        with open(f"{model_path}/config.yaml", "w") as f:
            yaml.safe_dump(self.config, f)
        torch.save(self.state_dict(), f"{model_path}/model.pt")
