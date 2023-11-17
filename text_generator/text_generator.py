import re
from typing import Union

import torch
from model import GazdanovPT
from transformers import AutoTokenizer


class TextGenerator:
    def __init__(self, model_path: str, device: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            "DeepPavlov/distilrubert-tiny-cased-conversational-5k"
        )
        self.model = GazdanovPT.load_model(model_path)
        self.model.to(device)
        self.model.eval()

    def generate(
        self,
        text: str,
        max_tokens: int,
        temp: float,
        top_k: Union[int, float],
        sample_strategy: str,
    ):
        idxs = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        if sample_strategy == "beam":
            idxs = self._beamsearch(idxs, max_tokens, temp)
        elif sample_strategy == "greedy":
            idxs = self._greedysearch(idxs, max_tokens)
        elif sample_strategy == "random":
            idxs = self._randomsearch(idxs, max_tokens, temp, top_k)
        else:
            raise Exception(
                f"Sample strategy must be on of (beam, greedy, random), but get {sample_strategy}"
            )
        return re.sub(r"\s+(?=(?:[,.?!:;…]))", r"", " ".join(idxs).replace(" ##", ""))

    @torch.no_grad()
    def _randomsearch(
        self, idxs: torch.Tensor, max_tokens: int, temp: float, top_k: Union[int, float]
    ) -> torch.Tensor:
        for _ in range(max_tokens):
            logits = self.model(
                idxs[-max_tokens:] if idxs.size(0) >= max_tokens else idxs
            )
            logits = logits[:, -1, :] / temp
            if top_k is not None:
                if top_k is float:
                    top_k = int(top_k * self.model.config["vocab_size"])
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[-1]] = -float("Inf")
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idxs = torch.cat((idxs, idx_next), dim=-1)
        return idxs

    @torch.no_grad()
    def _greedysearch(self, idxs: torch.Tensor, max_tokens: int) -> torch.Tensor:
        for _ in range(max_tokens):
            logits = self.model(
                idxs[-max_tokens:] if idxs.size(0) >= max_tokens else idxs
            )
            idx_next = torch.argmax(logits, keepdim=True)
            idxs = torch.cat((idxs, idx_next), dim=-1)
        return idxs

    @torch.no_grad()
    def __beamsearch(
        self, idxs: torch.Tensor, max_tokens: int, temp: float
    ) -> torch.Tensor:
        pass
