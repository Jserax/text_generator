import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        max_len: int,
    ):
        self.tokens = torch.load(data_path)
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.tokens) - self.max_len

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tokens[idx : idx + self.max_len]
        y = self.tokens[idx + 1 : idx + self.max_len + 1]
        return x, y
