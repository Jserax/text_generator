import torch
from dataset import Dataset
from model import GazdanovPT
from tqdm import tqdm
from utils import CosineScheduler


model_cfg = {
    "vocab_size": 5031,
    "n_layers": 4,
    "n_heads": 8,
    "emb_dim": 256,
    "ff_dim": 256,
    "dropout": 0.3,
    "max_len": 256,
    "flash": False,
}

epochs = 100
batch_size = 64
lr = 3e-2
min_lr = 3e-6
weight_decay = 1e-02
grad_clip = 2.0
device = "cuda" if torch.cuda.is_available() else "cpu"
num_workers = 1
warmup_epochs = 4


def train():
    train_loader = torch.utils.data.DataLoader(
        Dataset("data/data.pt", model_cfg["max_len"]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    epoch_iters = len(train_loader)

    model = GazdanovPT(**model_cfg)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    scheduler = CosineScheduler(
        optimizer,
        warmup_epochs * epoch_iters,
        (epochs - 1) * epoch_iters,
        min_lr,
        lr,
    )
    criterion = torch.nn.CrossEntropyLoss()
    progress = tqdm(range(epochs))

    for _ in progress:
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= epoch_iters
        progress.postfix(f" | Train_loss: {train_loss:.4f}")


if __name__ == "__main__":
    train()
