import torch
from dataset import Dataset
from model import GazdanovPT
from tqdm import tqdm
from utils import CosineScheduler


model_cfg = {
    "vocab_size": 30522,
    "n_layers": 4,
    "n_heads": 8,
    "emb_dim": 256,
    "ff_dim": 512,
    "dropout": 0.1,
    "max_len": 256,
    "flash": True,
}

epochs = 30
batch_size = 256
lr = 3e-3
min_lr = 1e-5
weight_decay = 3e-02
grad_clip = 1.0
device = "cuda" if torch.cuda.is_available() else "cpu"
num_workers = 2
warmup_epochs = 2


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
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    scheduler = CosineScheduler(
        optimizer,
        warmup_epochs * epoch_iters,
        (epochs - 2) * epoch_iters,
        min_lr,
        lr,
    )
    scaler = torch.cuda.amp.GradScaler()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    progress = tqdm(range(epochs))

    for epoch in progress:
        train_loss = 0.0
        for iter, (x, y) in tqdm(enumerate(train_loader), total=epoch_iters):
            x = x.to(device)
            y = y.to(device)
            scheduler.step()
            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.float16):
                pred = model(x).permute(0, 2, 1)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            if iter % (epoch_iters // 5) == 0:
                print(
                    f"Epochs: {epoch + 1}/{epochs} | Iters: {iter+1}/{epoch_iters} | Train_loss {train_loss / (iter+1):.4f}"
                )
    model.save_model("/content/text_generator")


if __name__ == "__main__":
    train()
