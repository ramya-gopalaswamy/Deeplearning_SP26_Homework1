import os
import random
import sys
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


def get_task_metadata() -> Dict:
    return {
        "id": "cnn_lvl5_augmented_mnist_like",
        "series": "Convolutional Neural Networks",
        "level": 5,
        "description": "Small CNN on synthetic MNIST-like blobs with data augmentation and early stopping.",
    }


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_blob_images(
    n_samples: int = 2000, img_size: int = 16, n_classes: int = 3
) -> Tuple[torch.Tensor, torch.Tensor]:
    images = torch.zeros(n_samples, 1, img_size, img_size)
    labels = torch.zeros(n_samples, 1)
    centers = [
        (4, 4),
        (4, 11),
        (11, 8),
    ]
    sigma = 2.0
    xs = torch.arange(img_size).view(1, -1).float()
    ys = torch.arange(img_size).view(-1, 1).float()
    for i in range(n_samples):
        cls = random.randint(0, n_classes - 1)
        cy, cx = centers[cls]
        dist2 = (xs - cx) ** 2 + (ys - cy) ** 2
        blob = torch.exp(-dist2 / (2 * sigma**2))
        noise = 0.1 * torch.randn_like(blob)
        img = blob + noise
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        images[i, 0] = img
        labels[i, 0] = cls
    perm = torch.randperm(n_samples)
    images = images[perm]
    labels = labels[perm]
    return images, labels.long()


def make_dataloaders(batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    x, y = _make_blob_images()
    n_train = int(0.8 * x.size(0))
    x_train, x_val = x[:n_train], x[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(device: torch.device) -> nn.Module:
    model = SmallCNN().to(device)
    return model


def _augment_batch(x: torch.Tensor) -> torch.Tensor:
    # simple random horizontal flip
    if random.random() < 0.5:
        x = torch.flip(x, dims=[3])
    return x


def _classification_metrics(
    y_true: torch.Tensor, y_pred: torch.Tensor, n_classes: int
) -> Dict[str, float]:
    y_true = y_true.view(-1).long()
    y_pred = y_pred.view(-1).long()
    acc = (y_true == y_pred).float().mean().item()
    f1_per_class = []
    for c in range(n_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum().item())
        fp = int(((y_true != c) & (y_pred == c)).sum().item())
        fn = int(((y_true == c) & (y_pred != c)).sum().item())
        prec = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
        rec = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
        if prec + rec == 0:
            f1 = 0.0
        else:
            f1 = 2 * prec * rec / (prec + rec)
        f1_per_class.append(f1)
    macro_f1 = float(sum(f1_per_class) / len(f1_per_class))
    return {"accuracy": float(acc), "macro_f1": macro_f1}


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 40,
    lr: float = 1e-3,
    patience: int = 5,
) -> Dict:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None
    epochs_without_improve = 0

    for _ in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.view(-1).to(device)
            xb = _augment_batch(xb)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        ys = []
        preds = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.view(-1).to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                y_hat = torch.argmax(logits, dim=1)
                ys.append(yb.cpu())
                preds.append(y_hat.cpu())
        val_loss = val_loss / len(val_loader.dataset)
        y_cat = torch.cat(ys, dim=0)
        p_cat = torch.cat(preds, dim=0)
        metrics = _classification_metrics(y_cat, p_cat, n_classes=3)
        val_acc = metrics["accuracy"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict:
    model.eval()

    def _eval_loader(loader: DataLoader) -> Dict[str, float]:
        ys = []
        preds = []
        probs_list = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.view(-1).to(device)
                logits = model(xb)
                probs = F.softmax(logits, dim=1)
                y_hat = torch.argmax(logits, dim=1)
                ys.append(yb.cpu())
                preds.append(y_hat.cpu())
                probs_list.append(probs.cpu())
        y_cat = torch.cat(ys, dim=0)
        p_cat = torch.cat(preds, dim=0)
        prob_cat = torch.cat(probs_list, dim=0)
        out = _classification_metrics(y_cat, p_cat, n_classes=3)
        # MSE and R² between one-hot labels and predicted probs (protocol)
        one_hot = F.one_hot(y_cat.long(), num_classes=3).float()
        mse = ((one_hot - prob_cat) ** 2).mean().item()
        ss_res = ((one_hot - prob_cat) ** 2).sum()
        ss_tot = ((one_hot - one_hot.mean()) ** 2).sum()
        r2 = (1.0 - ss_res / ss_tot).item() if ss_tot.item() > 0 else 0.0
        out["mse"] = float(mse)
        out["r2"] = float(r2)
        return out

    train_metrics = _eval_loader(train_loader)
    val_metrics = _eval_loader(val_loader)

    return {
        "train_mse": train_metrics["mse"],
        "train_r2": train_metrics["r2"],
        "train_accuracy": train_metrics["accuracy"],
        "train_macro_f1": train_metrics["macro_f1"],
        "val_mse": val_metrics["mse"],
        "val_r2": val_metrics["r2"],
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
    }


def predict(model: nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        probs = torch.softmax(logits, dim=1)
    return probs.cpu()


def save_artifacts(outputs: Dict, output_dir: str = "artifacts") -> None:
    os.makedirs(output_dir, exist_ok=True)
    torch.save(outputs, os.path.join(output_dir, "cnn_lvl5_results.pt"))


def main() -> int:
    set_seed(7)
    device = get_device()
    train_loader, val_loader = make_dataloaders(batch_size=64)
    model = build_model(device)

    history = train(model, train_loader, val_loader, device)
    metrics = evaluate(model, train_loader, val_loader, device)

    outputs = {"history": history, "metrics": metrics}
    save_artifacts(outputs)

    print("Validation accuracy:", metrics["val_accuracy"])
    print("Validation macro-F1:", metrics["val_macro_f1"])

    success = metrics["val_accuracy"] > 0.9
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

