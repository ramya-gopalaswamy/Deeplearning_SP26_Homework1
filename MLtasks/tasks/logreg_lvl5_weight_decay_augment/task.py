import os
import random
import sys
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def get_task_metadata() -> Dict:
    return {
        "id": "logreg_lvl5_weight_decay_augment",
        "series": "Logistic Regression",
        "level": 5,
        "description": "Binary logistic regression with L2 weight decay and Gaussian noise feature augmentation.",
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


def _make_gaussian_blobs(n: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
    n_per_class = n // 2
    mean0 = torch.tensor([-1.0, -1.0])
    mean1 = torch.tensor([1.0, 1.0])
    cov = torch.tensor([[0.8, 0.2], [0.2, 0.8]])
    L = torch.linalg.cholesky(cov)
    z0 = torch.randn(n_per_class, 2) @ L.T + mean0
    z1 = torch.randn(n_per_class, 2) @ L.T + mean1
    x = torch.cat([z0, z1], dim=0)
    y = torch.cat([torch.zeros(n_per_class), torch.ones(n_per_class)], dim=0).unsqueeze(1)
    perm = torch.randperm(x.size(0))
    x = x[perm]
    y = y[perm]
    x = (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True)
    return x, y


def make_dataloaders(batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    x, y = _make_gaussian_blobs(1000)
    n_train = int(0.8 * x.size(0))
    x_train, x_val = x[:n_train], x[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


class LogisticRegressor(nn.Module):
    def __init__(self, in_dim: int = 2) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def build_model(device: torch.device) -> nn.Module:
    model = LogisticRegressor().to(device)
    return model


def _classification_metrics(
    y_true: torch.Tensor, y_pred: torch.Tensor
) -> Dict[str, float]:
    y_true = y_true.view(-1).int()
    y_pred = y_pred.view(-1).int()
    tp = int(((y_true == 1) & (y_pred == 1)).sum().item())
    tn = int(((y_true == 0) & (y_pred == 0)).sum().item())
    fp = int(((y_true == 0) & (y_pred == 1)).sum().item())
    fn = int(((y_true == 1) & (y_pred == 0)).sum().item())

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
    rec = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
    if prec + rec == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 0.05,
    weight_decay: float = 1e-2,
    noise_std: float = 0.1,
) -> Dict:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
    )

    history = {"train_loss": [], "val_loss": []}

    for _ in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            noise = noise_std * torch.randn_like(xb)
            xb_noisy = xb + noise
            optimizer.zero_grad()
            logits = model(xb_noisy)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
        epoch_val_loss = val_loss / len(val_loader.dataset)
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)

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
                logits = model(xb)
                prob = torch.sigmoid(logits)
                y_hat = (prob > 0.5).float().cpu()
                ys.append(yb.cpu())
                preds.append(y_hat)
                probs_list.append(prob.cpu())
        y_cat = torch.cat(ys, dim=0)
        p_cat = torch.cat(preds, dim=0)
        prob_cat = torch.cat(probs_list, dim=0)
        cls = _classification_metrics(y_cat, p_cat)
        # MSE and R² between true 0/1 and predicted probability (protocol)
        mse = ((y_cat - prob_cat) ** 2).mean().item()
        ss_res = ((y_cat - prob_cat) ** 2).sum()
        ss_tot = ((y_cat - y_cat.mean()) ** 2).sum()
        r2 = (1.0 - ss_res / ss_tot).item() if ss_tot.item() > 0 else 0.0
        cls["mse"] = float(mse)
        cls["r2"] = float(r2)
        return cls

    train_metrics = _eval_loader(train_loader)
    val_metrics = _eval_loader(val_loader)

    metrics = {
        "train_mse": train_metrics["mse"],
        "train_r2": train_metrics["r2"],
        "train_accuracy": train_metrics["accuracy"],
        "train_precision": train_metrics["precision"],
        "train_recall": train_metrics["recall"],
        "train_f1": train_metrics["f1"],
        "val_mse": val_metrics["mse"],
        "val_r2": val_metrics["r2"],
        "val_accuracy": val_metrics["accuracy"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1": val_metrics["f1"],
    }
    return metrics


def predict(model: nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        prob = torch.sigmoid(logits)
    return prob.cpu()


def save_artifacts(outputs: Dict, output_dir: str = "artifacts") -> None:
    os.makedirs(output_dir, exist_ok=True)
    torch.save(outputs, os.path.join(output_dir, "logreg_lvl5_results.pt"))


def main() -> int:
    set_seed(123)
    device = get_device()
    train_loader, val_loader = make_dataloaders(batch_size=64)
    model = build_model(device)

    history = train(model, train_loader, val_loader, device)
    metrics = evaluate(model, train_loader, val_loader, device)

    outputs = {"history": history, "metrics": metrics}
    save_artifacts(outputs)

    print("Validation accuracy:", metrics["val_accuracy"])
    print("Validation precision:", metrics["val_precision"])
    print("Validation recall:", metrics["val_recall"])
    print("Validation F1:", metrics["val_f1"])

    success = (metrics["val_accuracy"] > 0.9) and (metrics["val_f1"] > 0.9)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

