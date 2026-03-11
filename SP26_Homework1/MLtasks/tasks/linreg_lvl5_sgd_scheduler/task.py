import math
import os
import random
import sys
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def get_task_metadata() -> Dict:
    return {
        "id": "linreg_lvl5_sgd_scheduler",
        "series": "Linear Regression",
        "level": 5,
        "description": "Univariate linear regression with mini-batch SGD, StepLR scheduler, and gradient clipping.",
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


def _make_synthetic_data(n: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
    true_w = torch.tensor([-1.5])
    true_b = torch.tensor([0.7])
    x = torch.linspace(-5.0, 5.0, n).unsqueeze(1)
    noise = 0.3 * torch.randn_like(x)
    y = x @ true_w.unsqueeze(1) + true_b + noise
    return x, y


def make_dataloaders(batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    x, y = _make_synthetic_data(1000)
    n_train = int(0.8 * x.shape[0])
    x_train, x_val = x[:n_train], x[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


class LinearRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def build_model(device: torch.device) -> nn.Module:
    model = LinearRegressor().to(device)
    return model


def _r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    if ss_tot.item() == 0.0:
        return 0.0
    return (1.0 - ss_res / ss_tot).item()


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 200,
    lr: float = 0.05,
    momentum: float = 0.9,
    clip_norm: float = 1.0,
) -> Dict:
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)
        epoch_val_loss = val_loss / len(val_loader.dataset)

        scheduler.step()

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)

    return history


def evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict:
    criterion = nn.MSELoss()
    model.eval()

    def _eval_loader(loader: DataLoader) -> Tuple[float, float]:
        total_loss = 0.0
        ys = []
        preds_all = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                total_loss += loss.item() * xb.size(0)
                ys.append(yb.cpu())
                preds_all.append(preds.cpu())
        y_cat = torch.cat(ys, dim=0)
        p_cat = torch.cat(preds_all, dim=0)
        mse = total_loss / len(loader.dataset)
        r2 = _r2_score(y_cat, p_cat)
        return mse, r2

    train_mse, train_r2 = _eval_loader(train_loader)
    val_mse, val_r2 = _eval_loader(val_loader)

    with torch.no_grad():
        lin: nn.Linear = model.lin  # type: ignore
        w = lin.weight.data.view(-1).cpu()
        b = lin.bias.data.view(-1).cpu()
    true_w = torch.tensor([-1.5])
    true_b = torch.tensor([0.7])
    param_error = torch.sqrt((w - true_w) ** 2 + (b - true_b) ** 2).item()

    metrics = {
        "train_mse": float(train_mse),
        "train_r2": float(train_r2),
        "val_mse": float(val_mse),
        "val_r2": float(val_r2),
        "param_l2_error": float(param_error),
        "learned_w": float(w.item()),
        "learned_b": float(b.item()),
    }
    return metrics


def predict(model: nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        y = model(x)
    return y.cpu()


def save_artifacts(outputs: Dict, output_dir: str = "artifacts") -> None:
    os.makedirs(output_dir, exist_ok=True)
    torch.save(outputs, os.path.join(output_dir, "linreg_lvl5_results.pt"))


def main() -> int:
    set_seed(42)
    device = get_device()
    train_loader, val_loader = make_dataloaders(batch_size=64)
    model = build_model(device)

    history = train(model, train_loader, val_loader, device)
    metrics = evaluate(model, train_loader, val_loader, device)

    outputs = {"history": history, "metrics": metrics}
    save_artifacts(outputs)

    print("Train MSE:", metrics["train_mse"])
    print("Train R2:", metrics["train_r2"])
    print("Val MSE:", metrics["val_mse"])
    print("Val R2:", metrics["val_r2"])
    print("Param L2 error:", metrics["param_l2_error"])
    print("Learned w, b:", metrics["learned_w"], metrics["learned_b"])

    success = (metrics["val_r2"] > 0.88) and (metrics["param_l2_error"] < 0.5)
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

