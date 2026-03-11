import os
import random
import sys
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def get_task_metadata() -> Dict:
    return {
        "id": "rnn_lvl5_gru_time_series",
        "series": "Sequence Models (RNN/LSTM)",
        "level": 5,
        "description": "GRU-based regression on synthetic noisy sine-wave time series.",
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


def _generate_sine_series(
    n_steps: int = 2000, noise_std: float = 0.1
) -> torch.Tensor:
    t = torch.linspace(0, 20 * torch.pi, n_steps)
    y = torch.sin(t) + 0.5 * torch.sin(0.5 * t)
    y = y + noise_std * torch.randn_like(y)
    return y.unsqueeze(1)


def _build_windows(
    series: torch.Tensor, window: int = 20
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for i in range(len(series) - window):
        xs.append(series[i : i + window])
        ys.append(series[i + window])
    x = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0)
    return x, y


def make_dataloaders(
    window: int = 20, batch_size: int = 64
) -> Tuple[DataLoader, DataLoader]:
    series = _generate_sine_series()
    x, y = _build_windows(series, window=window)
    n_train = int(0.8 * x.size(0))
    x_train, x_val = x[:n_train], x[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


class GRURegressor(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def build_model(device: torch.device) -> nn.Module:
    model = GRURegressor().to(device)
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
    epochs: int = 60,
    lr: float = 1e-3,
    patience: int = 8,
) -> Dict:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for _ in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss = val_loss / len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
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

    return {
        "train_mse": float(train_mse),
        "train_r2": float(train_r2),
        "val_mse": float(val_mse),
        "val_r2": float(val_r2),
    }


def predict(model: nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        out = model(x.to(device))
    return out.cpu()


def save_artifacts(outputs: Dict, output_dir: str = "artifacts") -> None:
    os.makedirs(output_dir, exist_ok=True)
    torch.save(outputs, os.path.join(output_dir, "rnn_lvl5_results.pt"))


def main() -> int:
    set_seed(99)
    device = get_device()
    train_loader, val_loader = make_dataloaders(window=20, batch_size=64)
    model = build_model(device)

    history = train(model, train_loader, val_loader, device)
    metrics = evaluate(model, train_loader, val_loader, device)

    outputs = {"history": history, "metrics": metrics}
    save_artifacts(outputs)

    print("Train MSE:", metrics["train_mse"])
    print("Train R2:", metrics["train_r2"])
    print("Val MSE:", metrics["val_mse"])
    print("Val R2:", metrics["val_r2"])

    success = (metrics["val_r2"] > 0.9) and (metrics["val_mse"] < 0.05)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

