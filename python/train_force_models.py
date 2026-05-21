"""Train two separate neural-network surrogates for sphere-sphere forces.

Targets:
- vdW model:   F_vdw_scalar_pN  (negative / attractive)
- Elec model:  F_elec_scalar_pN (positive / repulsive)

The model learns log10(force magnitude) as a function of log10(gap_nm).
The sign is kept outside the network.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


EPS = 1e-12


@dataclass
class TrainConfig:
    hidden: int = 64
    epochs: int = 400
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-6
    patience: int = 40
    seed: int = 42


class ForceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x[:, None], dtype=torch.float32)
        self.y = torch.tensor(y[:, None], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_train_val_test(n: int, train_frac: float = 0.7, val_frac: float = 0.15, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    i_train = idx[:n_train]
    i_val = idx[n_train:n_train + n_val]
    i_test = idx[n_train + n_val:]
    return i_train, i_val, i_test


def prepare_xy(df: pd.DataFrame, gap_column: str, target_column: str, sign: str) -> Tuple[np.ndarray, np.ndarray]:
    gap = df[gap_column].to_numpy(dtype=np.float64)
    if np.any(gap <= 0.0):
        raise ValueError("All gaps must be strictly positive for log10(gap).")

    force = df[target_column].to_numpy(dtype=np.float64)
    if sign == "negative":
        mag = -force
    elif sign == "positive":
        mag = force
    else:
        raise ValueError("sign must be 'negative' or 'positive'.")

    if np.any(mag <= 0.0):
        raise ValueError(f"Target magnitude must be positive after applying sign convention for {target_column}.")

    x = np.log10(gap)
    y = np.log10(np.maximum(mag, EPS))
    return x, y


def standardize(train_values: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mu = float(np.mean(train_values))
    sigma = float(np.std(train_values))
    if sigma < 1e-12:
        sigma = 1.0
    out = (values - mu) / sigma
    return out, mu, sigma


def train_one_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: str, cfg: TrainConfig) -> Tuple[nn.Module, Dict[str, list]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    history = {"train": [], "val": []}
    bad_epochs = 0

    for epoch in range(cfg.epochs):
        model.train()
        running_train = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train += loss.item() * len(xb)
        train_loss = running_train / len(train_loader.dataset)

        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                running_val += loss.item() * len(xb)
        val_loss = running_val / len(val_loader.dataset)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % 25 == 0 or epoch == cfg.epochs - 1:
            print(f"epoch {epoch:4d} | train={train_loss:.6e} | val={val_loss:.6e}")

        if bad_epochs >= cfg.patience:
            print("Early stopping triggered.")
            break

    if best_state is None:
        raise RuntimeError("Training failed to produce a best_state.")
    model.load_state_dict(best_state)
    return model, history


def evaluate_model(model: nn.Module, x_std: np.ndarray, y_std: np.ndarray, device: str) -> Dict[str, np.ndarray]:
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(x_std[:, None], dtype=torch.float32, device=device)
        pred = model(xt).cpu().numpy().ravel()
    rmse = float(np.sqrt(np.mean((pred - y_std) ** 2)))
    mae = float(np.mean(np.abs(pred - y_std)))
    return {"pred_std": pred, "rmse_std": rmse, "mae_std": mae}


def save_training_plots(out_dir: str, tag: str, history: Dict[str, list], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(history["train"], label="train")
    plt.plot(history["val"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title(f"Loss history: {tag}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_loss.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=10, alpha=0.7)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    plt.xlabel("true log10|F|")
    plt.ylabel("pred log10|F|")
    plt.title(f"Parity: {tag}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_parity.png"), dpi=180)
    plt.close()


def train_force_model(csv_path: str, gap_column: str, target_column: str, sign: str, out_dir: str, device: str, cfg: TrainConfig) -> Dict[str, object]:
    df = pd.read_csv(csv_path)
    # Restrict to the physically desired regime for this model
    df = df[df[gap_column] > 0].copy()
    x_raw, y_raw = prepare_xy(df, gap_column=gap_column, target_column=target_column, sign=sign)

    i_train, i_val, i_test = split_train_val_test(len(df), seed=cfg.seed)

    x_train_raw = x_raw[i_train]
    y_train_raw = y_raw[i_train]
    x_val_raw = x_raw[i_val]
    y_val_raw = y_raw[i_val]
    x_test_raw = x_raw[i_test]
    y_test_raw = y_raw[i_test]

    x_train, x_mu, x_sigma = standardize(x_train_raw, x_train_raw)
    x_val = (x_val_raw - x_mu) / x_sigma
    x_test = (x_test_raw - x_mu) / x_sigma

    y_train, y_mu, y_sigma = standardize(y_train_raw, y_train_raw)
    y_val = (y_val_raw - y_mu) / y_sigma
    y_test = (y_test_raw - y_mu) / y_sigma

    train_loader = DataLoader(ForceDataset(x_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(ForceDataset(x_val, y_val), batch_size=max(cfg.batch_size, 256), shuffle=False)

    model = MLP(hidden=cfg.hidden).to(device)
    model, history = train_one_model(model, train_loader, val_loader, device, cfg)

    test_eval = evaluate_model(model, x_test, y_test, device)
    pred_test_raw = test_eval["pred_std"] * y_sigma + y_mu

    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"{target_column}_model.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "target_column": target_column,
        "gap_column": gap_column,
        "sign": sign,
        "x_mu": x_mu,
        "x_sigma": x_sigma,
        "y_mu": y_mu,
        "y_sigma": y_sigma,
        "train_config": asdict(cfg),
    }, ckpt_path)

    metrics = {
        "rmse_std": test_eval["rmse_std"],
        "mae_std": test_eval["mae_std"],
        "rmse_log10": float(np.sqrt(np.mean((pred_test_raw - y_test_raw) ** 2))),
        "mae_log10": float(np.mean(np.abs(pred_test_raw - y_test_raw))),
    }
    with open(os.path.join(out_dir, f"{target_column}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_training_plots(out_dir, target_column, history, y_test_raw, pred_test_raw)
    print(f"Saved model to {ckpt_path}")
    print(f"Metrics for {target_column}: {metrics}")

    return {
        "checkpoint": ckpt_path,
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--gap_column", type=str, default="gap_core_nm")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        hidden=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
    )
    set_seed(cfg.seed)

    vdw = train_force_model(args.csv, args.gap_column, "F_vdw_scalar_pN", "negative", args.out_dir, args.device, cfg)
    elec = train_force_model(args.csv, "gap_shell_nm", "F_elec_scalar_pN", "positive", args.out_dir, args.device, cfg)

    summary = {"vdw": vdw, "elec": elec}
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
