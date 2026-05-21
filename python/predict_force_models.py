"""Load trained vdW and electrostatic NN surrogates and predict forces from gap."""

from __future__ import annotations

import argparse
import json
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


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

class RadialForceMLP(nn.Module):
    """
    EO-flow radial scalar force model.

    Input:
        normalized center-center distance d_nm, shape (batch, 1)

    Output:
        normalized signed scalar force, shape (batch, 1)
    """
    def __init__(self, hidden_dim: int = 64, n_hidden: int = 3):
        super().__init__()

        layers = []
        in_dim = 1

        for _ in range(n_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def load_model(path: str, device: str) -> Tuple[nn.Module, Dict[str, object]]:
    """
    Load either:

    1. Old vdW/electrostatic log-force model:
        keys: train_config, state_dict, x_mu, x_sigma, y_mu, y_sigma, sign

    2. New EO-flow radial force model:
        keys: model_state_dict, hidden_dim, n_hidden, x_mean, x_std, y_mean, y_std
    """

    ckpt = torch.load(path, map_location=device)

    # ------------------------------------------------------------
    # Old vdW/electrostatic checkpoint
    # ------------------------------------------------------------
    if "train_config" in ckpt and "state_dict" in ckpt:
        hidden = int(ckpt["train_config"]["hidden"])
        model = MLP(hidden=hidden).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model, ckpt

    # ------------------------------------------------------------
    # New EO-flow checkpoint
    # ------------------------------------------------------------
    if "model_state_dict" in ckpt:
        hidden_dim = int(ckpt.get("hidden_dim", 64))
        n_hidden = int(ckpt.get("n_hidden", 3))

        model = RadialForceMLP(
            hidden_dim=hidden_dim,
            n_hidden=n_hidden,
        ).to(device)

        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, ckpt

    raise KeyError(
        f"Unknown checkpoint format for {path}. "
        f"Available keys: {list(ckpt.keys())}"
    )


def predict_force_pN(model: nn.Module, ckpt: Dict[str, object], gap_nm: np.ndarray, device: str) -> np.ndarray:
    gap_nm = np.asarray(gap_nm, dtype=np.float64)
    if np.any(gap_nm <= 0.0):
        raise ValueError("All gap values must be > 0.")

    x = np.log10(gap_nm)
    x_std = (x - ckpt["x_mu"]) / ckpt["x_sigma"]
    xt = torch.tensor(x_std[:, None], dtype=torch.float32, device=device)
    with torch.no_grad():
        y_std = model(xt).cpu().numpy().ravel()
    y_raw = y_std * ckpt["y_sigma"] + ckpt["y_mu"]
    mag = np.power(10.0, y_raw)
    sign = ckpt["sign"]
    if sign == "negative":
        return -mag
    if sign == "positive":
        return mag
    raise ValueError(f"Unknown sign convention: {sign}")

def predict_eo_flow_force_N(model: torch.nn.Module,
                            ckpt: dict,
                            d_nm: np.ndarray,
                            device: str) -> np.ndarray:
    """
    Predict mobility-equivalent EO pair force.

    Input:
        d_nm = center-center distance in nm

    Output:
        f_N = signed scalar radial force in Newtons

    Convention:
        positive f_N means attraction:
            F_i = +f_N * rhat_ij
            F_j = -f_N * rhat_ij

        where rhat_ij points from particle i to particle j.
    """
    d_nm = np.asarray(d_nm, dtype=np.float64).reshape(-1, 1)

    if np.any(d_nm <= 0.0):
        raise ValueError("All center-center distances must be > 0.")

    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float64)
    x_std  = np.asarray(ckpt["x_std"], dtype=np.float64)
    y_mean = np.asarray(ckpt["y_mean"], dtype=np.float64)
    y_std  = np.asarray(ckpt["y_std"], dtype=np.float64)

    x_norm = (d_nm - x_mean) / (x_std + 1e-30)

    xt = torch.tensor(x_norm, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        y_norm = model(xt).cpu().numpy()

    f_N = y_norm * (y_std + 1e-30) + y_mean

    return f_N.reshape(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vdw_model", type=str, required=True)
    parser.add_argument("--elec_model", type=str, required=True)
    parser.add_argument("--gaps_nm", type=float, nargs="+", required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vdw_model, vdw_ckpt = load_model(args.vdw_model, args.device)
    elec_model, elec_ckpt = load_model(args.elec_model, args.device)

    gaps_nm = np.asarray(args.gaps_nm, dtype=np.float64)
    f_vdw = predict_force_pN(vdw_model, vdw_ckpt, gaps_nm, args.device)
    f_elec = predict_force_pN(elec_model, elec_ckpt, gaps_nm, args.device)

    rows = []
    for g, fv, fe in zip(gaps_nm, f_vdw, f_elec):
        rows.append({
            "gap_nm": float(g),
            "F_vdw_pN": float(fv),
            "F_elec_pN": float(fe),
            "F_total_no_flow_pN": float(fv + fe),
        })
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
