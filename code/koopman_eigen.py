"""Koopman matrix A eigenvalue visualization (shared by inf and scen2)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_eigenvalues_on_unit_circle(
    A_np: np.ndarray,
    save_path: str | Path,
    title: str = "Eigenvalues of Koopman Matrix A",
) -> None:
    eigvals = np.linalg.eigvals(A_np)
    fig, ax = plt.subplots(figsize=(8, 8))
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=1, alpha=0.3, label="Unit circle")

    ax.scatter(
        eigvals.real,
        eigvals.imag,
        c=np.arange(len(eigvals)),
        cmap="coolwarm",
        s=100,
        edgecolors="black",
        linewidth=1.5,
        zorder=5,
    )

    for i, ev in enumerate(eigvals):
        ax.annotate(
            f"λ{i}",
            (ev.real, ev.imag),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_title(title)
    ax.axhline(0, color="black", alpha=0.3)
    ax.axvline(0, color="black", alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dual_eigenvalues(
    A0_np: np.ndarray,
    delta_A_np: np.ndarray,
    out_dir: str | Path,
    prefix: str = "eigenvalues",
) -> dict[str, Path]:
    """Plot A0 and A0+delta_A eigenvalues. Returns paths dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    A1_np = A0_np + delta_A_np
    path_a0 = out_dir / f"{prefix}_A0.png"
    path_a1 = out_dir / f"{prefix}_A1.png"
    plot_eigenvalues_on_unit_circle(A0_np, path_a0, title="Eigenvalues of A0")
    plot_eigenvalues_on_unit_circle(A1_np, path_a1, title="Eigenvalues of A0 + ΔA (u=1)")
    return {"A0": path_a0, "A1": path_a1}


def extract_A_from_state_dict(state_dict: dict) -> np.ndarray | None:
    for key in ("A", "A0", "module.A", "module.A0"):
        if key in state_dict:
            return state_dict[key].detach().cpu().numpy()
    for key, value in state_dict.items():
        if key.endswith(".A") or key.endswith(".A0"):
            return value.detach().cpu().numpy()
    return None


def extract_delta_A_from_state_dict(state_dict: dict) -> np.ndarray | None:
    for key in ("delta_A", "module.delta_A"):
        if key in state_dict:
            return state_dict[key].detach().cpu().numpy()
    for key, value in state_dict.items():
        if key.endswith(".delta_A"):
            return value.detach().cpu().numpy()
    return None


def plot_model_eigenvalues(model, out_dir: str | Path, prefix: str = "eigenvalues") -> dict[str, Path]:
    """Plot eigenvalues from a KoopmanRoutesFormer instance."""
    import torch

    out_dir = Path(out_dir)
    paths: dict[str, Path] = {}
    if getattr(model, "uses_dual_A", False):
        A0 = model.A0.detach().cpu().numpy()
        delta = model.delta_A.detach().cpu().numpy()
        paths = plot_dual_eigenvalues(A0, delta, out_dir, prefix=prefix)
    else:
        A = model.A.detach().cpu().numpy()
        p = out_dir / f"{prefix}_A.png"
        plot_eigenvalues_on_unit_circle(A, p, title="Eigenvalues of Koopman Matrix A")
        paths["A"] = p
    return paths
