import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def find_latest_checkpoint(train_dir: Path) -> Path | None:
    candidates = sorted(train_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    return candidates[-1]


def extract_A_matrix(checkpoint: dict) -> torch.Tensor:
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError("checkpoint に 'model_state_dict' がありません。")

    direct_candidates = ("A", "module.A")
    for key in direct_candidates:
        if key in state_dict:
            return state_dict[key]

    # 念のため、末尾が ".A" のキーも許容
    for key, value in state_dict.items():
        if key.endswith(".A"):
            return value

    raise KeyError("state_dict から Koopman 行列 A を見つけられませんでした。")


def plot_eigenvalues_on_unit_circle(eigvals: np.ndarray, out_path: Path, title: str) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 512)
    unit_x = np.cos(theta)
    unit_y = np.sin(theta)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(unit_x, unit_y, linestyle="--", linewidth=1.2, label="Unit circle")
    ax.scatter(
        eigvals.real,
        eigvals.imag,
        s=28,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.4,
        label="Eigenvalues of A",
    )
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.axvline(0.0, color="gray", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    # 見切れ防止のため、固有値の広がりに応じて表示範囲を設定
    all_x = np.concatenate([eigvals.real, unit_x])
    all_y = np.concatenate([eigvals.imag, unit_y])
    x_pad = max(0.2, 0.1 * (all_x.max() - all_x.min()))
    y_pad = max(0.2, 0.1 * (all_y.max() - all_y.min()))
    ax.set_xlim(all_x.min() - x_pad, all_x.max() + x_pad)
    ax.set_ylim(all_y.min() - y_pad, all_y.max() + y_pad)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def iter_experiment_dirs(session_root: Path):
    for p in sorted(session_root.iterdir()):
        if p.is_dir():
            yield p


def main() -> None:
    parser = argparse.ArgumentParser(
        description="各実験ディレクトリの pth から Koopman 行列 A の固有値プロットを保存する"
    )
    parser.add_argument(
        "--session-root",
        type=Path,
        default=Path("/home/mizutani/projects/RF/runs/2604_grid_experiments/grid_20260409_193345"),
        help="run_experiment_grid.py の1セッション出力ディレクトリ",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="koopman_A_eigenvalues.png",
        help="各実験ディレクトリに保存する画像ファイル名",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="失敗した実験ディレクトリがある場合に即時例外で停止する",
    )
    args = parser.parse_args()

    session_root: Path = args.session_root
    if not session_root.exists():
        raise FileNotFoundError(f"session root が存在しません: {session_root}")

    success = 0
    skipped = 0

    for exp_dir in iter_experiment_dirs(session_root):
        train_dir = exp_dir / "train"
        if not train_dir.exists():
            print(f"[SKIP] train ディレクトリなし: {exp_dir}")
            skipped += 1
            continue

        ckpt_path = find_latest_checkpoint(train_dir)
        if ckpt_path is None:
            msg = f"[SKIP] pth が見つかりません: {train_dir}"
            if args.strict:
                raise FileNotFoundError(msg)
            print(msg)
            skipped += 1
            continue

        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            A = extract_A_matrix(checkpoint).detach().cpu().numpy()
            eigvals = np.linalg.eigvals(A)
            out_path = exp_dir / args.output_filename
            plot_eigenvalues_on_unit_circle(
                eigvals=eigvals,
                out_path=out_path,
                title=f"{exp_dir.name}\nA eigenvalues",
            )
            print(f"[OK] {exp_dir.name}: {out_path.name}")
            success += 1
        except Exception as e:  # noqa: BLE001
            msg = f"[FAIL] {exp_dir.name}: {e}"
            if args.strict:
                raise RuntimeError(msg) from e
            print(msg)
            skipped += 1

    print(f"\n完了: success={success}, skipped_or_failed={skipped}")


if __name__ == "__main__":
    main()
