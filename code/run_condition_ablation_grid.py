"""Grid runner for context ablation experiments (M0/M1/M2 + random-u control).
7/20 グリッド一括（12 run + VAL/TEST）
実行例
python run_condition_ablation_grid.py \
  --encoder-type transformer --prefix-length 5 --z-dim 32 \
  --epochs 150 --eval-splits VAL,TEST

"""

import argparse
import json
import os
import subprocess
from datetime import datetime
from glob import glob

import pandas as pd

CHECKPOINT_MANIFEST = "checkpoint.json"

CONDITION_FEATURES = ["event", "timezone", "stay"]
CONDITION_MODES = ["M0", "M1", "M2"]


def run_cmd(cmd):
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def resolve_model_path(train_out_dir):
    manifest_path = os.path.join(train_out_dir, CHECKPOINT_MANIFEST)
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        model_path = data.get("model_path")
        if model_path and os.path.exists(model_path):
            return model_path
        raise FileNotFoundError(
            f"checkpoint.json に記載の model_path が存在しません: {model_path}"
        )
    cands = sorted(glob(os.path.join(train_out_dir, "*.pth")), key=os.path.getmtime)
    if not cands:
        raise FileNotFoundError(f"No .pth found in {train_out_dir}")
    return cands[-1]


def summarize_eval(eval_out_dir, prefix_len, meta, suffix=""):
    csv_path = os.path.join(eval_out_dir, f"metrics_prefix{prefix_len}{suffix}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Eval csv not found: {csv_path}")
    df = pd.read_csv(csv_path)
    row = {
        **meta,
        "num_samples": int(len(df)),
        "k_acc_mean": float(df["k_acc"].mean()),
        "k_prob_mean": float(df["k_prob"].mean()),
        "k_ed_raw_mean": float(df["k_ed"].mean()),
        "k_ed_norm_mean": float(df["k_ed_norm"].mean()),
        "k_ged_norm_mean": float(df["k_ged_norm"].mean()),
        "k_dtw_norm_mean": float(df["k_dtw_norm"].mean()),
        "k_gdtw_norm_mean": float(df["k_gdtw_norm"].mean()),
    }
    if "condition_u" in df.columns:
        for u_val, label in [(0, "u0"), (1, "u1")]:
            sub = df[df["condition_u"] == u_val]
            if len(sub) > 0:
                row[f"k_acc_mean_{label}"] = float(sub["k_acc"].mean())
                row[f"k_prob_mean_{label}"] = float(sub["k_prob"].mean())
                row[f"k_ed_norm_mean_{label}"] = float(sub["k_ed_norm"].mean())
                row[f"num_samples_{label}"] = int(len(sub))
            else:
                row[f"k_acc_mean_{label}"] = float("nan")
                row[f"k_prob_mean_{label}"] = float("nan")
                row[f"k_ed_norm_mean_{label}"] = float("nan")
                row[f"num_samples_{label}"] = 0
    return row


def build_experiments(features, modes, include_random_control):
    exps = []
    for feat in features:
        for mode in modes:
            exps.append({
                "condition_feature": feat,
                "condition_mode": mode,
                "random_u_labels": False,
            })
        if include_random_control:
            exps.append({
                "condition_feature": feat,
                "condition_mode": "M2",
                "random_u_labels": True,
            })
    return exps


def main():
    p = argparse.ArgumentParser(description="Run context ablation grid (M0/M1/M2)")
    p.add_argument("--condition-features", type=str, default="event,timezone,stay")
    p.add_argument("--condition-modes", type=str, default="M0,M1,M2")
    p.add_argument("--include-random-control", action="store_true", default=True)
    p.add_argument("--no-random-control", action="store_true")
    p.add_argument("--encoder-type", type=str, default="transformer", choices=["transformer", "lstm", "mlp_flat"])
    p.add_argument("--prefix-length", type=int, default=5)
    p.add_argument("--z-dim", type=int, default=16)
    p.add_argument("--stay-u-threshold", type=int, default=3)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--eval-splits", type=str, default="VAL,TEST")
    p.add_argument("--output-root", type=str, default="/home/mizutani/projects/RF/runs/2607_condition_ablation")
    p.add_argument("--session-name", type=str, default=None)
    p.add_argument("--train-script", type=str, default="/home/mizutani/projects/RF/code/DKP_RF_train.py")
    p.add_argument("--inf-script", type=str, default="/home/mizutani/projects/RF/code/DKP_RF_inf.py")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic-cuda", action="store_true")
    p.add_argument("--ablation-model-path", type=str,
                   default="/home/mizutani/projects/RF/runs/20260127_014847/ablation_weights_20260127_014847.pth")
    args = p.parse_args()

    features = [x.strip() for x in args.condition_features.split(",") if x.strip()]
    modes = [x.strip().upper() for x in args.condition_modes.split(",") if x.strip()]
    eval_splits = [x.strip().upper() for x in args.eval_splits.split(",") if x.strip()]
    include_random = args.include_random_control and not args.no_random_control

    os.makedirs(args.output_root, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = f"cond_ablation_{ts}" if not args.session_name else f"cond_ablation_{ts}_{args.session_name}"
    session_root = os.path.join(args.output_root, session_name)
    os.makedirs(session_root, exist_ok=False)
    print(f"[INFO] session_root: {session_root}")

    experiments = build_experiments(features, modes, include_random)
    summary_rows = []

    for i, exp in enumerate(experiments, start=1):
        feat = exp["condition_feature"]
        mode = exp["condition_mode"]
        random_u = exp["random_u_labels"]
        tag = f"{feat}_{mode}"
        if random_u:
            tag += "_random_u"
        exp_name = f"{i:03d}_{tag}_p{args.prefix_length}_z{args.z_dim}"
        exp_root = os.path.join(session_root, exp_name)
        train_out = os.path.join(exp_root, "train")
        os.makedirs(train_out, exist_ok=True)

        train_cmd = [
            "python", args.train_script,
            "--encoder-type", args.encoder_type,
            "--fixed-prefix-length", str(args.prefix_length),
            "--z-dim", str(args.z_dim),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--learning-rate", str(args.learning_rate),
            "--out-dir", train_out,
            "--condition-feature", feat,
            "--condition-mode", mode,
            "--stay-u-threshold", str(args.stay_u_threshold),
        ]
        if random_u:
            train_cmd.append("--random-u-labels")
        if args.seed is not None:
            train_cmd.extend(["--seed", str(args.seed)])
        if args.deterministic_cuda:
            train_cmd.append("--deterministic-cuda")

        run_cmd(train_cmd)
        model_path = resolve_model_path(train_out)

        for split in eval_splits:
            eval_out = os.path.join(exp_root, f"eval_{split.lower()}")
            os.makedirs(eval_out, exist_ok=True)

            inf_cmd = [
                "python", args.inf_script,
                "--model-koopman-path", model_path,
                "--model-ablation-path", args.ablation_model_path,
                "--output-dir", eval_out,
                "--eval-data", split,
                "--prefix-lengths", str(args.prefix_length),
            ]
            if mode == "M2" and not random_u:
                inf_cmd.append("--eval-random-u")
            run_cmd(inf_cmd)

            eigen_glob = glob(os.path.join(eval_out, "eigenvalues*.png"))
            eigen_paths = ",".join(sorted(eigen_glob))

            meta = {
                "experiment": exp_name,
                "eval_split": split,
                "condition_feature": feat,
                "condition_mode": mode,
                "random_u_labels": random_u,
                "encoder_type": args.encoder_type,
                "prefix_len": args.prefix_length,
                "z_dim": args.z_dim,
                "stay_u_threshold": args.stay_u_threshold,
                "model_path": model_path,
                "eval_out_dir": eval_out,
                "eigenvalue_plots": eigen_paths,
            }
            summary_rows.append(summarize_eval(eval_out, args.prefix_length, meta))

            random_csv = os.path.join(eval_out, f"metrics_prefix{args.prefix_length}_random_u.csv")
            if os.path.exists(random_csv):
                summary_rows.append(summarize_eval(
                    eval_out, args.prefix_length,
                    {**meta, "u_eval_mode": "random_u"},
                    suffix="_random_u",
                ))

        manifest = {
            "experiment": exp_name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "session_root": session_root,
            "train_out_dir": train_out,
            "model_path": model_path,
            "hyperparams": {
                "condition_feature": feat,
                "condition_mode": mode,
                "random_u_labels": random_u,
                "encoder_type": args.encoder_type,
                "prefix_len": args.prefix_length,
                "z_dim": args.z_dim,
                "stay_u_threshold": args.stay_u_threshold,
                "epochs": args.epochs,
                "eval_splits": eval_splits,
            },
            "train_cmd": train_cmd,
        }
        with open(os.path.join(exp_root, "experiment.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(session_root, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary: {summary_csv}")


if __name__ == "__main__":
    main()
