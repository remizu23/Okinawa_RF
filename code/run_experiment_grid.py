import argparse
import os
import subprocess
from collections import defaultdict
from glob import glob
from datetime import datetime

import pandas as pd


def run_cmd(cmd):
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def latest_pth(train_out_dir):
    cands = sorted(glob(os.path.join(train_out_dir, "*.pth")), key=os.path.getmtime)
    if not cands:
        raise FileNotFoundError(f"No .pth found in {train_out_dir}")
    return cands[-1]


def summarize_eval(eval_out_dir, prefix_len, meta):
    csv_path = os.path.join(eval_out_dir, f"metrics_prefix{prefix_len}.csv")
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
    return row


def main():
    # 引数定義（実験条件）
    p = argparse.ArgumentParser(description="Run grid experiments for DKP_RF")
    p.add_argument("--mode", choices=["prefix", "zdim", "both"], default="both")
    p.add_argument("--model-types", type=str, default="transformer,mlp_flat")
    # p.add_argument("--prefix-lengths", type=str, default="2,3,4,5")
    p.add_argument("--prefix-lengths", type=str, default="5")
    # p.add_argument("--z-dims", type=str, default="8,16,32,64")
    p.add_argument("--z-dims", type=str, default="128")
    p.add_argument("--fixed-prefix-for-zdim", type=int, default=5)  # prefix長を変える実験（model_x_prefix）で使う固定 z_dim
    p.add_argument("--fixed-zdim-for-prefix", type=int, default=16)  # z_dimを変える実験（model_x_zdim）で使う固定 prefix_len
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--eval-data", choices=["TRAIN", "VAL", "TEST"], default="TEST")
    p.add_argument("--ablation-model-path", type=str, default = "/home/mizutani/projects/RF/runs/20260127_014847/ablation_weights_20260127_014847.pth")
    p.add_argument("--output-root", type=str, default="/home/mizutani/projects/RF/runs/2604_grid_experiments")
    p.add_argument("--session-name", type=str, default=None, help="optional suffix for this run directory")
    p.add_argument("--train-script", type=str, default="/home/mizutani/projects/RF/code/DKP_RF_train.py")
    p.add_argument("--inf-script", type=str, default="/home/mizutani/projects/RF/code/DKP_RF_inf.py")
    p.add_argument("--seed", type=int, default=42, help="DKP_RF_train.py に渡す乱数シード（再現性用．デフォルトは42）")
    p.add_argument("--deterministic-cuda", action="store_true", help="学習時に CUDNN deterministic を有効化（遅くなるが揺らぎ低減）")
    p.add_argument(
        "--dedupe",
        action="store_true", # 指定しなければFalseになる．
        default=True, 
        help="(model_type, prefix_len, z_dim) が重複する実験は学習1回のみ。summary には grid_type ごとに同一指標の行を複製",
    )
    args = p.parse_args()

    # CLI では "a,b,c" なので、ここで split + 型変換してリストにする．
    model_types = [x.strip() for x in args.model_types.split(",") if x.strip()]
    prefix_lengths = [int(x.strip()) for x in args.prefix_lengths.split(",") if x.strip()]
    z_dims = [int(x.strip()) for x in args.z_dims.split(",") if x.strip()]

    # 出力ディレクトリを作成．
    os.makedirs(args.output_root, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir_name = f"grid_{ts}" if not args.session_name else f"grid_{ts}_{args.session_name}"
    session_root = os.path.join(args.output_root, session_dir_name)
    os.makedirs(session_root, exist_ok=False)  # 既に存在する場合はエラー．
    print(f"[INFO] session_root: {session_root}")
    summary_rows = []  # 結果を格納するリスト．

    # 実験一覧の作成（prefix長とz次元の組み合わせ）．dictionaryのリストを作成．
    # 辞書のkeyは (model_type, prefix_len, z_dim) で，valueは そのグリッドのgrid_typeをリストにしたもの．
    experiments = []
    if args.mode in ("prefix", "both"):
        for model_type in model_types:
            for pfx in prefix_lengths:
                experiments.append({
                    "grid_type": "model_x_prefix", # xは掛け算の意味．model_typeとprefix_lenの組み合わせ．
                    "model_type": model_type,
                    "prefix_len": pfx,
                    "z_dim": args.fixed_zdim_for_prefix,
                })
    if args.mode in ("zdim", "both"):
        for model_type in model_types:
            for z in z_dims:
                experiments.append({
                    "grid_type": "model_x_zdim", # xは掛け算の意味．model_typeとz_dimの組み合わせ．
                    "model_type": model_type,
                    "prefix_len": args.fixed_prefix_for_zdim,
                    "z_dim": z,
                })

    if args.dedupe:  # dedupe引数がオンの場合：重複する実験を1回のみにする．
        groups = defaultdict(list)
        # 上で作った experiments を、キー (model_type, prefix_len, z_dim) でグルーピング
        # grid_type(掛け合わせパターン) が違っても、学習条件そのものが同じなら1回にまとめる．
        for exp in experiments:
            key = (exp["model_type"], exp["prefix_len"], exp["z_dim"])
            groups[key].append(exp["grid_type"])
        experiments = [
            {
                "model_type": k[0],
                "prefix_len": k[1],
                "z_dim": k[2],
                "grid_types": v,
            }
            for k, v in groups.items()
        ]

    # 実験の実行
    for i, exp in enumerate(experiments, start=1): # expはdictionary．experimentsの各要素（1条件）を順番に処理．iは001,002などに使う
        grid_types = exp["grid_types"] if args.dedupe else [exp["grid_type"]]
        # dedupe ON: 事前に統合済み exp は grid_types（配列）を持つ。
        # 例: ["model_x_prefix", "model_x_zdim"] 下記ではその最初の要素だけをフォルダ名のタグとして使う．
        gt_tag = grid_types[0]
        exp_name = f"{i:03d}_{gt_tag}_{exp['model_type']}_p{exp['prefix_len']}_z{exp['z_dim']}"
        exp_root = os.path.join(session_root, exp_name)
        train_out = os.path.join(exp_root, "train")
        eval_out = os.path.join(exp_root, "eval")
        os.makedirs(train_out, exist_ok=True)
        os.makedirs(eval_out, exist_ok=True)

        # train
        train_cmd = [
            "python", args.train_script,
            "--encoder-type", exp["model_type"],
            "--fixed-prefix-length", str(exp["prefix_len"]),
            "--z-dim", str(exp["z_dim"]),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--learning-rate", str(args.learning_rate),
            "--out-dir", train_out,
        ]
        if args.seed is not None:
            train_cmd.extend(["--seed", str(args.seed)])
        if args.deterministic_cuda:
            train_cmd.append("--deterministic-cuda")
        run_cmd(train_cmd)
        model_path = latest_pth(train_out)

        # eval
        inf_cmd = [
            "python", args.inf_script,
            "--model-koopman-path", model_path,
            "--model-ablation-path", args.ablation_model_path,
            "--output-dir", eval_out,
            "--eval-data", args.eval_data,
            "--prefix-lengths", str(exp["prefix_len"]),
        ]
        run_cmd(inf_cmd)

        # 結果の要約（dedupe 時は grid_type ごとに同一指標の行を出す）
        for grid_type in grid_types:
            summary_rows.append(
                summarize_eval(
                    eval_out_dir=eval_out,
                    prefix_len=exp["prefix_len"],
                    meta={
                        "experiment": exp_name,
                        "grid_type": grid_type,
                        "model_type": exp["model_type"],
                        "prefix_len": exp["prefix_len"],
                        "z_dim": exp["z_dim"],
                        "model_path": model_path,
                        "eval_out_dir": eval_out,
                    },
                )
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(session_root, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary: {summary_csv}")


if __name__ == "__main__":
    main()

