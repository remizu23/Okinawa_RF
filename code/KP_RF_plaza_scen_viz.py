import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob

# 設定: CSVがあるフォルダ (rollout_scenario.py の out_dir)
CSV_DIR = "/home/mizutani/projects/RF/runs/simulation_2_13" 
OUT_DIR = "/home/mizutani/projects/RF/runs/simulation_2_13/fig"
os.makedirs(OUT_DIR, exist_ok=True)

def plot_occupancy(csv_path):
    print(f"Plotting {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # ピボットテーブル作成 (行: Node, 列: Time)
    # 値は presence_prob
    pivot = df.pivot(index="base_node", columns="t", values="presence_prob")
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, cmap="viridis", vmin=0, vmax=None) # 必要ならvmaxを固定 (例: 0.5)
    plt.title(f"Occupancy Probability Map\n({os.path.basename(csv_path)})")
    plt.xlabel("Time Step (Future)")
    plt.ylabel("Node ID")
    plt.tight_layout()
    
    save_name = os.path.basename(csv_path).replace(".csv", ".png")
    plt.savefig(os.path.join(OUT_DIR, save_name), dpi=300)
    plt.close()

def plot_diff(csv_path):
    print(f"Plotting difference {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 値は presence_prob_diff
    pivot = df.pivot(index="base_node", columns="t", values="presence_prob_diff")
    
    # 差分なので 赤(プラス) - 白(0) - 青(マイナス) のカラーマップを使う
    # 最大絶対値で正規化して、0が中心に来るようにする
    max_val = max(abs(pivot.min().min()), abs(pivot.max().max()))
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, cmap="coolwarm", center=0, vmin=-max_val, vmax=max_val)
    plt.title(f"Impact Difference Map (Scenario A - Scenario B)\nRed = Increased by A, Blue = Decreased by A\n({os.path.basename(csv_path)})")
    plt.xlabel("Time Step (Future)")
    plt.ylabel("Node ID")
    plt.tight_layout()
    
    save_name = os.path.basename(csv_path).replace(".csv", ".png")
    plt.savefig(os.path.join(OUT_DIR, save_name), dpi=300)
    plt.close()

def main():
    # 1. Occupancy CSVを探してプロット
    occ_files = glob.glob(os.path.join(CSV_DIR, "occupancy_*.csv"))
    for f in occ_files:
        plot_occupancy(f)
        
    # 2. Diff CSVを探してプロット
    diff_files = glob.glob(os.path.join(CSV_DIR, "diff_*.csv"))
    for f in diff_files:
        plot_diff(f)
        
    print(f"All plots saved to {OUT_DIR}/")

if __name__ == "__main__":
    main()