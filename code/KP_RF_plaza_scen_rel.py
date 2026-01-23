import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

# ★設定: simulation_[2, 4] などのフォルダパス
CSV_DIR = "/home/mizutani/projects/RF/runs/simulation_2_13_m4_2hop" 
OUT_DIR = os.path.join(CSV_DIR, "fig_relative")
os.makedirs(OUT_DIR, exist_ok=True)

def plot_relative_impact(occ_file_A, occ_file_B):
    print(f"Comparing {os.path.basename(occ_file_A)} vs {os.path.basename(occ_file_B)}...")
    
    df_A = pd.read_csv(occ_file_A)
    df_B = pd.read_csv(occ_file_B)
    
    # ピボットテーブル (Node x Time)
    p_A = df_A.pivot(index="base_node", columns="t", values="presence_prob")
    p_B = df_B.pivot(index="base_node", columns="t", values="presence_prob")
    
    # 相対変化率の計算: (A - B) / (B + epsilon)
    # Bを基準とした時のAの増減率
    # 0除算を防ぐため微小値を足す
    relative_change = (p_A - p_B) / (p_B + 1e-4) * 100 # %単位
    
    # --- ヒートマップ ---
    plt.figure(figsize=(14, 8))
    # -50% から +50% の範囲で色付け
    sns.heatmap(relative_change, cmap="coolwarm", center=0, vmin=-50, vmax=50, cbar_kws={'label': '% Change'})
    
    plt.title(f"Relative Impact Map (% Change)\n(How much traffic increased compared to baseline?)")
    plt.xlabel("Time Step")
    plt.ylabel("Node ID")
    plt.tight_layout()
    
    save_name = "relative_change_map.png"
    plt.savefig(os.path.join(OUT_DIR, save_name), dpi=300)
    plt.close()
    
    # --- 累積インパクト (棒グラフ) ---
    # 時間方向に合計して、「トータルの滞在量が何%変わったか」
    total_A = p_A.sum(axis=1)
    total_B = p_B.sum(axis=1)
    diff_total = total_A - total_B
    
    # 上位の変化があったノードを表示
    plt.figure(figsize=(10, 6))
    diff_total.plot(kind='bar', color=['red' if x > 0 else 'blue' for x in diff_total])
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title("Total Change in Stay Volume (Sum over Time)\nPositive = Plaza A attracts more")
    plt.xlabel("Node ID")
    plt.ylabel("Total Probability Mass Change")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "total_impact_bar.png"), dpi=300)
    plt.close()
    
    print(f"Saved relative analysis to {OUT_DIR}")

def main():
    # ファイルを探す
    files = os.listdir(CSV_DIR)
    occ_files = sorted([f for f in files if f.startswith("occupancy_") and f.endswith(".csv")])
    
    if len(occ_files) < 2:
        print("Not enough occupancy files to compare.")
        return
        
    # 2つを選んで比較 (例: plaza2 vs plaza4)
    file_A = os.path.join(CSV_DIR, occ_files[0]) # plaza 2
    file_B = os.path.join(CSV_DIR, occ_files[1]) # plaza 4 (baseline)
    
    plot_relative_impact(file_A, file_B)

if __name__ == "__main__":
    main()