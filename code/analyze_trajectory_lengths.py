"""
データの経路長統計を分析するスクリプト

Usage:
    python analyze_trajectory_lengths.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# Config
# =========================================================
DATA_PATH = '/home/mizutani/projects/RF/data/input_real_m5.npz'
OUTPUT_DIR = '/home/mizutani/projects/RF/0128analysis'
PAD_TOKEN = 38

# =========================================================
# Main
# =========================================================

def main():
    print("="*60)
    print("Trajectory Length Analysis")
    print("="*60)
    print(f"Loading data from: {DATA_PATH}")
    
    # データ読み込み
    data = np.load(DATA_PATH)
    routes = data['route_arr']  # [N, max_len]
    
    print(f"Total samples: {len(routes)}")
    print(f"Max sequence length (with padding): {routes.shape[1]}")
    
    # 各経路の有効長を計算（パディングを除く）
    valid_lengths = []
    for route in routes:
        valid_mask = (route != PAD_TOKEN)
        length = valid_mask.sum()
        valid_lengths.append(length)
    
    valid_lengths = np.array(valid_lengths)
    
    # 統計量を計算
    print("\n" + "="*60)
    print("Statistics")
    print("="*60)
    print(f"Min length:    {valid_lengths.min()}")
    print(f"Max length:    {valid_lengths.max()}")
    print(f"Mean length:   {valid_lengths.mean():.2f}")
    print(f"Median length: {np.median(valid_lengths):.2f}")
    print(f"Std deviation: {valid_lengths.std():.2f}")
    
    # パーセンタイル
    print("\nPercentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(valid_lengths, p)
        print(f"  {p:2d}th: {val:.1f}")
    
    # 長さごとのカウント
    print("\n" + "="*60)
    print("Length Distribution (Top 20)")
    print("="*60)
    unique_lengths, counts = np.unique(valid_lengths, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]  # 降順
    
    print("Length | Count | Percentage")
    print("-"*40)
    for i in sorted_indices[:20]:
        length = unique_lengths[i]
        count = counts[i]
        pct = count / len(valid_lengths) * 100
        print(f"{length:6d} | {count:5d} | {pct:6.2f}%")
    
    # K, L の推奨値を提案
    print("\n" + "="*60)
    print("Recommendations for K and L")
    print("="*60)
    
    # 25パーセンタイルを基準に K を決定
    p25 = np.percentile(valid_lengths, 25)
    suggested_K = int(p25 // 2)
    suggested_L = suggested_K
    
    print(f"Based on 25th percentile ({p25:.1f}):")
    print(f"  Suggested K (Prefix length):  {suggested_K}")
    print(f"  Suggested L (Rollout length): {suggested_L}")
    print(f"  Minimum required length: {suggested_K + suggested_L}")
    
    # K+L で使えるサンプル数を計算
    usable = (valid_lengths >= suggested_K + suggested_L).sum()
    usable_pct = usable / len(valid_lengths) * 100
    print(f"  Usable samples: {usable} / {len(valid_lengths)} ({usable_pct:.1f}%)")
    
    # ヒストグラム作成
    print("\n" + "="*60)
    print("Creating histogram...")
    print("="*60)
    
    # 出力ディレクトリ作成
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # プロット
    plt.figure(figsize=(14, 6))
    
    # サブプロット1: 全体のヒストグラム
    plt.subplot(1, 2, 1)
    plt.hist(valid_lengths, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(valid_lengths.mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {valid_lengths.mean():.1f}')
    plt.axvline(np.median(valid_lengths), color='green', linestyle='--', 
                linewidth=2, label=f'Median: {np.median(valid_lengths):.1f}')
    plt.axvline(suggested_K + suggested_L, color='blue', linestyle='--',
                linewidth=2, label=f'K+L: {suggested_K + suggested_L}')
    plt.xlabel('Trajectory Length (steps)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Trajectory Length Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # サブプロット2: 累積分布
    plt.subplot(1, 2, 2)
    sorted_lengths = np.sort(valid_lengths)
    cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths) * 100
    plt.plot(sorted_lengths, cumulative, linewidth=2)
    plt.axhline(50, color='green', linestyle='--', linewidth=1, label='50th percentile')
    plt.axhline(75, color='orange', linestyle='--', linewidth=1, label='75th percentile')
    plt.axvline(suggested_K + suggested_L, color='blue', linestyle='--',
                linewidth=2, label=f'K+L: {suggested_K + suggested_L}')
    plt.xlabel('Trajectory Length (steps)', fontsize=12)
    plt.ylabel('Cumulative Percentage (%)', fontsize=12)
    plt.title('Cumulative Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    output_path = output_dir / 'trajectory_length_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved histogram to: {output_path}")
    
    # 詳細な長さごとの分布をCSVで保存
    csv_path = output_dir / 'trajectory_length_counts.csv'
    import pandas as pd
    df = pd.DataFrame({
        'length': unique_lengths,
        'count': counts,
        'percentage': counts / len(valid_lengths) * 100
    })
    df = df.sort_values('count', ascending=False)
    df.to_csv(csv_path, index=False)
    print(f"Saved detailed counts to: {csv_path}")
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)

if __name__ == "__main__":
    main()