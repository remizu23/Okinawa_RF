import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, Subset

# 自作モジュール (データ読み込みのため必要)
from network import Network, expand_adjacency_matrix
from tokenization import Tokenization

# =========================================================
# 設定 (さっきと同じパスを指定してください)
# =========================================================
DATA_PATH = '/home/mizutani/projects/RF/data/input_e.npz'
ADJ_PATH = '/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt'

TARGET_DIR = '/home/mizutani/projects/RF/runs/SHAP_20260109_202418'

print(f"Target Directory: {TARGET_DIR}")

# =========================================================
# データのロード (SHAP値 & 元データ)
# =========================================================
# 1. SHAP値のロード
shap_data = np.load(os.path.join(TARGET_DIR, "shap_values.npz"))
shap_token = shap_data['token'] # [50, T]
shap_stay  = shap_data['stay']
shap_agent = shap_data['agent']

# 2. 元データのロード (X軸のラベル用)
trip_arrz = np.load(DATA_PATH)
trip_arr = trip_arrz['route_arr']
route_pt = torch.from_numpy(trip_arr).long()

# データセット構築
adj_matrix = torch.load(ADJ_PATH, weights_only=True)
expanded_adj = expand_adjacency_matrix(adj_matrix)
dummy_feature = torch.zeros((len(adj_matrix)*2, 1))
network = Network(expanded_adj, dummy_feature)
tokenizer = Tokenization(network)

# =========================================================
# 可視化関数
# =========================================================
def plot_individual_shap(sample_idx):
    # 元のルートを取得
    raw_route = route_pt[sample_idx] # [T]
    
    # トークン化してIDと滞在カウントを取得
    input_tokens = tokenizer.tokenization(raw_route.unsqueeze(0), mode="simple").long()[0]
    stay_counts = tokenizer.calculate_stay_counts(input_tokens.unsqueeze(0))[0]
    
    # このサンプルのSHAP値を取得
    s_token = shap_token[sample_idx]
    s_stay  = shap_stay[sample_idx]
    s_agent = shap_agent[sample_idx]
    
    # 系列長を合わせる（SHAP計算時にカットされている可能性があるため）
    seq_len = min(len(input_tokens), len(s_token))
    
    input_tokens = input_tokens[:seq_len]
    stay_counts = stay_counts[:seq_len]
    s_token = s_token[:seq_len]
    s_stay = s_stay[:seq_len]
    s_agent = s_agent[:seq_len]

    # --- プロット作成 ---
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    x = np.arange(seq_len)
    
    # SHAP値を積層棒グラフで表示 (絶対値ではなく、プラスマイナスを見るため生の値を表示)
    # ※ 絶対値で見たい場合は np.abs() を付けてください
    ax1.plot(x, s_token, label='Node ID SHAP', color='blue', marker='o', linestyle='-', alpha=0.7)
    ax1.plot(x, s_stay, label='Stay Count SHAP', color='orange', marker='s', linestyle='-', alpha=0.7)
    ax1.plot(x, s_agent, label='Agent ID SHAP', color='green', marker='^', linestyle=':', alpha=0.5)
    
    ax1.set_xlabel("Actual Node ID")
    ax1.set_ylabel("SHAP Value")
    ax1.set_title(f"Individual SHAP Values for Sample #{sample_idx}")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # ゼロライン
    ax1.axhline(0, color='black', linewidth=0.5)

    # --- X軸のラベルを工夫する ---
    # ステップ数だけでなく、「実際にどのノードにいたか」を表示
    # ラベル形式: "Step\n(Node:Count)"
    
    x_labels = []
    for i in range(seq_len):
        node_id = input_tokens[i].item()
        count = stay_counts[i].item()
        
        # 特殊トークンの表示
        if node_id >= network.N: # パディングなど
            lbl = f"(PAD)"
        else:
            lbl = f"{node_id}:C{count}"
        x_labels.append(lbl)

    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=60, fontsize=8)
    
    # レイアウト調整
    plt.tight_layout()
    save_path = os.path.join(TARGET_DIR, f"individual_shap_sample_{sample_idx}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

# =========================================================
# 実行 (最初の5人分)
# =========================================================
print("Plotting individual samples...")
for i in range(5):
    plot_individual_shap(i)

print("Done.")