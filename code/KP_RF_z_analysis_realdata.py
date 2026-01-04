import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import networkx as nx
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from datetime import datetime
import matplotlib.cm as cm

# KP_RF.py の読み込み
try:
    from KP_RF import KoopmanRoutesFormer
except ImportError:
    raise ImportError("KP_RF.py not found.")

# =========================================================
# 0. 設定 & パス
# =========================================================
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"/home/mizutani/projects/RF/runs/z_analysis_real_{run_id}"
os.makedirs(out_dir, exist_ok=True)

print(f"=== Real Data Analysis: {run_id} ===")
print(f"Results will be saved to: {out_dir}")

# ★学習済みモデルのパス（With Koopmanのもの推奨）
MODEL_PATH = "/home/mizutani/projects/RF/runs/20251217_202852/model_weights_20251217_202852.pth"

# ★実データのパス
DATA_PATH = "/home/mizutani/projects/RF/data/input_a.npz"

# クラスタ数（仮設定：4〜6くらいが妥当と推測）
NUM_CLUSTERS = 4 

# =========================================================
# 1. データの読み込みと前処理
# =========================================================
def load_real_data(data_path):
    print(f"Loading real data from {data_path}...")
    # input_a.npz の中身を確認し、ルート配列を取得
    # 想定: 'route_arr' キーなどにデータが入っている
    data = np.load(data_path, allow_pickle=True)
    
    # キー名の確認 (環境に合わせて調整してください)
    keys = list(data.keys())
    target_key = 'route_arr' if 'route_arr' in keys else keys[0]
    
    routes = data[target_key]
    print(f"Loaded {len(routes)} routes.")
    return routes

# =========================================================
# 2. モデルロード
# =========================================================
def load_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get('config', {
        'vocab_size': 20, 'token_emb_dim': 24, 'd_model': 24, 
        'nhead': 4, 'num_layers': 6, 'd_ff': 128, 'z_dim': 16, 'pad_token_id': 19
    })
    model = KoopmanRoutesFormer(
        vocab_size=config['vocab_size'], token_emb_dim=config['token_emb_dim'],
        d_model=config['d_model'], nhead=config['nhead'], num_layers=config['num_layers'],
        d_ff=config['d_ff'], z_dim=config['z_dim'], pad_token_id=config.get('pad_token_id', 19)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

# =========================================================
# 3. 潜在変数抽出 & クラスタリング
# =========================================================
def analyze_real_data(model, routes, device):
    latent_list = []
    valid_routes = []
    
    PAD_TOKEN = 19
    
    print("Extracting latent variables...")
    with torch.no_grad():
        for i, seq in enumerate(routes):
            # numpy array -> tensor
            # 実データの前処理（パディング除去や長さ制限）が必要な場合はここで行う
            # ここではそのまま突っ込むが、tensor変換時に型に注意
            if isinstance(seq, np.ndarray):
                seq_list = seq.tolist()
            else:
                seq_list = seq
            
            # 短すぎるデータはノイズになるのでスキップ
            valid_tokens = [x for x in seq_list if x != PAD_TOKEN]
            if len(valid_tokens) < 5: 
                continue

            # 入力用に整形 (Batch=1)
            # モデルの最大長制限などを考慮してカットしても良い
            inp = torch.tensor([seq_list], dtype=torch.long).to(device)
            
            # 推論
            out = model(inp)
            z = out[1] if isinstance(out, tuple) else None
            
            if z is not None:
                z_np = z[0].cpu().numpy() # [seq_len, z_dim]
                
                # 重心（平均）を計算して代表値とする
                # ※ パディング部分は除外して平均を取るのがベスト
                # z_np の長さは seq_list と同じはずなので、validなインデックスのみ使う
                valid_indices = [idx for idx, val in enumerate(seq_list) if val != PAD_TOKEN]
                if not valid_indices: continue
                
                z_mean = np.mean(z_np[valid_indices], axis=0)
                
                latent_list.append(z_mean)
                valid_routes.append(seq_list) # 後で可視化に使う

    latent_matrix = np.array(latent_list) # [N_samples, z_dim]
    print(f"Extracted features for {len(latent_matrix)} valid routes.")

    # --- K-Means Clustering ---
    print(f"Running K-Means (k={NUM_CLUSTERS})...")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
    labels = kmeans.fit_predict(latent_matrix)
    
    # --- PCA for Visualization ---
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(latent_matrix)
    
    # --- Plot 1: Latent Space with Clusters ---
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='tab10', alpha=0.6, s=20)
    plt.colorbar(scatter, label='Cluster ID')
    plt.title(f"Real Data Clustering (K-Means k={NUM_CLUSTERS}) in Latent Space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "real_data_clusters_pca.png"))
    plt.close()
    
    return valid_routes, labels

# =========================================================
# 4. クラスタごとの物理的意味の解釈 (Trajectory Visualization)
# =========================================================
def visualize_cluster_patterns(routes, labels):
    print("Visualizing physical trajectories per cluster...")
    
    # 各クラスタから数件ランダムに抽出してプロット
    # または「平均的なルート」を出せればベストだが、ノード列の平均は定義できないので
    # ランダムサンプリングで傾向を見る
    
    n_samples = 5 # 各クラスタにつき5人表示
    
    fig, axes = plt.subplots(NUM_CLUSTERS, 1, figsize=(10, 4 * NUM_CLUSTERS))
    if NUM_CLUSTERS == 1: axes = [axes]
    
    cmap = plt.get_cmap('tab10')
    
    for c_id in range(NUM_CLUSTERS):
        ax = axes[c_id]
        indices = [i for i, x in enumerate(labels) if x == c_id]
        
        if not indices:
            ax.set_title(f"Cluster {c_id}: No Data")
            continue
            
        # ランダムに選択
        selected_indices = np.random.choice(indices, min(len(indices), n_samples), replace=False)
        
        for idx in selected_indices:
            route = routes[idx]
            # パディング除去
            valid_route = [x for x in route if x != 19] # PAD=19と仮定
            
            # グラフにプロット (横軸:Step, 縦軸:Node ID)
            ax.plot(range(len(valid_route)), valid_route, marker='.', alpha=0.7, label=f"User {idx}")
            
        ax.set_title(f"Cluster {c_id} (N={len(indices)}) - Trajectory Examples")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Node ID")
        ax.set_ylim(-1, 20) # ノード数に応じて調整
        ax.grid(True, alpha=0.3)
        # ax.legend() # 凡例が多いと見づらいので消すか調整
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cluster_physical_patterns.png"))
    plt.close()
    
    print("Pattern visualization saved.")

# =========================================================
# 実行
# =========================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. データとモデルのロード
    routes = load_real_data(DATA_PATH)
    model = load_model(MODEL_PATH, device)
    
    # 2. 分析実行
    valid_routes, labels = analyze_real_data(model, routes, device)
    
    # 3. 結果の解釈
    visualize_cluster_patterns(valid_routes, labels)
    
    print("Done.")