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

print(f"=== Real Data Comparison Analysis: {run_id} ===")
print(f"Results will be saved to: {out_dir}")

# ★★★ 要修正: モデルパス ★★★
MODEL_PATH_WITH = "/home/mizutani/projects/RF/runs/20251217_202852/model_weights_20251217_202852.pth"
MODEL_PATH_WITHOUT = "/home/mizutani/projects/RF/runs/20251217_184803/model_weights_20251217_184803.pth"

# ★実データのパス
DATA_PATH = "/home/mizutani/projects/RF/data/input_a.npz"

# クラスタ数
NUM_CLUSTERS = 4 

# =========================================================
# 1. データの読み込み
# =========================================================
def load_real_data(data_path):
    print(f"Loading real data from {data_path}...")
    data = np.load(data_path, allow_pickle=True)
    keys = list(data.keys())
    # キー名探索 ('route_arr' など)
    target_key = 'route_arr' if 'route_arr' in keys else keys[0]
    routes = data[target_key]
    print(f"Loaded {len(routes)} routes.")
    return routes

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
# 2. 処理ロジック (潜在変数抽出 -> K-Means -> PCA)
# =========================================================
def process_model(model, routes, device, model_name="Model"):
    print(f"Processing {model_name}...")
    latent_list = []
    valid_indices = [] # 有効だったデータのインデックス
    
    PAD_TOKEN = 19
    
    with torch.no_grad():
        for i, seq in enumerate(routes):
            if isinstance(seq, np.ndarray): seq_list = seq.tolist()
            else: seq_list = seq
            
            # 短すぎるデータを除外
            valid_tokens = [x for x in seq_list if x != PAD_TOKEN]
            if len(valid_tokens) < 5: continue

            inp = torch.tensor([seq_list], dtype=torch.long).to(device)
            out = model(inp)
            z = out[1] if isinstance(out, tuple) else None
            
            if z is not None:
                z_np = z[0].cpu().numpy()
                # パディング除外して平均
                valid_pos = [idx for idx, val in enumerate(seq_list) if val != PAD_TOKEN]
                if not valid_pos: continue
                
                z_mean = np.mean(z_np[valid_pos], axis=0)
                latent_list.append(z_mean)
                valid_indices.append(i)

    if not latent_list:
        print(f"Warning: No valid latent variables found for {model_name}")
        return None

    latent_matrix = np.array(latent_list) # [N, z_dim]
    
    # 1. K-Means
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
    labels = kmeans.fit_predict(latent_matrix)
    
    # 2. PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(latent_matrix)
    
    return {
        'z_pca': pca_result,
        'labels': labels,
        'valid_indices': valid_indices,
        'latent_matrix': latent_matrix
    }

# =========================================================
# 3. 比較可視化ロジック
# =========================================================
def get_common_limits(res1, res2):
    """2つのPCA結果から共通の描画範囲を決定"""
    d1 = res1['z_pca']
    d2 = res2['z_pca']
    
    x_min = min(d1[:,0].min(), d2[:,0].min())
    x_max = max(d1[:,0].max(), d2[:,0].max())
    y_min = min(d1[:,1].min(), d2[:,1].min())
    y_max = max(d1[:,1].max(), d2[:,1].max())
    
    # マージン追加
    mx = (x_max - x_min) * 0.1
    my = (y_max - y_min) * 0.1
    
    return (x_min - mx, x_max + mx), (y_min - my, y_max + my)

def visualize_clustering_comparison(res_with, res_without):
    print("Generating clustering comparison plot...")
    
    xlim, ylim = get_common_limits(res_with, res_without)
    print(f"Unified Scale -- X: {xlim}, Y: {ylim}")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    
    # With Koopman
    ax = axes[0]
    scatter = ax.scatter(res_with['z_pca'][:,0], res_with['z_pca'][:,1], 
                         c=res_with['labels'], cmap='tab10', alpha=0.6, s=20)
    ax.set_title(f"With Koopman (K-Means k={NUM_CLUSTERS})")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)
    
    # Without Koopman
    ax = axes[1]
    ax.scatter(res_without['z_pca'][:,0], res_without['z_pca'][:,1], 
               c=res_without['labels'], cmap='tab10', alpha=0.6, s=20)
    ax.set_title(f"Without Koopman (K-Means k={NUM_CLUSTERS})")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)
    
    # カラーバー（共通の意味はないが、クラスタIDの目安として）
    fig.colorbar(scatter, ax=axes.ravel().tolist(), label='Cluster ID')
    
    plt.suptitle(f"Real Data Clustering Comparison (Unified Scale)", fontsize=16)
    plt.savefig(os.path.join(out_dir, "real_data_clustering_comparison.png"))
    plt.close()

def visualize_cluster_trajectories(routes, res, model_name):
    """クラスタごとの物理軌跡を可視化（モデルごと）"""
    print(f"Visualizing trajectories for {model_name}...")
    
    labels = res['labels']
    indices = res['valid_indices']
    
    # 各クラスタから数件抽出
    n_samples = 5
    
    fig, axes = plt.subplots(NUM_CLUSTERS, 1, figsize=(10, 3.5 * NUM_CLUSTERS))
    if NUM_CLUSTERS == 1: axes = [axes]
    
    for c_id in range(NUM_CLUSTERS):
        ax = axes[c_id]
        # このクラスタに属するデータの元のインデックス
        cluster_indices = [indices[i] for i, lbl in enumerate(labels) if lbl == c_id]
        
        if not cluster_indices:
            ax.set_title(f"Cluster {c_id}: No Data")
            continue
            
        # ランダムサンプリング
        selected_indices = np.random.choice(cluster_indices, min(len(cluster_indices), n_samples), replace=False)
        
        for idx in selected_indices:
            raw_route = routes[idx]
            # numpy -> list
            if isinstance(raw_route, np.ndarray): r_list = raw_route.tolist()
            else: r_list = raw_route
            
            # PAD除去
            valid_route = [x for x in r_list if x != 19]
            ax.plot(range(len(valid_route)), valid_route, marker='.', alpha=0.7)
            
        ax.set_title(f"Cluster {c_id} (N={len(cluster_indices)})")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Node ID")
        ax.set_ylim(-1, 20)
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.suptitle(f"{model_name} - Cluster Trajectories", y=1.02)
    plt.savefig(os.path.join(out_dir, f"trajectories_v2_{model_name.replace(' ', '_')}.png"))
    plt.close()

# =========================================================
# 実行
# =========================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. データロード
    routes = load_real_data(DATA_PATH)
    
    # 2. モデル処理
    model_with = load_model(MODEL_PATH_WITH, device)
    res_with = process_model(model_with, routes, device, "With Koopman")
    
    model_without = load_model(MODEL_PATH_WITHOUT, device)
    res_without = process_model(model_without, routes, device, "Without Koopman")
    
    # 3. 比較可視化 (散布図)
    if res_with and res_without:
        visualize_clustering_comparison(res_with, res_without)
        
        # 4. 物理軌跡の確認
        visualize_cluster_trajectories(routes, res_with, "With Koopman")
        visualize_cluster_trajectories(routes, res_without, "Without Koopman")
    
    print("Done.")