import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 自作モジュール (同じディレクトリにある前提)
from KP_RF import KoopmanRoutesFormer

# ==========================================
# 設定: 解析したいモデルのパスを指定してください
# ==========================================
MODEL_PATH = '/home/mizutani/projects/RF/runs/20260105_165834/model_weights_20260105_165834.pth'
graph_filename = '/home/mizutani/projects/RF/runs/20260105_165834/AB.png'
# ==========================================

def load_model_for_inspection(model_path):
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None, None

    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu')) # 解析だけならCPUでOK
    
    if 'config' not in checkpoint:
        print("Error: Config not found in checkpoint.")
        return None, None
        
    config = checkpoint['config']
    
    # Configの内容を表示（確認用）
    print("\n--- Model Config ---")
    for k, v in config.items():
        print(f"{k}: {v}")
    print("--------------------\n")

    # モデルの構築 (Configを使って動的に初期化)
    # ※新しいコードで学習した場合は config に num_agents 等が含まれているはずです
    # ※古いコードのモデルを読み込む場合のエラー回避のため、getでデフォルト値を設定するか、
    #   **config で展開して渡すのが一番汎用的です。
    
    try:
        model = KoopmanRoutesFormer(**config)
    except TypeError as e:
        print(f"Warning: Config arguments mismatch. Trying manual mapping... ({e})")
        # 万が一 **config が失敗した場合のフォールバック (古いモデル読み込み用など)
        model = KoopmanRoutesFormer(
            vocab_size=config['vocab_size'],
            token_emb_dim=config['token_emb_dim'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            d_ff=config['d_ff'],
            z_dim=config['z_dim'],
            pad_token_id=config['pad_token_id']
        )

    # 重みのロード
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config

def visualize_matrices(model):
    # パラメータを取り出してNumpy配列に変換
    A_matrix = model.A.detach().cpu().numpy()
    B_matrix = model.B.detach().cpu().numpy()
    
    print(f"Matrix A shape: {A_matrix.shape}")
    print(f"Matrix B shape: {B_matrix.shape}")
    
    # --- 数値の統計情報を表示 ---
    print("\n=== Matrix A Stats ===")
    print(f"Mean: {np.mean(A_matrix):.4f}, Std: {np.std(A_matrix):.4f}")
    print(f"Max:  {np.max(A_matrix):.4f}, Min: {np.min(A_matrix):.4f}")
    # 対角成分の平均（単位行列に近いかどうかの指標）
    diag_mean = np.mean(np.diag(A_matrix))
    print(f"Diagonal Mean: {diag_mean:.4f} (Close to 1.0 means stable identity-like dynamics)")

    print("\n=== Matrix B Stats ===")
    print(f"Mean: {np.mean(B_matrix):.4f}, Std: {np.std(B_matrix):.4f}")
    print(f"Max:  {np.max(B_matrix):.4f}, Min: {np.min(B_matrix):.4f}")

    # --- ヒートマップで可視化 ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 行列A
    sns.heatmap(A_matrix, ax=axes[0], cmap="RdBu_r", center=0, annot=False)
    axes[0].set_title("Koopman Matrix A (Dynamics)")
    axes[0].set_xlabel("z_t (Current State)")
    axes[0].set_ylabel("z_{t+1} (Next State)")

    # 行列B
    # Bは横長になる可能性があるため、サイズによっては転置して表示しても良いかも
    sns.heatmap(B_matrix, ax=axes[1], cmap="RdBu_r", center=0, annot=False)
    axes[1].set_title("Input Matrix B (Control/Forcing)")
    axes[1].set_xlabel("u_t (Input Embedding)")
    axes[1].set_ylabel("z_{t+1} (Influence on State)")

    plt.tight_layout()
    plt.show()
    plt.savefig(graph_filename)
    

    # --- 具体的な値の表示（最初の数行）---
    print("\n=== Matrix A (First 5x5) ===")
    print(np.round(A_matrix[:5, :5], 3))
    
    print("\n=== Matrix B (First 5x5) ===")
    print(np.round(B_matrix[:5, :5], 3))

if __name__ == "__main__":
    model, config = load_model_for_inspection(MODEL_PATH)
    if model:
        visualize_matrices(model)