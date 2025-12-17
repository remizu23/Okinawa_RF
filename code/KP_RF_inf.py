import torch
import os
import glob
from network import Network
from tokenization import Tokenization
from KP_RF import KoopmanRoutesFormer
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"/home/mizutani/projects/RF/runs/figure2"
os.makedirs(out_dir, exist_ok=True)

def stamp(name):
    return os.path.join(out_dir, name)

# =========================================================
#  設定：ここを自分の環境に合わせて書き換えてください
# =========================================================
# 学習済みモデルのパス (.pthファイル)
# 例: "/home/mizutani/projects/RF/runs/202502XX_XXXXXX/model_weights_xxxx.pth"
# ※自動で最新のモデルを探すロジックも main() に入れていますが、指定すると確実です

MODEL_PATH = '/home/mizutani/projects/RF/runs/20251217_130327/model_weights_20251217_130327.pth'

# 隣接行列のパス (学習時と同じもの)
ADJ_PATH = '/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt'

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =========================================================

def load_model(model_path, network):
    """保存されたファイルからモデルを復元する"""
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 保存しておいた設定(config)を取り出す
    if 'config' not in checkpoint:
        raise ValueError("Config not found in checkpoint. Make sure to use the latest training code.")
        
    config = checkpoint['config']
    print(f"Model Config: {config}")

    # モデルの再構築 (configを使うのでパラメータ変更不要)
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
    model.to(device)
    model.eval() # 推論モードへ
    
    return model, config

import torch.nn.functional as F # 追加が必要ならファイルの冒頭に

def generate_route(model, network, start_node_id, max_len=50, strategy="sample", temperature=1.0):
    """
    strategy: "greedy" (最大確率), "sample" (確率的), "no_stay" (滞在禁止)
    temperature: 確率分布の平坦化 (高いほどランダム、低いほど保守的)
    """
    tokenizer = Tokenization(network)
    TOKEN_START = tokenizer.SPECIAL_TOKENS["<b>"]
    TOKEN_END   = tokenizer.SPECIAL_TOKENS["<e>"]
    
    current_seq = [TOKEN_START, start_node_id]
    z_history = [] # ★ zを記録するリスト 
    
    print(f"Generating route... Start Node: {start_node_id}")
    
    with torch.no_grad():
        for _ in range(max_len):
            input_tensor = torch.tensor([current_seq], dtype=torch.long).to(device)
            
            # z_hat (Transformerの推定値) を受け取る
            logits, z_hat, z_pred = model(input_tensor)

            # 最新時刻の z を取得して保存 (CPUに移してnumpy化)
            last_z = z_hat[0, -1, :].cpu().numpy()
            z_history.append(last_z)

            last_logits = logits[0, -1, :]
            
            # --- 戦略ごとの処理 ---
            
            # 戦略1: "no_stay" (強制移動モード)
            # 現在の場所(current_node)の確率を強制的にマイナス無限大にして選ばせない
            if strategy == "no_stay":
                current_node = current_seq[-1]
                last_logits[current_node] = float('-inf')
                # ついでに特殊トークン<b>なども選ばせない
                last_logits[TOKEN_START] = float('-inf')

            # 確率分布に変換 (Temperature付き)
            # temp < 1.0 : 確率高いものをより強調
            # temp > 1.0 : いろんな可能性を試す
            probs = F.softmax(last_logits / temperature, dim=0)
            
            if strategy == "greedy":
                next_token = torch.argmax(probs).item()
            else:
                # strategy="sample" or "no_stay" の場合は確率で抽選する
                next_token = torch.multinomial(probs, num_samples=1).item()
            
            # --- ループ終了判定 ---
            current_seq.append(next_token)
            if next_token == TOKEN_END:
                break
                
    return current_seq, z_history


def analyze_eigenvalues(model):
    """
    行列Aの固有値を解析・可視化する関数
    これが「円の内側」にあれば、システムは安定的です。
    """
    # 学習済み行列 A を取得 (tensor -> numpy)
    A_matrix = model.A.detach().cpu().numpy()
    
    # 固有値を計算
    eigenvalues, _ = np.linalg.eig(A_matrix)
    
    # プロット
    plt.figure(figsize=(6, 6))
    
    # 単位円を描く
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta), linestyle='--', color='gray', label='Unit Circle')
    
    # 固有値をプロット
    plt.scatter(eigenvalues.real, eigenvalues.imag, color='blue', marker='x', s=100, label='Eigenvalues of A')
    
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.title("Eigenvalues of Koopman Matrix A")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.grid(True)
    plt.legend()
    plt.axis('equal') # 縦横比を同じにする
    
    # 保存
    plt.savefig(stamp(f"eigenvalues_{run_id}.png"))
    print("Eigenvalue plot saved to eigenvalues_analysis.png")

def visualize_trajectory_with_time(z_history_list, start_nodes, title="Trajectories"):
    """
    複数の軌跡を、時間グラデーション付きでプロットする
    z_history_list: 各試行のzの履歴のリスト
    start_nodes: 各試行の開始ノード番号
    """
    # 全データをまとめてPCAにかける（空間を統一するため）
    all_z = np.concatenate(z_history_list, axis=0)
    pca = PCA(n_components=2)
    pca.fit(all_z)
    
    plt.figure(figsize=(10, 8))
    
    # カラーマップ (時間経過用)
    cmap = plt.get_cmap('viridis')
    
    for i, z_hist in enumerate(z_history_list):
        # この軌跡をPCA変換
        z_2d = pca.transform(np.array(z_hist))
        
        # 時間の長さ
        T = len(z_2d)
        
        # 線を描画（薄く）
        plt.plot(z_2d[:, 0], z_2d[:, 1], color='gray', alpha=0.3, linewidth=1)
        
        # 点を描画（時間経過で色を変える）
        # c=range(T) で時間による色付け
        sc = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=range(T), cmap=cmap, s=30, vmin=0, vmax=50, edgecolor='none')
        
        # スタート地点に番号を表示
        plt.text(z_2d[0, 0], z_2d[0, 1], str(start_nodes[i]), fontsize=12, fontweight='bold', color='red')

    plt.colorbar(sc, label='Time Step')
    plt.title(f"Koopman Latent Trajectories (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.savefig(stamp(f"PCA_{run_id}.png"))

# 冒頭のimportに追加が必要
from mpl_toolkits.mplot3d import Axes3D 

def visualize_trajectory_3d(z_history_list, start_nodes, title="Trajectories_3D"):
    """
    軌跡を3次元でプロットする関数
    """
    # 全データをまとめてPCAにかける
    all_z = np.concatenate(z_history_list, axis=0)
    
    # ★変更点1: 3次元に圧縮
    pca = PCA(n_components=3)
    pca.fit(all_z)
    
    fig = plt.figure(figsize=(10, 8))
    # ★変更点2: 3D projectionを指定
    ax = fig.add_subplot(111, projection='3d')
    
    # カラーマップ
    cmap = plt.get_cmap('viridis')
    
    for i, z_hist in enumerate(z_history_list):
        # 変換 (Steps, 3)
        z_3d = pca.transform(np.array(z_hist))
        T = len(z_3d)
        
        # 線を描画 (x, y, z)
        ax.plot(z_3d[:, 0], z_3d[:, 1], z_3d[:, 2], color='gray', alpha=0.3, linewidth=1)
        
        # 点を描画 (時間経過で色変え)
        sc = ax.scatter(z_3d[:, 0], z_3d[:, 1], z_3d[:, 2], 
                        c=range(T), cmap=cmap, s=20, vmin=0, vmax=50)
        
        # スタート地点に番号
        ax.text(z_3d[0, 0], z_3d[0, 1], z_3d[0, 2], 
                str(start_nodes[i]), fontsize=10, fontweight='bold', color='red')

    ax.set_title(f"Koopman Latent Trajectories (3D PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3") # Z軸ラベル
    
    fig.colorbar(sc, label='Time Step', pad=0.1)
    
    # ★ポイント: 静止画だと3Dは分かりにくいので、視点(角度)を変えて保存すると良いです
    # view_init(elev=仰角, azim=方位角)
    
    # パターン1: 標準的な角度
    ax.view_init(elev=30, azim=45)
    plt.savefig(f"{title}_view1.png")
    
    # パターン2: 上から見る
    ax.view_init(elev=80, azim=0)
    plt.savefig(f"{title}_top.png")
    
    # パターン3: 横から見る
    ax.view_init(elev=0, azim=90)
    plt.savefig(stamp(f"PCA3d_{run_id}.png"))
    
    print(f"3D Trajectory plots saved as {title}_*.png")

def main():
    # 1. Networkの準備 (学習時と同様にダミー特徴量で初期化)
    if not os.path.exists(ADJ_PATH):
        print(f"Error: Adjacency matrix not found at {ADJ_PATH}")
        return

    adj_matrix = torch.load(ADJ_PATH, weights_only=True)
    dummy_node_features = torch.zeros((len(adj_matrix), 1))
    network = Network(adj_matrix, dummy_node_features)
    
    # 2. モデルファイルの特定
    model_path = MODEL_PATH
    if model_path is None:
        # runsフォルダから一番新しいpthファイルを探す
        search_dir = "/home/mizutani/projects/RF/runs/"
        pth_files = glob.glob(os.path.join(search_dir, "*", "*.pth"))
        if not pth_files:
            print("No .pth files found in runs directory. Please specify MODEL_PATH manually.")
            return
        # 更新日時が新しい順にソート
        model_path = max(pth_files, key=os.path.getctime)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # 3. モデルロード
    model, config = load_model(model_path, network)
    
    # 1. 固有値解析を実行（ぜひ一度見てみてください）
    analyze_eigenvalues(model)
    
    # 2. まとめて推論して可視化
    all_z_histories = []
    target_start_nodes = [0, 1, 2, 3, 4, 5, 6, 7] # 試したいノード
    
    for start_node in target_start_nodes:
        # sampleモードで生成
        route, z_hist = generate_route(model, network, start_node, strategy="sample")
        all_z_histories.append(z_hist)
        
    # 3. グラデーション付きで可視化
    visualize_trajectory_with_time(all_z_histories, target_start_nodes, title="all_trajectories_colored")

    visualize_trajectory_3d(all_z_histories, target_start_nodes, title="all_trajectories_3d")

if __name__ == "__main__":
    main()