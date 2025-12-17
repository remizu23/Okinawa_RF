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
import torch.nn.functional as F

# =========================================================
#  設定：出力先ディレクトリ
# =========================================================
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
# 保存先フォルダ
out_dir = f"/home/mizutani/projects/RF/runs/comparison_{run_id}"
os.makedirs(out_dir, exist_ok=True)

def stamp(name):
    return os.path.join(out_dir, name)

# =========================================================
#  設定：モデルとデータのパス
# =========================================================
# 学習済みモデルのパス
MODEL_PATH = '/home/mizutani/projects/RF/runs/20251217_184803/model_weights_20251217_184803.pth'

# 隣接行列のパス
ADJ_PATH = '/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt'

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
#  1. 評価指標の計算ロジック (維持)
# =========================================================

def levenshtein_distance(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x): matrix[x, 0] = x
    for y in range(size_y): matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix[x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return matrix[size_x-1, size_y-1]

def evaluate_metrics(model, network, device):
    """
    データセットを使ってモデルの定量性能(Accuracy, Edit Distance)を評価する
    """
    print("\n=== Evaluating Quantitative Metrics ===")
    
    # データのロード (Trainと同じ手順)
    # ※パスは環境に合わせて修正してください
    trip_arrz = np.load('/home/mizutani/projects/RF/data/input_a.npz')
    trip_arr = trip_arrz['route_arr']
    route = torch.from_numpy(trip_arr)
    
    # 時間短縮のためサンプル数を制限 (全データなら len(route))
    num_samples = min(len(route), 1000) 
    subset_route = route[:num_samples]
    
    tokenizer = Tokenization(network)
    model.eval()
    
    total_acc = 0
    total_tokens = 0
    total_edit_dist = 0
    
    batch_size = 64
    num_batches = (len(subset_route) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            batch_routes = subset_route[i*batch_size : (i+1)*batch_size].to(device)
            
            # 入力と正解
            input_tokens = tokenizer.tokenization(batch_routes, mode="simple").long().to(device)
            target_tokens = tokenizer.tokenization(batch_routes, mode="next").long().to(device)
            
            # 推論
            logits, _, _ = model(input_tokens)
            preds = torch.argmax(logits, dim=-1) # [B, T]
            
            # --- Accuracy ---
            # パディング(network.N)以外の部分で正解率を計算
            mask = target_tokens != network.N 
            correct = (preds == target_tokens) & mask
            
            total_acc += correct.sum().item()
            total_tokens += mask.sum().item()
            
            # --- Edit Distance ---
            for j in range(len(batch_routes)):
                # tensor -> list (padding除去)
                p_seq = preds[j][mask[j]].cpu().tolist()
                t_seq = target_tokens[j][mask[j]].cpu().tolist()
                
                dist = levenshtein_distance(p_seq, t_seq)
                total_edit_dist += dist

    avg_acc = total_acc / total_tokens if total_tokens > 0 else 0
    avg_edit = total_edit_dist / num_samples
    
    print(f"Samples: {num_samples}")
    print(f"Next Token Accuracy: {avg_acc:.4f} (Higher is better)")
    print(f"Avg Edit Distance:   {avg_edit:.4f} (Lower is better)")
    print("=======================================\n")
    return avg_acc, avg_edit

# =========================================================
#  2. モデル関連関数
# =========================================================

def load_model(model_path, network):
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'config' not in checkpoint:
        raise ValueError("Config not found in checkpoint.")
        
    config = checkpoint['config']
    
    # Koopman設定の表示
    use_koopman = config.get('use_koopman_loss', 'Unknown')
    print(f"Model Config - Use Koopman: {use_koopman}")

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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config

def generate_route(model, network, start_node_id, max_len=50, strategy="sample", temperature=1.0):
    tokenizer = Tokenization(network)
    TOKEN_START = tokenizer.SPECIAL_TOKENS["<b>"]
    TOKEN_END   = tokenizer.SPECIAL_TOKENS["<e>"]
    
    current_seq = [TOKEN_START, start_node_id]
    z_history = [] 
    
    with torch.no_grad():
        for _ in range(max_len):
            input_tensor = torch.tensor([current_seq], dtype=torch.long).to(device)
            
            logits, z_hat, z_pred = model(input_tensor)

            last_z = z_hat[0, -1, :].cpu().numpy()
            z_history.append(last_z)

            last_logits = logits[0, -1, :]
            
            if strategy == "no_stay":
                current_node = current_seq[-1]
                last_logits[current_node] = float('-inf')
                last_logits[TOKEN_START] = float('-inf')

            probs = F.softmax(last_logits / temperature, dim=0)
            
            if strategy == "greedy":
                next_token = torch.argmax(probs).item()
            else:
                next_token = torch.multinomial(probs, num_samples=1).item()
            
            current_seq.append(next_token)
            if next_token == TOKEN_END:
                break
                
    return current_seq, z_history

# =========================================================
#  3. 可視化ロジック (グラデーション表示・複数地点)
# =========================================================

def analyze_eigenvalues(model):
    """固有値解析"""
    A_matrix = model.A.detach().cpu().numpy()
    eigenvalues, _ = np.linalg.eig(A_matrix)
    
    plt.figure(figsize=(6, 6))
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta), linestyle='--', color='gray', label='Unit Circle')
    plt.scatter(eigenvalues.real, eigenvalues.imag, color='blue', marker='x', s=100, label='Eigenvalues')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.title("Eigenvalues of Koopman Matrix A")
    plt.grid(True)
    plt.axis('equal')
    
    save_path = stamp("eigenvalues.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved eigenvalues plot: {save_path}")

def visualize_trajectory_with_time(z_history_list, routes_list, start_nodes, title="Trajectories"):
    """
    複数の軌跡を、時間グラデーション付きで2次元プロットし、各点にノード番号を表示する
    routes_list: 生成されたルートIDのリスト (z_history_listと対応)
    """
    # 全データをまとめてPCAにかける
    all_z = np.concatenate(z_history_list, axis=0)
    pca = PCA(n_components=2)
    pca.fit(all_z)
    
    plt.figure(figsize=(12, 10)) # 少し大きくする
    cmap = plt.get_cmap('viridis')
    
    for i, z_hist in enumerate(z_history_list):
        z_2d = pca.transform(np.array(z_hist))
        route = routes_list[i]
        
        # 線を描画
        plt.plot(z_2d[:, 0], z_2d[:, 1], color='gray', alpha=0.3, linewidth=1)
        
        # 点を描画
        sc = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=range(len(z_2d)), cmap=cmap, s=30, vmin=0, vmax=50, edgecolor='none')
        
        # ★改良点: 各点にノード番号を表示
        # z_hist[k] は route[k+1] (Start Node以降) に対応します
        for k in range(len(z_2d)):
            # 対応するノードID
            # routeは [<b>, start, next1, ...] なので、z_hist[0]に対応するのは route[1]
            if k + 1 < len(route):
                node_id = route[k+1]
                
                # スタート地点(k=0)は赤く太字で、それ以外は小さく表示
                if k == 0:
                    plt.text(z_2d[k, 0], z_2d[k, 1], str(node_id), fontsize=12, fontweight='bold', color='red', zorder=10)
                else:
                    # 文字が重ならないように少しずらす
                    plt.text(z_2d[k, 0]+0.02, z_2d[k, 1]+0.02, str(node_id), fontsize=8, color='black')

    plt.colorbar(sc, label='Time Step')
    plt.title(f"Koopman Latent Trajectories (2D PCA) with Node IDs")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)

    save_path = stamp(f"{title}_2d.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved 2D plot: {save_path}")

def visualize_trajectory_3d(z_history_list, routes_list, start_nodes, title="Trajectories"):
    """
    軌跡を3次元でプロットし、各点にノード番号を表示する
    """
    all_z = np.concatenate(z_history_list, axis=0)
    
    pca = PCA(n_components=3)
    pca.fit(all_z)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap('viridis')
    
    for i, z_hist in enumerate(z_history_list):
        z_3d = pca.transform(np.array(z_hist))
        route = routes_list[i]
        
        ax.plot(z_3d[:, 0], z_3d[:, 1], z_3d[:, 2], color='gray', alpha=0.3, linewidth=1)
        sc = ax.scatter(z_3d[:, 0], z_3d[:, 1], z_3d[:, 2], 
                        c=range(len(z_3d)), cmap=cmap, s=20, vmin=0, vmax=50)
        
        # ★改良点: 3D空間でのテキスト表示
        for k in range(len(z_3d)):
            if k + 1 < len(route):
                node_id = route[k+1]
                if k == 0:
                    ax.text(z_3d[k, 0], z_3d[k, 1], z_3d[k, 2], str(node_id), fontsize=10, fontweight='bold', color='red')
                else:
                    ax.text(z_3d[k, 0], z_3d[k, 1], z_3d[k, 2], str(node_id), fontsize=6, color='black')

    ax.set_title(f"Koopman Latent Trajectories (3D PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    fig.colorbar(sc, label='Time Step', pad=0.1)

    # 視点を変えて3パターン保存
    ax.view_init(elev=30, azim=45)
    plt.savefig(stamp(f"{title}_3d_view1.png"))
    
    ax.view_init(elev=80, azim=0)
    plt.savefig(stamp(f"{title}_3d_top.png"))
    
    ax.view_init(elev=0, azim=90)
    plt.savefig(stamp(f"{title}_3d_side.png"))
    
    plt.close()
    print(f"Saved 3D plots to: {out_dir}")


def main():
    # 1. 準備
    if not os.path.exists(ADJ_PATH):
        print(f"Error: Adjacency matrix not found at {ADJ_PATH}")
        return

    adj_matrix = torch.load(ADJ_PATH, weights_only=True)
    dummy_node_features = torch.zeros((len(adj_matrix), 1))
    network = Network(adj_matrix, dummy_node_features)
    
    # 2. モデルロード
    model_path = MODEL_PATH
    if model_path is None:
        search_dir = "/home/mizutani/projects/RF/runs/"
        pth_files = glob.glob(os.path.join(search_dir, "*", "*.pth"))
        if pth_files:
            model_path = max(pth_files, key=os.path.getctime)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    model, config = load_model(model_path, network)
    
    # ★ 3. 定量評価の実行 (Accuracy, EditDistance)
    acc, edit_dist = evaluate_metrics(model, network, device)

    # 結果をテキストファイルに書き込む
    metrics_file = stamp(f"metrics_{run_id}.txt")

    with open(metrics_file, "w") as f:
        f.write("=== Quantitative Metrics ===\n")
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Use Koopman: {config.get('use_koopman_loss', 'Unknown')}\n")
        f.write("-" * 30 + "\n")
        f.write(f"Next Token Accuracy: {acc:.4f}\n")
        f.write(f"Avg Edit Distance:   {edit_dist:.4f}\n")
        f.write("============================\n")

    # ★ 4. 固有値解析
    # analyze_eigenvalues(model)
    
    # ★ 5. 複数ノードからの推論実行 & グラデーション可視化
    all_z_histories = []
    all_routes = [] # ★ルートを保存するリストを追加

    # 可視化したいスタート地点のリスト
    target_start_nodes = [0,1,2,3,4,5,6,7] 
    
    print(f"Generating routes from nodes: {target_start_nodes}")
    
    for start_node in target_start_nodes:
        # sampleモードで1回生成
        route, z_hist = generate_route(model, network, start_node, strategy="sample", temperature=1.0)
        all_z_histories.append(z_hist)

        all_routes.append(route) # ★保存
        
    # 可視化 (2D & 3D)
    visualize_trajectory_with_time(all_z_histories, all_routes, target_start_nodes, title="routes")
    visualize_trajectory_3d(all_z_histories, all_routes, target_start_nodes, title="routes")

if __name__ == "__main__":
    main()