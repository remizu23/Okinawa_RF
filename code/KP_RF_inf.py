import torch
import os
import glob
from network import Network, expand_adjacency_matrix
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

trip_arrz = np.load('/home/mizutani/projects/RF/data/input_real.npz')

# 学習済みモデルのパス
MODEL_PATH = '/home/mizutani/projects/RF/runs/20260111_200333/model_weights_best_20260111_200333.pth'

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
    trip_arr = trip_arrz['route_arr']
    route = torch.from_numpy(trip_arr)

    if 'agent_ids' in trip_arrz:
        agent_ids_arr = trip_arrz['agent_ids']
    else:
        # なければ仮のID (全員0) を作成、あるいはエラーにする
        print("Warning: 'agent_ids' not found in npz. Using dummy IDs.")
        agent_ids_arr = np.zeros(len(trip_arr), dtype=int)
    
    # 時間短縮のためサンプル数を制限 (全データなら len(route))
    num_samples = min(len(route), 1000) 
    subset_route = route[:num_samples]
    subset_agents = torch.from_numpy(agent_ids_arr[:num_samples]).long()
    
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
            batch_agents = subset_agents[i*batch_size : (i+1)*batch_size].to(device)
            
            # 入力と正解
            input_tokens = tokenizer.tokenization(batch_routes, mode="simple").long().to(device)
            target_tokens = tokenizer.tokenization(batch_routes, mode="next").long().to(device)
            
            # 滞在カウント計算
            stay_counts = tokenizer.calculate_stay_counts(input_tokens)

            # ★修正: 4つの戻り値を受け取る (logits, z_hat, z_pred, u_all)
            # u_allは評価では使わないので _ で受ける
            logits, _, _, _ = model(input_tokens, stay_counts, batch_agents)
            
            preds = torch.argmax(logits, dim=-1) # [B, T]
            
            # --- Accuracy ---
            mask = target_tokens != network.N 
            correct = (preds == target_tokens) & mask
            
            total_acc += correct.sum().item()
            total_tokens += mask.sum().item()
            
            # --- Edit Distance ---
            for j in range(len(batch_routes)):
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
#  2. モデル関連関数 (修正版)
# =========================================================

def load_model(model_path, network):
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'config' not in checkpoint:
        raise ValueError("Config not found in checkpoint.")
        
    config = checkpoint['config']
    
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
        pad_token_id=config['pad_token_id'],
        num_agents=config.get('num_agents', 1),
        agent_emb_dim=config.get('agent_emb_dim', 16),
        max_stay_count=config.get('max_stay_count', 500),
        stay_emb_dim=config.get('stay_emb_dim', 16)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config

# =========================================================
#  3. 生成ロジック (修正版)
# =========================================================

def generate_route(model, network, start_node_id, agent_id, max_len=50, strategy="sample", temperature=1.0):
    tokenizer = Tokenization(network)
    TOKEN_START = tokenizer.SPECIAL_TOKENS["<b>"]
    TOKEN_END   = tokenizer.SPECIAL_TOKENS["<e>"]
    
    current_seq = [TOKEN_START, start_node_id]
    current_stay_counts = [0, 1]
    
    z_history = [] 
    
    with torch.no_grad():
        for _ in range(max_len):
            input_tensor = torch.tensor([current_seq], dtype=torch.long).to(device)
            stay_tensor = torch.tensor([current_stay_counts], dtype=torch.long).to(device)
            agent_tensor = torch.tensor([agent_id], dtype=torch.long).to(device)
            
            # ★修正: 4つの戻り値を受け取る
            # 生成時は u_all は不要なので無視
            logits, z_hat, z_pred, _ = model(input_tensor, stay_tensor, agent_tensor)
            
            last_z = z_hat[0, -1, :].cpu().numpy()
            z_history.append(last_z)

            last_logits = logits[0, -1, :]
            
            if strategy == "no_stay":
                current_node = current_seq[-1]
                last_logits[current_node] = float('-inf')
                last_logits[TOKEN_START] = float('-inf')

            probs = F.softmax(last_logits / temperature, dim=0)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            current_seq.append(next_token)
            
            last_token = current_seq[-2]
            last_count = current_stay_counts[-1]
            
            if next_token == last_token:
                new_count = last_count + 1
            else:
                new_count = 1
            
            if next_token in tokenizer.SPECIAL_TOKENS.values():
                new_count = 0
            
            current_stay_counts.append(new_count)

            if next_token == TOKEN_END:
                break
                
    return current_seq, z_history


# =========================================================
#  4. 可視化ロジック (維持)
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
    all_z = np.concatenate(z_history_list, axis=0)
    pca = PCA(n_components=2)
    pca.fit(all_z)
    
    plt.figure(figsize=(12, 10))
    cmap = plt.get_cmap('viridis')
    
    for i, z_hist in enumerate(z_history_list):
        z_2d = pca.transform(np.array(z_hist))
        route = routes_list[i]
        
        plt.plot(z_2d[:, 0], z_2d[:, 1], color='gray', alpha=0.3, linewidth=1)
        sc = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=range(len(z_2d)), cmap=cmap, s=30, vmin=0, vmax=50, edgecolor='none')
        
        for k in range(len(z_2d)):
            if k + 1 < len(route):
                node_id = route[k+1]
                if k == 0:
                    plt.text(z_2d[k, 0], z_2d[k, 1], str(node_id), fontsize=12, fontweight='bold', color='red', zorder=10)
                else:
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

    ax.view_init(elev=30, azim=45)
    plt.savefig(stamp(f"{title}_3d_view1.png"))
    
    ax.view_init(elev=80, azim=0)
    plt.savefig(stamp(f"{title}_3d_top.png"))
    
    ax.view_init(elev=0, azim=90)
    plt.savefig(stamp(f"{title}_3d_side.png"))
    
    plt.close()
    print(f"Saved 3D plots to: {out_dir}")

# 固定点図
def analyze_stability_and_dynamics(model, network, device):
    """
    線形システムの安定性解析と、ノードごとの力学特性を可視化する
    """
    print("\n=== Linear System Analysis ===")
    
    # 1. 固有値解析 (Eigenvalue Analysis)
    A_np = model.A.detach().cpu().numpy()
    eigenvalues, eigenvectors = np.linalg.eig(A_np)
    
    # 絶対値（大きさ）を確認
    abs_eig = np.abs(eigenvalues)
    max_eig = np.max(abs_eig)
    
    print(f"Max Eigenvalue |λ|: {max_eig:.4f}")
    if max_eig > 1.0:
        print("-> System is UNSTABLE (Explodes over time)")
    elif max_eig > 0.99:
        print("-> System is MARGINALLY STABLE (Good for memory/cycles)")
    else:
        print("-> System is STABLE (Decays to zero, forgets quickly)")

    # プロット: 単位円と固有値
    # plt.figure(figsize=(6, 6))
    # theta = np.linspace(0, 2*np.pi, 100)
    # plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Unit Circle')
    # plt.scatter(eigenvalues.real, eigenvalues.imag, c='red', marker='x', label='Eigenvalues')
    # plt.axhline(0, color='gray', lw=0.5)
    # plt.axvline(0, color='gray', lw=0.5)
    # plt.title(f"Stability Analysis (Max |λ|={max_eig:.3f})")
    # plt.xlabel("Real Part")
    # plt.ylabel("Imaginary Part")
    # plt.legend()
    # plt.grid(True)
    # plt.axis('equal')
    # plt.savefig(stamp("analysis_eigenvalues.png"))
    # plt.close()
    
    # 2. 固定点解析 (Fixed Point Analysis)
    # z_{t+1} = A z_t + B u_t において、z_{t+1} = z_t となる点 z* を探す
    # z* = (I - A)^(-1) B u
    # これが「そのノードにずっと留まろうとする時の重心位置」
    
    I = np.eye(A_np.shape[0])
    try:
        # (I - A) の逆行列
        Inv_I_minus_A = np.linalg.inv(I - A_np)
        has_inverse = True
    except np.linalg.LinAlgError:
        print("Matrix (I-A) is singular. System has purely integral components.")
        has_inverse = False

    if has_inverse:
        B_np = model.B.detach().cpu().numpy()
        
        # 各ノードの埋め込み u を取得 (滞在カウント0の状態)
        # Agent IDなどはとりあえず0番で固定
        tokenizer = Tokenization(network)
        node_indices = torch.arange(network.N).to(device) # 全ノード
        dummy_counts = torch.zeros_like(node_indices).to(device)
        dummy_agents = torch.zeros(network.N, dtype=torch.long).to(device)
        
        # モデルから u_all を取得するため、一度forwardを回す必要があるが、
        # ここでは内部構造を知っているので手動で埋め込み作成
        with torch.no_grad():
            token_vec = model.token_embedding(node_indices)
            stay_vec = model.stay_embedding(dummy_counts)
            agent_vec = model.agent_embedding(dummy_agents).unsqueeze(1) # shape合わせが必要かも
            # 簡易的にtoken_vecだけでBとの積を見る (uの主成分はtokenなので)
            # 正確には結合が必要ですが、傾向を見るにはこれでもOK
            
            # B行列の形状に合わせて入力作成 (B: z_dim x input_dim)
            # ここでは簡単のため、input_dim全体を取得するのは手間なので
            # "入力 u によって z がどう動かされるか (B u)" だけを計算
            
            # KP_RF.pyのforwardを参考に u_all を作る
            u_all = torch.cat([token_vec, stay_vec, model.agent_embedding(dummy_agents)], dim=-1)
            u_np = u_all.cpu().numpy() # [N, input_dim]
            
            # 各ノードの固定点 z* = (I-A)^(-1) * B * u
            # B: [z_dim, u_dim], u: [N, u_dim] -> B u.T : [z_dim, N]
            forcing = B_np @ u_np.T 
            fixed_points = Inv_I_minus_A @ forcing # [z_dim, N]
            fixed_points = fixed_points.T # [N, z_dim]
            
            # PCAで可視化
            pca = PCA(n_components=2)
            z_2d = pca.fit_transform(fixed_points)
            
            plt.figure(figsize=(10, 8))
            plt.scatter(z_2d[:, 0], z_2d[:, 1], c='blue', s=100, alpha=0.6)
            
            for i in range(network.N):
                plt.text(z_2d[i, 0], z_2d[i, 1], str(i), fontsize=12, fontweight='bold')
                
            plt.title("Attractors (Fixed Points) of Each Node in Latent Space")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.grid(True)
            plt.savefig(stamp("fixed_points.png"))
            plt.close()
            print("Saved fixed points analysis.")


def main():
    if not os.path.exists(ADJ_PATH):
        print(f"Error: Adjacency matrix not found at {ADJ_PATH}")
        return

    adj_matrix = torch.load(ADJ_PATH, weights_only=True)
    expanded_adj = expand_adjacency_matrix(adj_matrix)
    dummy_feature_dim = 1
    dummy_node_features = torch.zeros((len(adj_matrix), dummy_feature_dim))
    expanded_features = torch.cat([dummy_node_features, dummy_node_features], dim=0)
    network = Network(expanded_adj, expanded_features)    
    
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
    
    acc, edit_dist = evaluate_metrics(model, network, device)
    
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

    analyze_eigenvalues(model)

    analyze_stability_and_dynamics(model, network, device)
    
    all_z_histories = []
    all_routes = []
    target_start_nodes = [0,1,2,3,4,5,6,7] 
    
    test_agent_id = 0 
    print(f"Generating routes from nodes: {target_start_nodes} for Agent ID: {test_agent_id}")
    
    for start_node in target_start_nodes:
        route, z_hist = generate_route(model, network, start_node, agent_id=test_agent_id, strategy="sample", temperature=1.0)
        all_z_histories.append(z_hist)
        all_routes.append(route)        
    
    visualize_trajectory_with_time(all_z_histories, all_routes, target_start_nodes, title="routes")
    visualize_trajectory_3d(all_z_histories, all_routes, target_start_nodes, title="routes")

if __name__ == "__main__":
    main()