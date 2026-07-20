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
import networkx as nx


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

trip_arrz = np.load('/home/mizutani/projects/RF/data/input_real_m4.npz')

# 学習済みモデルのパス
MODEL_PATH = '/home/mizutani/projects/RF/runs/20260121_145835/model_weights_20260121_145835.pth'

# 隣接行列のパス
ADJ_PATH = '/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt'

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_shortest_path_distance_matrix(adj: torch.Tensor, directed: bool = False) -> torch.Tensor:
    """All-pairs shortest-path hop distances from adjacency matrix.

    Unreachable pairs are set to N+1.
    """
    if not isinstance(adj, torch.Tensor):
        adj = torch.tensor(adj)
    adj_cpu = adj.detach().cpu()
    N = adj_cpu.shape[0]
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(N))
    rows, cols = torch.nonzero(adj_cpu, as_tuple=True)
    edges = [(int(r), int(c)) for r, c in zip(rows, cols) if int(r) != int(c)]
    G.add_edges_from(edges)

    dist = torch.full((N, N), fill_value=N + 1, dtype=torch.long)
    for s in range(N):
        dist[s, s] = 0
        for t, d in nx.single_source_shortest_path_length(G, s).items():
            dist[s, t] = int(d)
    return dist

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
    
    # データのロード
    trip_arr = trip_arrz['route_arr']
    route = torch.from_numpy(trip_arr)

    # Agent IDのロード
    if 'agent_ids' in trip_arrz:
        agent_ids_arr = trip_arrz['agent_ids']
    else:
        print("Warning: 'agent_ids' not found in npz. Using dummy IDs.")
        agent_ids_arr = np.zeros(len(trip_arr), dtype=int)

    # ★追加: Time情報のロード
    if 'time_arr' in trip_arrz:
        time_arr = trip_arrz['time_arr']
    else:
        print("Warning: 'time_arr' not found. Using dummy 2025 time.")
        # ない場合は全員 2025/11/22 10:00 とする (広場あり評価のため)
        time_arr = np.full(len(trip_arr), 202511221000, dtype=np.int64)

    # サンプル数制限
    num_samples = min(len(route), 1000) 
    subset_route = route[:num_samples]
    subset_agents = torch.from_numpy(agent_ids_arr[:num_samples]).long()
    # ★追加: サブセット抽出
    subset_time = torch.from_numpy(time_arr[:num_samples]).long()
    
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
            # ★追加: Timeバッチ作成
            batch_time = subset_time[i*batch_size : (i+1)*batch_size].to(device)
            
            input_tokens = tokenizer.tokenization(batch_routes, mode="simple").long().to(device)
            target_tokens = tokenizer.tokenization(batch_routes, mode="next").long().to(device)
            
            stay_counts = tokenizer.calculate_stay_counts(input_tokens)

            # ★修正: time_tensor を渡す
            # 戻り値も4つで受け取る
            logits, _, _, _ = model(input_tokens, stay_counts, batch_agents, time_tensor=batch_time)
            
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

def load_model(model_path, network, dist_mat_base: torch.Tensor | None = None, base_N: int | None = None, dist_is_directed: bool = False):
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'config' not in checkpoint:
        raise ValueError("Config not found in checkpoint.")
        
    config = checkpoint['config']
    
    use_koopman = config.get('use_koopman_loss', 'Unknown')
    print(f"Model Config - Use Koopman: {use_koopman}")

    # ★修正1: 昔のモデル(100)か今のモデル(500)かを、チェックポイント内の重み形状から自動判定するロジック
    # configに正しく保存されていない場合もあるため、念のためstate_dictを見る
    state_dict = checkpoint['model_state_dict']
    if 'stay_embedding.weight' in state_dict:
        # 形状を取得 (例: [101, 16] or [501, 16])
        num_embeddings = state_dict['stay_embedding.weight'].shape[0]
        actual_max_stay_count = num_embeddings - 1
        print(f"Detected max_stay_count from weights: {actual_max_stay_count}")
    else:
        # なければconfigを信じる、あるいはデフォルト
        actual_max_stay_count = config.get('max_stay_count', 100)

    model = KoopmanRoutesFormer(
        vocab_size=config['vocab_size'],
        token_emb_dim=config['token_emb_dim'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        z_dim=config['z_dim'],
        pad_token_id=config['pad_token_id'],
        # --- Δ距離(広場)バイアス用 ---
        dist_mat_base=dist_mat_base,
        base_N=base_N,
        delta_bias_move_only=True,
        dist_is_directed=dist_is_directed,
        num_agents=config.get('num_agents', 1),
        agent_emb_dim=config.get('agent_emb_dim', 16),
        # ★ここで検出した値を適用
        max_stay_count=actual_max_stay_count, 
        stay_emb_dim=config.get('stay_emb_dim', 16)
    )
    
    # ★修正2: mode_classifierなどが無くてもエラーにしない (strict=False)
    # これにより、昔のモデルに無かった層は「初期化されたまま（ランダムな値）」になりますが、
    # 推論やzの分析をする分には、その層を使わなければ問題ありません。
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys in checkpoint (OK for legacy models): {missing_keys}")
    
    model.to(device)
    model.eval()
    
    return model, config

# =========================================================
#  3. 生成ロジック (修正版)
# =========================================================

def generate_route(model, network, start_node_id, agent_id, max_len=500, strategy="sample", temperature=1.0):
    tokenizer = Tokenization(network)
    TOKEN_START = tokenizer.SPECIAL_TOKENS["<b>"]
    TOKEN_END   = tokenizer.SPECIAL_TOKENS["<e>"]
    
    current_seq = [TOKEN_START, start_node_id]
    current_stay_counts = [0, 1]
    
    z_history = [] 
    u_history = [] # ★追加: 入力uの履歴を保存
    probs_history = [] # ★追加: 確率分布の履歴
    
    max_model_len = 500 #positional encodingやstay_countと同様

    dummy_time = torch.tensor([202511221200], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_len):
            
            if len(current_seq) >= max_model_len:
                print(f"Force stopping: Sequence length reached model limit ({max_model_len})")
                break

            input_tensor = torch.tensor([current_seq], dtype=torch.long).to(device)
            stay_tensor = torch.tensor([current_stay_counts], dtype=torch.long).to(device)
            agent_tensor = torch.tensor([agent_id], dtype=torch.long).to(device)
            
            # ★修正: 4番目の戻り値 u_all を受け取る
            logits, z_hat, z_pred, u_all = model(input_tensor, stay_tensor, agent_tensor, time_tensor=dummy_time)            
            
            # zの保存
            last_z = z_hat[0, -1, :].cpu().numpy()
            z_history.append(last_z)

            # ★追加: u(入力)の保存
            # u_all: [Batch, Seq, InputDim] -> 最後の時刻のベクトルを取得
            last_u = u_all[0, -1, :].cpu().numpy()
            u_history.append(last_u)

            last_logits = logits[0, -1, :]
            
            if strategy == "no_stay":
                current_node = current_seq[-1]
                last_logits[current_node] = float('-inf')
                last_logits[TOKEN_START] = float('-inf')

            probs = F.softmax(last_logits / temperature, dim=0)
            
            # ★追加: 確率分布を保存 (CPU numpyへ)
            probs_history.append(probs.cpu().numpy())

            # 戦略による分岐 
            if strategy == "greedy":
                # 最大確率のトークンを確定的に選ぶ
                next_token = torch.argmax(probs).item()
            else:
                # 確率分布に従ってサンプリング (デフォルト)
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

            if next_token == TOKEN_END: #endトークンの出力
                break
                
    return current_seq, z_history, u_history, probs_history # ★戻り値に追加

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
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved eigenvalues plot: {save_path}")

def visualize_trajectory_with_time(z_history_list, routes_list, start_nodes, title="Trajectories"):
    all_z = np.concatenate(z_history_list, axis=0)
    pca = PCA(n_components=2)
    pca.fit(all_z)
    
    plt.figure(figsize=(14, 12))
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
    plt.savefig(save_path, dpi=300)
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
    plt.savefig(stamp(f"{title}_3d_view1.png"), dpi=300)
    
    ax.view_init(elev=80, azim=0)
    plt.savefig(stamp(f"{title}_3d_top.png"), dpi=300)
    
    ax.view_init(elev=0, azim=90)
    plt.savefig(stamp(f"{title}_3d_side.png"), dpi=300)
    
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
            agent_vec = model.agent_embedding(dummy_agents)# shape合わせが必要かも
            # 簡易的にtoken_vecだけでBとの積を見る (uの主成分はtokenなので)
            # 正確には結合が必要ですが、傾向を見るにはこれでもOK
            
            # B行列の形状に合わせて入力作成 (B: z_dim x input_dim)
            # ここでは簡単のため、input_dim全体を取得するのは手間なので
            # "入力 u によって z がどう動かされるか (B u)" だけを計算
            
            # KP_RF.pyのforwardを参考に u_all を作る
            # ★★★ 修正: ここに広場埋め込み(Plaza Embedding)を追加 ★★★
            
            # 広場なし(0)の状態として作成
            dummy_plaza_status = torch.zeros_like(node_indices).long().to(device) 
            plaza_vec = model.plaza_embedding(dummy_plaza_status) # [N, 4]

            # 結合: Token + Stay + Agent + Plaza (合計100次元になるはず)
            u_all = torch.cat([token_vec, stay_vec, agent_vec, plaza_vec], dim=-1)
            
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
            plt.savefig(stamp("fixed_points.png"), dpi=300)
            plt.close()
            print("Saved fixed points analysis.")

def visualize_z_all_dimensions(z_history_list, routes_list, start_nodes, run_id):
    """
    潜在変数 z (16次元) の全次元の時系列変化を可視化する
    1. ヒートマップ (全体俯瞰)
    2. 多段ラインプロット (詳細)
    """
    import seaborn as sns
    
    # 全サンプル可視化
    num_samples_to_plot = min(len(z_history_list), 8)

    local_dir = os.path.join(out_dir, "z_history")
    os.makedirs(local_dir, exist_ok=True)
    def local_stamp(name):
        return os.path.join(local_dir, name) # ★ローカルスタンプ定義

    for i in range(num_samples_to_plot):
        z_seq = np.array(z_history_list[i]) # [SeqLen, z_dim]
        route = routes_list[i]
        seq_len, z_dim = z_seq.shape
        
        start_node = start_nodes[i]
        
        # --- A. ヒートマップ (次元ごとの活動度) ---
        # ★修正: ラベルを route[1:] に合わせる
        generated_tokens = route[1:]
        disp_tokens = generated_tokens[:seq_len]

        plt.figure(figsize=(12, 6))
        sns.heatmap(z_seq.T, cmap="viridis", center=0, cbar=True)
        plt.title(f"Latent State Heatmap (Sample {i})")
        plt.xlabel("Time Step")
        plt.ylabel("Dimension Index")
        if seq_len < 60:
            step_ticks = np.arange(seq_len)
            plt.xticks(step_ticks + 0.5, disp_tokens, rotation=0, fontsize=10) # 修正

        save_path = local_stamp(f"z_heatmap_sample_{i}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        # --- B. ラインプロット (次元ごとの波形) ---
        # 16次元を 4x4 または 縦に並べて表示
        fig, axes = plt.subplots(z_dim, 1, figsize=(10, 2 * z_dim), sharex=True)
        
        # x軸（時間）
        t = np.arange(seq_len)
        
        for dim in range(z_dim):
            ax = axes[dim]
            val = z_seq[:, dim]
            ax.plot(t, val, label=f"Dim {dim}", color="tab:blue")
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
            ax.set_ylabel(f"dim {dim}", rotation=0, labelpad=20)
            ax.grid(True, alpha=0.3)
            
            # 背景色で「滞在」と「移動」を区別するアイデア
            # (値が一定なら滞在、変動すれば移動、のように見えるか確認)
            
        plt.xlabel("Time Step")
        plt.suptitle(f"Latent Dynamics by Dimension (Sample {i})", y=1.00)
        plt.tight_layout()
        
        save_path_line = local_stamp(f"z_lines_sample_{i}.png")
        plt.savefig(save_path_line, dpi=300)
        plt.close()
        
        print(f"Saved 16-dim visualization for sample {i}")

def visualize_eigen_projection(model, z_history_list, routes_list, run_id):
    """
    zの軌跡を、Aの固有ベクトル空間に射影して可視化する。
    これにより、「減衰モード」や「振動モード」ごとの寄与度を確認できる。
    """
    import seaborn as sns
    
    print("\n--- Visualizing Eigen-Projections ---")
    
    # 1. A行列の固有分解
    A_np = model.A.detach().cpu().numpy()
    eigenvalues, eigenvectors = np.linalg.eig(A_np)
    
    # 逆行列 V^{-1} を計算 (これが射影行列になる)
    try:
        V_inv = np.linalg.inv(eigenvectors)
    except np.linalg.LinAlgError:
        print("Matrix A is singular/defective. Cannot project.")
        return

    # 固有値の絶対値（|λ|）で降順ソートするためのインデックス
    # |λ|が大きい（1に近い）＝「なかなか減衰しない（記憶維持）モード」
    # |λ|が小さい（0に近い）＝「すぐに消える（短期記憶）モード」
    sort_idx = np.argsort(np.abs(eigenvalues))[::-1]
    
    sorted_evals = eigenvalues[sort_idx]
    sorted_V_inv = V_inv[sort_idx, :] # 行を並べ替え

    # 全サンプル可視化
    num_samples = min(len(z_history_list), 8)
    
    local_dir = os.path.join(out_dir, "z_eigen")
    os.makedirs(local_dir, exist_ok=True)
    def local_stamp(name):
        return os.path.join(local_dir, name) # ★ローカルスタンプ定義


    for i in range(num_samples):
        # [SeqLen, z_dim]
        z_seq = np.array(z_history_list[i]) 
        
        # 2. 射影変換: z_tilde = (V^{-1} @ z.T).T -> z @ (V^{-1}).T
        # 結果は複素数になる
        z_projected_complex = z_seq @ sorted_V_inv.T
        
        # # 3. 振幅（絶対値）を取る
        # z_projected_abs = np.abs(z_projected_complex) #np.realで実部になる
        route = routes_list[i]
        generated_tokens = route[1:] # z[0]はroute[2]に対応
        disp_tokens = generated_tokens[:len(z_seq)]        
        
        # ★ 修正: RealとImagの2回ループして描画
        for component_name, z_data in [("Real", z_projected_complex.real), ("Imag", z_projected_complex.imag)]:
            
            plt.figure(figsize=(14, 8))
            
            # ヒートマップ
            sns.heatmap(z_data.T, cmap="RdBu_r", center=0, cbar=True)
            
            plt.title(f"Latent Eigen-Mode Projection ({component_name} Part) - Sample {i}")
            plt.xlabel("Time Step")
            plt.ylabel("Eigen Mode")
            
            ytick_labels = [f"|λ|={np.abs(lam):.2f}" for lam in sorted_evals]
            plt.yticks(np.arange(len(ytick_labels)) + 0.5, ytick_labels, rotation=0, fontsize=10)
            
            if len(route) < 100:
                ax = plt.gca()
                ax2 = ax.twiny()
                ax2.set_xlim(ax.get_xlim())
                ax2.set_xticks(np.arange(len(z_seq)) + 0.5)
                ax2.set_xticklabels(disp_tokens, rotation=90, fontsize=8)      
            
            # ファイル名に _real / _imag を付与
            save_path = local_stamp(f"z_eigen_proj_sample_{i}_{component_name.lower()}.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
        print(f"Saved eigen-projection for sample {i}")

# =========================================================
#  5. 入力強制力(Forcing)の解析ロジック (新規追加)
# =========================================================

def analyze_input_forcing(model, u_history_list, routes_list, start_nodes, run_id):
    """
    制御入力 term B*u(t) が、潜在空間zおよび固有モードにどう影響しているかを解析する。
    1. Latent Space Forcing: F = B @ u
    2. Eigen Space Forcing:  F_tilde = V^{-1} @ F
    """
    import seaborn as sns
    import pandas as pd
    
    print("\n--- Analyzing Input Forcing Dynamics (Bu) ---")
    
    # --- 準備: 行列Bと固有モード分解 ---
    B_np = model.B.detach().cpu().numpy() # [z_dim, u_dim]
    A_np = model.A.detach().cpu().numpy() # [z_dim, z_dim]
    
    # 固有値分解 A = V Λ V^{-1}
    eigenvalues, eigenvectors = np.linalg.eig(A_np)
    
    try:
        V_inv = np.linalg.inv(eigenvectors) # [z_dim, z_dim]
    except np.linalg.LinAlgError:
        print("Matrix A is singular/defective. Cannot project to eigen-space.")
        return

    # 固有値の絶対値でソート (可視化の順序を合わせるため)
    sort_idx = np.argsort(np.abs(eigenvalues))[::-1]
    sorted_evals = eigenvalues[sort_idx]
    sorted_V_inv = V_inv[sort_idx, :] # 行を並べ替え

    # 解析対象数(全8サンプルとする)
    num_samples = min(len(u_history_list), 8)

    local_dir = os.path.join(out_dir, "Bu_history")
    os.makedirs(local_dir, exist_ok=True)
    def local_stamp(name):
        return os.path.join(local_dir, name) # ★ローカルスタンプ定義

    
    for i in range(num_samples):
        # u_seq: [SeqLen, u_dim]
        u_seq = np.array(u_history_list[i])
        route = routes_list[i]
        seq_len = len(u_seq)

        generated_tokens = route[1:]
        disp_tokens = generated_tokens[:seq_len]

        # --- 1. Latent Space Forcing (F = B u) ---
        # (u_dimが一致するように転置して計算)
        # F: [z_dim, SeqLen]
        F_latent = B_np @ u_seq.T 
        
        # --- 2. Eigen Space Forcing (F_tilde = V^{-1} B u) ---
        # F_tilde: [z_dim, SeqLen] (Complex)
        F_eigen_complex = sorted_V_inv @ F_latent
        # F_eigen_abs = np.abs(F_eigen_complex) # 振幅を取る．np.realで実部になる
        
        # --- 可視化 A: Latent Space Forcing (元のz次元への寄与) ---
        plt.figure(figsize=(14, 6))
        # ヒートマップ (行: 次元, 列: 時間)
        sns.heatmap(F_latent, cmap="RdBu_r", center=0, cbar=True)
        plt.title(f"Latent Forcing (Bu) Heatmap - Sample {i}\n(Direct impact on each z-dimension)")
        plt.xlabel("Time Step")
        plt.ylabel("Latent Dimension Index")
        
        # ルート情報表示
        if seq_len < 60:
            ax = plt.gca()
            ax.set_xticks(np.arange(seq_len) + 0.5)
            ax.set_xticklabels(route[:seq_len], rotation=0, fontsize=10)

        save_path_lat = local_stamp(f"forcing_latent_sample_{i}.png")
        plt.savefig(save_path_lat, dpi=300)
        plt.close()
        
        # --- 可視化 B: Eigen Space Forcing (固有モードへの寄与) ---
        for component_name, F_data in [("Real", F_eigen_complex.real), ("Imag", F_eigen_complex.imag)]: #実部・虚部
            
            plt.figure(figsize=(14, 8))
            sns.heatmap(F_data, cmap="RdBu_r", center=0, cbar=True)
            
            plt.title(f"Eigen-Mode Forcing ({component_name} Part) - Sample {i}")
            plt.xlabel("Time Step")
            plt.ylabel("Eigen Mode")
            
            ytick_labels = [f"|λ|={np.abs(lam):.2f}" for lam in sorted_evals]
            plt.yticks(np.arange(len(ytick_labels)) + 0.5, ytick_labels, rotation=0, fontsize=10)
            
            if seq_len < 60:
                ax = plt.gca()
                ax.set_xticks(np.arange(seq_len) + 0.5)
                ax.set_xticklabels(disp_tokens, rotation=90, fontsize=10)

            save_path = local_stamp(f"forcing_eigen_sample_{i}_{component_name.lower()}.png")
            plt.savefig(save_path, dpi=300)
            plt.close()        

        # --- CSV出力 (詳細分析用: Real & Imag) ---
        # 実部と虚部の両方をDataFrame化
        df_forcing_real = pd.DataFrame(F_eigen_complex.real.T, 
                                     columns=[f"Mode_{k}_Real_|lam|={np.abs(l):.2f}" for k, l in enumerate(sorted_evals)])
        df_forcing_imag = pd.DataFrame(F_eigen_complex.imag.T, 
                                     columns=[f"Mode_{k}_Imag_|lam|={np.abs(l):.2f}" for k, l in enumerate(sorted_evals)])
        
        # 横に結合 (TimeStepをインデックスとして共有)
        df_forcing = pd.concat([df_forcing_real, df_forcing_imag], axis=1)

        # CSVにもトークン情報を入れる (列の先頭に挿入)
        if len(disp_tokens) == len(df_forcing):
            df_forcing.insert(0, "Target_Token", disp_tokens)

        csv_path = local_stamp(f"forcing_eigen_values_sample_{i}.csv")
        df_forcing.to_csv(csv_path, index_label="TimeStep")

def analyze_end_token_trigger(model, network, z_history_list, routes_list, start_nodes, run_id):
    """
    「なぜEndトークンが出たのか？」を解析する。
    1. Output層の重みを取得し、<e>トークンに対する各次元の寄与度を計算。
    2. zの時系列データと、Endスコアへの寄与度をCSV/画像で出力。
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    print("\n--- Analyzing End Token Triggers ---")
    
    local_dir = os.path.join(out_dir, "end_trigger")
    os.makedirs(local_dir, exist_ok=True)
    def local_stamp(name):
        return os.path.join(local_dir, name) # ★ローカルスタンプ定義

    # 1. Output層の重みを取得
    W_out = model.to_logits.weight.detach().cpu().numpy() # [vocab_size, z_dim]
    b_out = model.to_logits.bias.detach().cpu().numpy()   # [vocab_size]
    
    # EndトークンIDの特定 (Tokenization定義準拠: <e> = N + 1)
    end_token_id = network.N + 1
    w_end = W_out[end_token_id] # [16]
    
    num_samples = min(len(z_history_list), 8)
    
    for i in range(num_samples):
        z_seq = np.array(z_history_list[i]) # [T, 16]
        route = routes_list[i]              # [<b>, Start, Token1, Token2, ..., <e>]
        
        # 配列長チェック
        if len(z_seq) == 0: continue

        # --- 寄与度分解 (Contribution Analysis) ---
        # Score = w_end . z_t + bias
        # Contribution = w_end * z_t (要素ごとの積)
        # shape: [T, 16] * [16] -> [T, 16] (Broadcasting)
        contributions = z_seq * w_end 
        total_score = contributions.sum(axis=1) + b_out[end_token_id]
        
        # --- CSV出力データの作成 ---
        # z[t] は route[t+2] (次のトークン) を予測するために使われた
        # 例: route = [<b>, Start, A, <e>]
        # z[0] -> A を予測
        # z[1] -> <e> を予測
        # なので、zの長さ分だけ、routeの「2番目以降」を取得すると対応が取れる
        generated_tokens = route[2:] 
        
        # 長さ合わせ (万が一ズレている場合の安全策)
        min_len = min(len(z_seq), len(generated_tokens))
        valid_z = z_seq[:min_len]
        valid_contrib = contributions[:min_len]
        valid_tokens = generated_tokens[:min_len]
        valid_score = total_score[:min_len]

        df_z = pd.DataFrame(valid_z, columns=[f"z_{d}" for d in range(16)])
        df_contrib = pd.DataFrame(valid_contrib, columns=[f"Contrib_{d}" for d in range(16)])
        
        df_merged = pd.concat([df_z, df_contrib], axis=1)
        df_merged["End_Logit_Score"] = valid_score
        df_merged["Generated_Token"] = valid_tokens
        
        # 実際に <e> で終わったか確認
        is_ended_properly = (valid_tokens[-1] == end_token_id) if len(valid_tokens) > 0 else False
        
        csv_path = local_stamp(f"z_end_analysis_sample_{i}.csv")
        df_merged.to_csv(csv_path, index_label="TimeStep")
        
        # --- 可視化: 最後の瞬間の寄与度 ---
        # 終了した場合のみ、または打ち切りでも「最後の瞬間の気持ち」を表示
        last_step_contrib = valid_contrib[-1]
        
        plt.figure(figsize=(10, 6))
        colors = ['red' if x > 0 else 'blue' for x in last_step_contrib]
        plt.bar(range(len(last_step_contrib)), last_step_contrib, color=colors)
        plt.axhline(0, color='black', linewidth=0.8)
        plt.xlabel("Latent Dimension")
        plt.ylabel(f"Contribution to End-Token Logit")
        
        status_str = "Ended with <e>" if is_ended_properly else "Force Terminated (Max Len)"
        plt.title(f"End Trigger Analysis (Sample {i}) - {status_str}\nPositive bars (Red) = Pushing for End")
        plt.xticks(range(len(last_step_contrib)))
        plt.grid(axis='y', alpha=0.3)
        
        save_path = local_stamp(f"end_trigger_bar_sample_{i}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        # --- 可視化: 時系列でのEndスコアの変化 ---
        # Endスコアに最も寄与している上位3次元の推移
        top3_dims = np.argsort(np.abs(last_step_contrib))[-3:][::-1]
        
        plt.figure(figsize=(12, 6))
        plt.plot(valid_score, label="Total End Logit", color="black", linewidth=2, linestyle="--")
        
        for dim in top3_dims:
            w_val = w_end[dim]
            plt.plot(valid_contrib[:, dim], label=f"Dim {dim} (weight={w_val:.2f})")
            
        plt.xlabel("Time Step")
        plt.ylabel("Logit Contribution")
        plt.title(f"Evolution of 'End' Signal (Sample {i})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path_ts = local_stamp(f"end_signal_evolution_sample_{i}.png")
        plt.savefig(save_path_ts, dpi=300)
        plt.close()
        
        print(f"Saved End-Trigger analysis for sample {i}")


def analyze_end_token_trigger_eigen(model, network, z_history_list, routes_list, start_nodes, run_id):
    """
    【固有モード空間版】なぜEndトークンが出たのか？
    Output層の重みをAの固有ベクトル空間に射影し、
    「どの固有モード(長期記憶/短期振動など)がEnd出力を駆動したか」を解析する。
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    print("\n--- Analyzing End Token Triggers (Eigen-Space) ---")

    local_dir = os.path.join(out_dir, "end_trigger_eigen")
    os.makedirs(local_dir, exist_ok=True)
    def local_stamp(name):
        return os.path.join(local_dir, name) # ★ローカルスタンプ定義

    # 1. 固有値分解の準備 (A = V Λ V^-1)
    A_np = model.A.detach().cpu().numpy()
    eigenvalues, eigenvectors = np.linalg.eig(A_np)
    
    # 固有ベクトル行列 V (columns are eigenvectors)
    V = eigenvectors
    
    try:
        V_inv = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        print("Matrix A is singular. Cannot analyze in eigen-space.")
        return

    # ソート順の定義 (|λ|の大きい順)
    sort_idx = np.argsort(np.abs(eigenvalues))[::-1]
    sorted_evals = eigenvalues[sort_idx]
    
    # V, V_inv をソート順に並べ替え
    # V は「列」が固有ベクトルなので、列を並べ替える
    V_sorted = V[:, sort_idx]
    # V_inv は「行」が固有ベクトル(双対)に対応するので、行を並べ替える
    V_inv_sorted = V_inv[sort_idx, :]

    # 2. Output層の重みを取得 & 固有空間へ射影
    # Logit = z . w_end + b
    # z = z_eigen @ V.T (z_eigenは係数ベクトル)
    # Logit = (z_eigen @ V.T) . w_end = z_eigen . (V.T @ w_end)
    # なので、固有空間での重み w_eigen = V.T @ w_end
    
    W_out = model.to_logits.weight.detach().cpu().numpy()
    end_token_id = network.N + 1
    w_end_real = W_out[end_token_id] # [16]
    
    # 重みの射影 (複素数になる)
    w_end_eigen = V_sorted.T @ w_end_real # [16]

    num_samples = min(len(z_history_list), 8)

    for i in range(num_samples):
        z_seq = np.array(z_history_list[i]) # [T, 16] (Real)
        route = routes_list[i]
        
        if len(z_seq) == 0: continue

        # --- A. 状態zを固有空間へ変換 ---
        # z_eigen = z @ V_inv.T
        z_eigen_seq = z_seq @ V_inv_sorted.T # [T, 16] (Complex)

        # --- B. 寄与度計算 (Eigen-Space) ---
        # Contribution[t, k] = z_eigen[t, k] * w_eigen[k]
        # 結果は複素数だが、合計値(Logit)は実数になるはずなので、
        # 解釈のために「実部(Real Part)」をとるのが物理的に正しい。
        # (複素共役ペアの虚部は打ち消し合うため)
        
        contrib_eigen = z_eigen_seq * w_end_eigen # [T, 16] (Complex)
        contrib_eigen_real = contrib_eigen.real   # [T, 16] (Real)
        
        # 検算: これを合計したら元のEndスコアと一致するか？
        # total_score_eigen = contrib_eigen_real.sum(axis=1) + b_out[end_token_id]
        
        # --- C. 可視化 ---
        # 1. 最後の瞬間の寄与度 (棒グラフ)
        last_step_contrib = contrib_eigen_real[-1]
        
        plt.figure(figsize=(12, 6))
        
        # 固有値ラベルを作成
        x_labels = [f"|λ|={np.abs(e):.2f}" for e in sorted_evals]
        
        colors = ['red' if x > 0 else 'blue' for x in last_step_contrib]
        plt.bar(range(16), last_step_contrib, color=colors)
        plt.axhline(0, color='black', linewidth=0.8)
        
        plt.xlabel("Eigen Mode (Sorted by |λ|)")
        plt.ylabel("Contribution to End-Token Logit (Real Part)")
        plt.title(f"Eigen-Modal Analysis of Trip End (Sample {i})")
        
        plt.xticks(range(16), x_labels, rotation=45, fontsize=8)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        save_path = local_stamp(f"end_trigger_eigen_bar_sample_{i}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

        # 2. CSV出力
        # どのモードがどう効いたかを詳細保存
        df_contrib = pd.DataFrame(contrib_eigen_real, columns=[f"Mode_{k}_|lam|={np.abs(l):.2f}" for k, l in enumerate(sorted_evals)])
        # 元のトークン列も付与
        generated_tokens = route[2:]
        min_len = min(len(df_contrib), len(generated_tokens))
        df_contrib = df_contrib.iloc[:min_len]
        df_contrib["Generated_Token"] = generated_tokens[:min_len]
        
        csv_path = local_stamp(f"end_trigger_eigen_values_sample_{i}.csv")
        df_contrib.to_csv(csv_path, index_label="TimeStep")

        print(f"Saved Eigen-End-Trigger analysis for sample {i}")


def analyze_sequence_drivers_eigen(model, network, z_history_list, routes_list, start_nodes, run_id):
    """
    【時系列・競合分析】
    1. 各ステップで「実際に選ばれたトークン」に対し、どの固有モードが寄与したかを算出。
    2. 「選ばれたトークンのスコア」vs「Endトークンのスコア」の推移を比較。
    3. End直前のトークンとEndトークンの寄与度構成を比較。
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    print("\n--- Analyzing Sequence Drivers (Eigen-Space) ---")

    local_dir = os.path.join(out_dir, "score_history")
    os.makedirs(local_dir, exist_ok=True)
    def local_stamp(name):
        return os.path.join(local_dir, name) # ★ローカルスタンプ定義

    # 1. 固有値分解 & 準備
    A_np = model.A.detach().cpu().numpy()
    eigenvalues, eigenvectors = np.linalg.eig(A_np)
    V = eigenvectors
    try:
        V_inv = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        print("Matrix A is singular. Skipping.")
        return

    # ソート (|λ|降順)
    sort_idx = np.argsort(np.abs(eigenvalues))[::-1]
    sorted_evals = eigenvalues[sort_idx]
    V_sorted = V[:, sort_idx]
    V_inv_sorted = V_inv[sort_idx, :] # [z_dim, z_dim]

    # 出力層重み
    W_out = model.to_logits.weight.detach().cpu().numpy() # [vocab, z_dim]
    b_out = model.to_logits.bias.detach().cpu().numpy()   # [vocab]
    
    end_token_id = network.N + 1
    w_end_real = W_out[end_token_id]
    w_end_eigen = V_sorted.T @ w_end_real # End用の重み(固有空間)

    num_samples = min(len(z_history_list), 8)
    
    # カラーパレット (16色)
    colors_16 = plt.cm.tab20(np.linspace(0, 1, 16))

    for i in range(num_samples):
        z_seq = np.array(z_history_list[i]) # [T, 16]
        route = routes_list[i]
        
        # zに対応するトークン（routeの2番目以降）
        # z[0] は route[2] を予測した
        generated_tokens = route[2:]
        min_len = min(len(z_seq), len(generated_tokens))
        
        if min_len < 2: continue # 短すぎる場合はスキップ

        # スライス
        z_seq = z_seq[:min_len]
        tokens = generated_tokens[:min_len]
        
        # --- A. 時系列データの作成 ---
        # 各ステップについて計算
        # 1. Chosen Tokenへの寄与 (16次元)
        # 2. End Tokenへの寄与 (16次元, 比較用)
        # 3. Total Logits (Chosen vs End)
        
        z_eigen_seq = z_seq @ V_inv_sorted.T # [T, 16] (Complex)
        
        contrib_history = [] # Chosen Tokenへの寄与
        score_diff_history = [] # Chosen Score - End Score
        
        for t in range(min_len):
            token_id = tokens[t]
            z_eig_t = z_eigen_seq[t]
            
            # 選ばれたトークンの重み
            w_chosen_real = W_out[token_id]
            w_chosen_eigen = V_sorted.T @ w_chosen_real
            
            # 寄与度計算 (Real Part)
            contrib_chosen = (z_eig_t * w_chosen_eigen).real
            contrib_end    = (z_eig_t * w_end_eigen).real
            
            score_chosen = contrib_chosen.sum() + b_out[token_id]
            score_end    = contrib_end.sum() + b_out[end_token_id]
            
            contrib_history.append(contrib_chosen)
            score_diff_history.append((score_chosen, score_end))
            
        contrib_history = np.array(contrib_history) # [T, 16]
        score_diff_history = np.array(score_diff_history) # [T, 2]
        
        # --- CSV出力 ---
        col_names = [f"Mode_{k}_|lam|={np.abs(l):.2f}" for k, l in enumerate(sorted_evals)]
        df_contrib = pd.DataFrame(contrib_history, columns=col_names)
        df_contrib["Chosen_Token"] = tokens
        df_contrib["Chosen_Logit"] = score_diff_history[:, 0]
        df_contrib["End_Logit"]    = score_diff_history[:, 1]
        
        csv_path = local_stamp(f"drivers_eigen_sample_{i}.csv")
        df_contrib.to_csv(csv_path, index_label="TimeStep")
        
        # --- 可視化 1: 折れ線グラフ (16本) ---
        plt.figure(figsize=(14, 8))
        time_steps = np.arange(min_len)
        
        # 16本の線を描画
        for dim in range(16):
            label_str = f"Mode {dim} (|λ|={np.abs(sorted_evals[dim]):.2f})"
            # 視認性のため、絶対値の平均が大きい上位5つ以外は薄くする等の工夫も可
            # ここでは要望通り全描画
            plt.plot(time_steps, contrib_history[:, dim], label=label_str, color=colors_16[dim], linewidth=1.5, alpha=0.8)
            
        # イベントマーカー（移動した瞬間など）
        # トークンが変わったタイミングに縦線
        changes = np.where(np.array(tokens)[:-1] != np.array(tokens)[1:])[0] + 1
        for c in changes:
            plt.axvline(x=c, color='gray', linestyle=':', alpha=0.5)

        plt.title(f"Contribution to CHOSEN Token Logit by Eigen-Mode (Sample {i})")
        plt.xlabel("Time Step")
        plt.ylabel("Logit Contribution (Real)")
        # 凡例は外に出す
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path_line = local_stamp(f"drivers_line_sample_{i}.png")
        plt.savefig(save_path_line, dpi=300)
        plt.close()

        # --- 可視化 2: スコア競合 (Winner vs End) ---
        plt.figure(figsize=(12, 5))
        plt.plot(time_steps, score_diff_history[:, 0], label="Chosen Token Score", color="blue", linewidth=2)
        plt.plot(time_steps, score_diff_history[:, 1], label="End Token Score", color="red", linestyle="--", linewidth=2)
        
        plt.title(f"Competition: Chosen Token vs End Token (Sample {i})")
        plt.xlabel("Time Step")
        plt.ylabel("Total Logit")
        plt.legend()
        plt.grid(True)
        
        save_path_comp = local_stamp(f"drivers_competition_sample_{i}.png")
        plt.savefig(save_path_comp, dpi=300)
        plt.close()

        # --- 可視化 3: End直前 vs End (棒グラフ比較) ---
        # 最後がEndトークンだった場合のみ有効
        if tokens[-1] == end_token_id and min_len >= 2:
            step_end = min_len - 1
            step_prev = min_len - 2
            
            contrib_at_end = contrib_history[step_end] # これは <e> への寄与
            contrib_at_prev = contrib_history[step_prev] # これは 直前のトークン への寄与
            
            # トークン名
            token_prev_id = tokens[step_prev]
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            x_labels = [f"|λ|={np.abs(e):.2f}" for e in sorted_evals]
            bar_x = range(16)
            
            # 上段: 直前 (T-1)
            colors_prev = ['red' if x > 0 else 'blue' for x in contrib_at_prev]
            axes[0].bar(bar_x, contrib_at_prev, color=colors_prev)
            axes[0].axhline(0, color='black', lw=0.8)
            axes[0].set_title(f"Step {step_prev}: Winner = ID {token_prev_id} (Why NOT End?)")
            axes[0].set_ylabel("Contribution to Winner")
            axes[0].grid(axis='y', alpha=0.3)
            
            # 下段: 終了 (T)
            colors_end = ['red' if x > 0 else 'blue' for x in contrib_at_end]
            axes[1].bar(bar_x, contrib_at_end, color=colors_end)
            axes[1].axhline(0, color='black', lw=0.8)
            axes[1].set_title(f"Step {step_end}: Winner = End Token (Why End?)")
            axes[1].set_ylabel("Contribution to End")
            axes[1].set_xticks(bar_x)
            axes[1].set_xticklabels(x_labels, rotation=45, fontsize=8)
            axes[1].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            save_path_bar = local_stamp(f"drivers_compare_last_sample_{i}.png")
            plt.savefig(save_path_bar, dpi=300)
            plt.close()
            
        print(f"Saved Driver analysis for sample {i}")


def analyze_selection_probability(probs_history_list, routes_list, start_nodes, run_id):
    """
    各タイムステップでのトークン選択確率をヒートマップで可視化する。
    「確率が高いのに選ばれなかった」vs「確率が低いのに選ばれた」を判別可能にする。
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import pandas as pd
    
    print("\n--- Analyzing Token Selection Probabilities ---")

    num_samples = min(len(probs_history_list), 8)
    
    local_dir = os.path.join(out_dir, "selection_prob")
    os.makedirs(local_dir, exist_ok=True)
    def local_stamp(name):
        return os.path.join(local_dir, name) # ★ローカルスタンプ定義

    for i in range(num_samples):
        probs_seq = np.array(probs_history_list[i]) # [SeqLen, VocabSize]
        route = routes_list[i]
        
        # probs_seq[t] は route[t+2] (generated_tokens[t]) を選択する際の確率分布
        generated_tokens = route[2:]
        min_len = min(len(probs_seq), len(generated_tokens))
        
        if min_len == 0: continue

        # データ切り出し
        probs_seq = probs_seq[:min_len] # [T, V]
        tokens = generated_tokens[:min_len]
        time_steps = np.arange(min_len)

        # 転置して [Vocab, Time] にする (ヒートマップ用)
        probs_matrix = probs_seq.T 
        
        # 実際に選ばれたトークンの確率を取得
        chosen_probs = []
        for t, token_id in enumerate(tokens):
            p = probs_seq[t, token_id]
            chosen_probs.append(p)
        
        # --- 可視化作成 (2段構成) ---
        fig, axes = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        
        # 上段: 確率ヒートマップ
        # Y軸: トークンID (Vocab), X軸: 時間
        # 全Vocabだと縦に長過ぎる場合があるので、確率が0でない行だけに絞ることも可能ですが、
        # 全体感を見るために一旦全部出します（Vocab数が100程度ならOK）
        sns.heatmap(probs_matrix, cmap="Reds", ax=axes[0], cbar_kws={'label': 'Probability'})
        
        # 実際に選ばれたパスをプロット (Xマーク)
        # x座標: time + 0.5 (セルの真ん中), y座標: token_id + 0.5
        time_indices = np.arange(min_len) + 0.5
        token_indices = np.array(tokens) + 0.5
        axes[0].scatter(time_indices, token_indices, color='blue', marker='x', s=30, label='Chosen Path')
        
        axes[0].set_title(f"Next Token Probability Heatmap (Sample {i})")
        axes[0].set_ylabel("Token ID")
        axes[0].legend(loc='upper right')
        axes[0].tick_params(labelbottom=True)

        # 下段: 確信度の推移 (選ばれたトークンの確率 vs 最大確率)
        max_probs = probs_seq.max(axis=1) # そのステップでの最大確率（トップの候補）
        
        axes[1].plot(time_steps, max_probs, color='gray', linestyle='--', label='Max Probability (Top Candidate)', alpha=0.7)
        axes[1].plot(time_steps, chosen_probs, color='blue', marker='o', label='Chosen Token Probability', linewidth=1.5)
        
        # 乖離がある場所（低確率なのに選ばれた場所）を強調
        # 例えば Maxとの差が0.3以上ある場合など
        for t in range(min_len):
            if max_probs[t] - chosen_probs[t] > 0.3:
                axes[1].annotate('Surprise!', (t, chosen_probs[t]), xytext=(t, chosen_probs[t]+0.1), 
                                 arrowprops=dict(facecolor='red', shrink=0.05), fontsize=8, color='red')

        axes[1].set_title("Model Confidence vs Actual Choice")
        axes[1].set_ylabel("Probability")
        axes[1].set_xlabel("Time Step")
        axes[1].set_ylim(0, 1.05)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # X軸ラベルにトークン名を表示
        if min_len < 60:
            axes[1].set_xticks(np.arange(min_len) + 0.5)
            axes[1].set_xticklabels(tokens, rotation=90, fontsize=8)

        plt.tight_layout()
        
        # 保存
        save_path = local_stamp(f"prob_heatmap_{i}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        # CSV保存 (各ステップの選択確率とエントロピーなど)
        # エントロピー計算: -sum(p * log(p))
        entropy = -np.sum(probs_seq * np.log(probs_seq + 1e-9), axis=1)
        
        df_probs = pd.DataFrame({
            "TimeStep": range(min_len),
            "Chosen_Token": tokens,
            "Chosen_Prob": chosen_probs,
            "Max_Prob": max_probs,
            "Entropy": entropy,
            "Is_Greedy": (np.array(chosen_probs) == np.array(max_probs)) # 最尤選択だったか？
        })
        csv_path = local_stamp(f"prob_analysis_{i}.csv")
        df_probs.to_csv(csv_path, index=False)
        
        print(f"Saved Probability analysis for sample {i}")


def analyze_token_weights_in_eigen_space(model, network, run_id):
    """
    出力層の重みベクトル W_out を固有空間に射影し、
    「各トークンがどの固有モード（長期記憶/短期振動）を好むか」を解析する。
    特にEndトークンがどのモードに反応するかを確認する。
    """
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    
    print("\n--- Analyzing Token Weights in Eigen-Space ---")
    
    # local_dir = os.path.join(out_dir, "weight_analysis")
    # os.makedirs(local_dir, exist_ok=True)
    # def local_stamp(name): return os.path.join(local_dir, name)

    # 1. 固有値分解
    A_np = model.A.detach().cpu().numpy()
    eigenvalues, eigenvectors = np.linalg.eig(A_np)
    V = eigenvectors # [z_dim, z_dim]
    
    # ソート
    sort_idx = np.argsort(np.abs(eigenvalues))[::-1]
    sorted_evals = eigenvalues[sort_idx]
    V_sorted = V[:, sort_idx]

    # 2. 重みの取得と射影
    # Linear(z_dim, vocab_size) -> weight shape is [vocab, z_dim]
    W_out = model.to_logits.weight.detach().cpu().numpy()
    
    # 射影: W_tilde = W @ V (各行 w_k に対して、w_tilde_k = V^T w_k なので、行列積としては W @ V でOK)
    # 検証: Logit = z @ w.T = (z_tilde @ V.T) @ w.T = z_tilde @ (w @ V).T
    # つまり 新しい重み行列 W_eigen = W @ V
    # 形状: [vocab, z_dim]
    W_eigen = W_out @ V_sorted
    
    # 複素数なので振幅（絶対値）と実部を見る
    # W_eigen_abs = np.abs(W_eigen)
    # W_eigen_real = W_eigen.real

    # 3. 可視化 (ヒートマップ)
    # 全トークンだと多いので、Endトークン周辺や主要なノードに絞ることも考えられますが、
    # ここでは全体像と、Endトークンの特写を行います。
    
    end_token_id = network.N + 1
    x_labels = [f"|λ|={np.abs(e):.2f}" for e in sorted_evals]
    
    # ★ 修正: RealとImagのループ
    for component_name, W_data in [("Real", W_eigen.real), ("Imag", W_eigen.imag)]:
        
        # A. 全体ヒートマップ
        plt.figure(figsize=(14, 10))
        sns.heatmap(W_data, cmap="coolwarm", center=0, cbar_kws={'label': f'Weight ({component_name})'})
        plt.xlabel("Eigen Mode (Sorted by |λ|)")
        plt.ylabel("Token ID")
        plt.title(f"Token Weights projected to Eigen-Space ({component_name} Part)")
        
        plt.xticks(np.arange(16)+0.5, x_labels, rotation=45, fontsize=8)
        
        plt.axhline(end_token_id + 0.5, color='yellow', linewidth=2, linestyle='--')
        plt.text(16.5, end_token_id + 0.5, ' <e>', color='black', va='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(stamp(f"weights_eigen_heatmap_{component_name.lower()}.png"), dpi=300)
        plt.close()

        # B. Endトークンの詳細バープロット
        w_end_val = W_data[end_token_id]
        
        plt.figure(figsize=(10, 6))
        colors = ['red' if x > 0 else 'blue' for x in w_end_val]
        plt.bar(range(16), w_end_val, color=colors)
        plt.axhline(0, color='black', linewidth=0.8)
        plt.xlabel("Eigen Mode")
        plt.ylabel(f"Weight Value ({component_name})")
        plt.title(f"Eigen-Space Weights for END Token ({component_name} Part)")
        plt.xticks(range(16), x_labels, rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(stamp(f"weight_end_token_profile_{component_name.lower()}.png"), dpi=300)
        plt.close()

    # 4. CSV出力 (Real & Imag)
    # 実部
    df_weights_real = pd.DataFrame(W_eigen.real, 
                                 columns=[f"Mode_{k}_Real_|lam|={np.abs(l):.2f}" for k, l in enumerate(sorted_evals)])
    # 虚部
    df_weights_imag = pd.DataFrame(W_eigen.imag, 
                                 columns=[f"Mode_{k}_Imag_|lam|={np.abs(l):.2f}" for k, l in enumerate(sorted_evals)])
    
    # 結合
    df_weights = pd.concat([df_weights_real, df_weights_imag], axis=1)
    
    df_weights.index.name = "TokenID"
    
    # 保存 (ファイル名を変更: weights_eigen_real.csv -> weights_eigen_complex.csv とするのがベター)
    df_weights.to_csv(stamp("weights_eigen_complex.csv"))
    
    print(f"Saved Weight analysis (Real/Imag in CSV)")

def analyze_full_token_contribution(model, network, z_history_list, routes_list, start_nodes, run_id):
    """
    【全トークン寄与度分解】
    TimeStep x Token x EigenMode の寄与度を全計算し、CSVとして保存する。
    「滞在中になぜEndトークンの確率が上がったのか？」などの詳細分析用。
    """
    import pandas as pd
    import numpy as np
    import os
    from tokenization import Tokenization

    print("\n--- Analyzing Full Token Contributions (3D Tensor) ---")
    
    # 保存先
    local_dir = os.path.join(out_dir, "full_contribution")
    os.makedirs(local_dir, exist_ok=True)
    def local_stamp(name): 
        return os.path.join(local_dir, name)

    # 1. 固有値分解 & 射影行列の準備
    A_np = model.A.detach().cpu().numpy()
    eigenvalues, eigenvectors = np.linalg.eig(A_np)
    V = eigenvectors
    try:
        V_inv = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        print("Matrix A is singular. Skipping.")
        return

    # ソート (|λ|降順)
    sort_idx = np.argsort(np.abs(eigenvalues))[::-1]
    sorted_evals = eigenvalues[sort_idx]
    V_sorted = V[:, sort_idx]         # [16, 16]
    V_inv_sorted = V_inv[sort_idx, :] # [16, 16]

    # 2. 重み行列の射影
    # W_out: [Vocab, 16]
    W_out = model.to_logits.weight.detach().cpu().numpy()
    b_out = model.to_logits.bias.detach().cpu().numpy()
    
    # 固有空間での重み: W_tilde = W @ V
    W_tilde = W_out @ V_sorted # [Vocab, 16] (Complex)

    # トークン名取得用
    tokenizer = Tokenization(network)
    # IDから文字列への逆引き辞書 (簡易作成)
    id_to_token = {v: k for k, v in tokenizer.SPECIAL_TOKENS.items()}
    
    num_samples = min(len(z_history_list), 8)

    for i in range(num_samples):
        z_seq = np.array(z_history_list[i]) # [T, 16]
        route = routes_list[i]
        
        # zに対応する生成トークン（可視化のラベル用）
        # z[t] は route[t+2] を予測する状態
        generated_tokens = route[2:]
        min_len = min(len(z_seq), len(generated_tokens))
        
        if min_len == 0: continue
        
        # データ切り出し
        z_seq = z_seq[:min_len]
        
        # 3. 状態ベクトルの射影
        # z_tilde = z @ V^{-T}
        z_tilde_seq = z_seq @ V_inv_sorted.T # [T, 16] (Complex)
        
        # 4. 寄与度の計算 (Broadcasting)
        # z_tilde_seq: [T, 1, 16]
        # W_tilde:     [1, Vocab, 16]
        # Result:      [T, Vocab, 16]
        
        # 要素ごとの積をとる（まだ合計しない）
        contribution_tensor = z_tilde_seq[:, np.newaxis, :] * W_tilde[np.newaxis, :, :]
        
        # 実部をとる（これがスコアへの寄与）
        contribution_real = contribution_tensor.real # [T, Vocab, 16]
        
        # 5. DataFrame化 (Long Format)
        # 3次元配列を2次元の表に展開
        T, V, K = contribution_real.shape
        
        # インデックスの作成
        time_idx = np.repeat(np.arange(T), V) # [0, 0, ..., 1, 1, ...]
        vocab_idx = np.tile(np.arange(V), T)  # [0, 1, ..., 0, 1, ...]
        
        # フラット化
        contrib_flat = contribution_real.reshape(T * V, K)
        
        df = pd.DataFrame(contrib_flat, columns=[f"Mode_{k}_|lam|={np.abs(l):.2f}" for k, l in enumerate(sorted_evals)])
        
        df.insert(0, "TimeStep", time_idx)
        df.insert(1, "TokenID", vocab_idx)
        
        # トークン名の付与（読みやすくするため）
        def get_token_str(tid):
            return id_to_token.get(tid, str(tid))
        df.insert(2, "TokenStr", df["TokenID"].apply(get_token_str))
        
        # 合計スコア（検算用: sum(contrib) + bias）
        bias_flat = np.tile(b_out, T)
        df["Sum_Contrib"] = df.iloc[:, 4:].sum(axis=1) # Mode列の合計
        df["Bias"] = bias_flat
        df["Total_Logit"] = df["Sum_Contrib"] + df["Bias"]
        
        # CSV保存
        # サイズが大きくなる可能性があるため、圧縮しても良いが、まずはそのまま
        csv_path = local_stamp(f"full_contrib_sample_{i}.csv")
        df.to_csv(csv_path, index=False)
        
        print(f"Saved Full Contribution Tensor for sample {i} (Shape: {df.shape})")


def main():
    if not os.path.exists(ADJ_PATH):
        print(f"Error: Adjacency matrix not found at {ADJ_PATH}")
        return

    adj_matrix = torch.load(ADJ_PATH, weights_only=True)
    # --- Δ距離(広場)バイアス用: base最短距離行列 ---
    dist_is_directed = False
    if adj_matrix.shape[0] == 38:
        base_N = 19
        base_adj = adj_matrix[:base_N, :base_N]
    else:
        base_adj = adj_matrix
        base_N = int(base_adj.shape[0])
    dist_mat_base = compute_shortest_path_distance_matrix(base_adj, directed=dist_is_directed)
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

    model, config = load_model(model_path, network, dist_mat_base=dist_mat_base, base_N=base_N, dist_is_directed=dist_is_directed)
    
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
    all_u_histories = []
    all_probs_histories = [] # ★追加

    target_start_nodes = [0,1,2,3,4,5,6,7]
    test_agent_id = 0
    print(f"Generating routes from nodes: {target_start_nodes} for Agent ID: {test_agent_id}")
    
    for start_node in target_start_nodes:
        route, z_hist, u_hist, probs_hist = generate_route(model, network, start_node, agent_id=test_agent_id, strategy="sample", temperature=1.0)
        
        all_z_histories.append(z_hist)
        all_routes.append(route)
        all_u_histories.append(u_hist) # ★追加
        all_probs_histories.append(probs_hist) # ★追加

    visualize_trajectory_with_time(all_z_histories, all_routes, target_start_nodes, title="routes")
    visualize_trajectory_3d(all_z_histories, all_routes, target_start_nodes, title="routes")
    
    # 既存の可視化
    visualize_z_all_dimensions(all_z_histories, all_routes, target_start_nodes, run_id)
    visualize_eigen_projection(model, all_z_histories, all_routes, run_id)
    
    # ★新規追加: 入力Forcingの可視化
    analyze_input_forcing(model, all_u_histories, all_routes, target_start_nodes, run_id)
    analyze_end_token_trigger(model, network, all_z_histories, all_routes, target_start_nodes, run_id)
    analyze_end_token_trigger_eigen(model, network, all_z_histories, all_routes, target_start_nodes, run_id)
    analyze_sequence_drivers_eigen(model, network, all_z_histories, all_routes, target_start_nodes, run_id)
    analyze_selection_probability(all_probs_histories, all_routes, target_start_nodes, run_id)
    analyze_token_weights_in_eigen_space(model, network, run_id)
    analyze_full_token_contribution(model, network, all_z_histories, all_routes, target_start_nodes, run_id)

if __name__ == "__main__":
    main()