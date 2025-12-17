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

# 保存先
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"/home/mizutani/projects/RF/runs/comparison_{run_id}"
os.makedirs(out_dir, exist_ok=True)

def stamp(name):
    return os.path.join(out_dir, name)

# =========================================================
#  設定：比較したいモデルのパスを指定してください
# =========================================================
# 学習済みモデルのパスを指定してください
# Noneの場合、自動で最新のものを取得しますが、比較実験時は明示した方が良いです。
# MODEL_PATH = None 
MODEL_PATH = '/home/mizutani/projects/RF/runs/20251217_130327/model_weights_20251217_130327.pth'

ADJ_PATH = '/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 編集距離計算用 (ライブラリ不要版) ---
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
    検証用データセット(Val)を使うのが理想ですが、簡易的に全データまたは一部を使います。
    """
    print("\n=== Evaluating Quantitative Metrics ===")
    
    # データのロード (Trainと同じ手順)
    trip_arrz = np.load('/home/mizutani/projects/RF/data/input_a.npz')
    trip_arr = trip_arrz['route_arr']
    route = torch.from_numpy(trip_arr)
    
    # 簡易的に最初の500サンプル程度で評価 (時間短縮のため)
    # 全データでやるなら num_samples = len(route)
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


def load_model(model_path, network):
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
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
    # (既存のコードそのまま)
    tokenizer = Tokenization(network)
    TOKEN_START = tokenizer.SPECIAL_TOKENS["<b>"]
    TOKEN_END   = tokenizer.SPECIAL_TOKENS["<e>"]
    
    current_seq = [TOKEN_START, start_node_id]
    z_history = [] 
    
    # print(f"Generating route... Start Node: {start_node_id}")
    
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

            probs = torch.nn.functional.softmax(last_logits / temperature, dim=0)
            if strategy == "greedy":
                next_token = torch.argmax(probs).item()
            else:
                next_token = torch.multinomial(probs, num_samples=1).item()

            current_seq.append(next_token)
            if next_token == TOKEN_END:
                break
                
    return current_seq, z_history

def visualize_divergence_2d(z_history_list, start_node):
    # (既存のコードそのまま、保存先をout_dirに変更)
    all_z = np.concatenate(z_history_list, axis=0)
    pca = PCA(n_components=2)
    pca.fit(all_z)
    
    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap('tab10')
    
    for i, z_hist in enumerate(z_history_list):
        z_2d = pca.transform(np.array(z_hist))
        plt.plot(z_2d[:, 0], z_2d[:, 1], marker='.', markersize=3, alpha=0.6, label=f'Trial {i+1}', color=cmap(i))
        plt.scatter(z_2d[0, 0], z_2d[0, 1], c='black', s=100, marker='*')
        plt.scatter(z_2d[-1, 0], z_2d[-1, 1], c=cmap(i), s=80, marker='x')

    plt.title(f"2D Divergence (Node {start_node})")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend(); plt.grid(True)
    plt.savefig(stamp(f"divergence_node{start_node}_2d.png"))
    plt.close()

def visualize_divergence_3d(z_history_list, start_node):
    # (既存のコードそのまま、保存先をout_dirに変更)
    all_z = np.concatenate(z_history_list, axis=0)
    pca = PCA(n_components=3)
    pca.fit(all_z)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap('tab10')
    
    for i, z_hist in enumerate(z_history_list):
        z_3d = pca.transform(np.array(z_hist))
        ax.plot(z_3d[:, 0], z_3d[:, 1], z_3d[:, 2], alpha=0.6, label=f'Trial {i+1}', color=cmap(i))
        ax.scatter(z_3d[0, 0], z_3d[0, 1], z_3d[0, 2], c='black', s=50, marker='*')
        ax.scatter(z_3d[-1, 0], z_3d[-1, 1], z_3d[-1, 2], c=[cmap(i)], s=50, marker='x')

    ax.set_title(f"3D Divergence (Node {start_node})")

    ax.view_init(elev=0, azim=90)
    plt.savefig(stamp(f"divergence_node{start_node}_3d1.png"))

    ax.view_init(elev=30, azim=45)
    plt.savefig(stamp(f"divergence_node{start_node}_3d2.png"))
    plt.close()

def main():
    if not os.path.exists(ADJ_PATH):
        print(f"Error: Adjacency matrix not found.")
        return
    adj_matrix = torch.load(ADJ_PATH, weights_only=True)
    dummy_node_features = torch.zeros((len(adj_matrix), 1))
    network = Network(adj_matrix, dummy_node_features)
    
    # モデルロード
    model_path = MODEL_PATH
    if model_path is None:
        search_dir = "/home/mizutani/projects/RF/runs/"
        pth_files = glob.glob(os.path.join(search_dir, "*", "*.pth"))
        if not pth_files: return
        model_path = max(pth_files, key=os.path.getctime)
    
    model, config = load_model(model_path, network)
    
    # ★ 1. 定量評価の実行 (Accuracy, EditDistance)
    acc, edit_dist = evaluate_metrics(model, network, device)
    
    # ★ 2. 定性評価 (可視化)
    target_start_nodes = [0, 1, 3] 
    N_TRIALS = 5

    print(f"Generating visualizations in: {out_dir}")
    for start_node in target_start_nodes:
        trials_z_history = []
        for i in range(N_TRIALS):
            route, z_hist = generate_route(model, network, start_node, strategy="sample", temperature=1.0)
            trials_z_history.append(z_hist)
        
        visualize_divergence_2d(trials_z_history, start_node)
        visualize_divergence_3d(trials_z_history, start_node)

if __name__ == "__main__":
    main()