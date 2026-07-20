### 固定ルートをなぞってzの遷移を出す（１パラメータに対してのみ） ###


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
#  設定
# =========================================================
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"/home/mizutani/projects/RF/runs/interpretability_c_{run_id}"
os.makedirs(out_dir, exist_ok=True)

def stamp(name):
    return os.path.join(out_dir, name)

# 比較したいモデルのパスを指定してください
# ★ここにそれぞれのモデルパスを入れる
MODEL_PATH = '/home/mizutani/projects/RF/runs/20251218_020243/model_weights_20251218_020243.pth'

ADJ_PATH = '/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
#  モデルロード関数 (共通)
# =========================================================
def load_model(model_path, network):
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
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
        pad_token_id=config['pad_token_id']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, config

# =========================================================
#  ★新機能: 固定ルートをなぞる関数 (Teacher Forcing)
# =========================================================
def trace_fixed_route(model, route_tensor):
    """
    固定されたルート(route_tensor)を入力し、その時のzの系列を返す
    route_tensor: [SeqLen] (Token IDs)
    """
    with torch.no_grad():
        # バッチ次元追加 [1, SeqLen]
        input_tensor = route_tensor.unsqueeze(0).to(device)
        
        # モデルに入力 (全ステップ一括計算)
        # z_hat: [1, SeqLen, z_dim]
        logits, z_hat, z_pred = model(input_tensor)
        
        # zを取り出す (CPUへ)
        z_seq = z_hat[0].cpu().numpy() # [SeqLen, z_dim]
        
    return z_seq

# =========================================================
#  可視化関数
# =========================================================
def visualize_fixed_trajectory(z_seq, route_ids, title_suffix=""):
    """
    1本のルートに対するzの軌跡を描画する
    """
    # PCAで2次元化
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z_seq)
    
    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap('viridis')
    T = len(z_2d)
    
    # 軌跡（線）
    plt.plot(z_2d[:, 0], z_2d[:, 1], color='gray', alpha=0.5, linewidth=1.5)
    
    # 点（時間グラデーション）
    sc = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=range(T), cmap=cmap, s=50, zorder=5)
    
    # ノード番号の注釈
    for k in range(T):
        node_id = route_ids[k]
        
        # 特殊トークン(paddingなど)は表示しない、あるいは記号にする
        if node_id >= 20: # 20以上は特殊トークンと仮定(環境に合わせて調整)
             label = str(node_id) # 必要なら "PAD" とかに変える
        else:
             label = str(node_id)

        # Startは赤、Endは青、それ以外は黒
        if k == 0:
            plt.text(z_2d[k, 0], z_2d[k, 1], label, fontsize=12, fontweight='bold', color='red')
        elif k == T-1:
            plt.text(z_2d[k, 0], z_2d[k, 1], label, fontsize=12, fontweight='bold', color='blue')
        else:
            plt.text(z_2d[k, 0]+0.02, z_2d[k, 1]+0.02, label, fontsize=9, color='black')

    plt.colorbar(sc, label='Time Step')
    plt.title(f"Latent Trajectory for Fixed Route\n{title_suffix}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    
    save_name = stamp(f"trace_{title_suffix}.png")
    plt.savefig(save_name)
    plt.close()
    print(f"Saved: {save_name}")

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
    if not os.path.exists(model_path): return
    model, config = load_model(model_path, network)
    
    # 3. 評価用ルートの検索
    trip_arrz = np.load('/home/mizutani/projects/RF/data/input_a.npz')
    trip_arr = trip_arrz['route_arr']
    all_routes = torch.from_numpy(trip_arr)
    tokenizer = Tokenization(network)
    
    print(f"Total routes: {len(all_routes)}")
    
    # ランダムに探索（シード固定）
    np.random.seed(44)
    shuffled_indices = np.random.permutation(len(all_routes))
    
    target_indices = []
    print("Searching for routes (Len 8-50)...")
    
    # 特殊トークンのID定義 (ログから特定: <p>=19, <e>=20)
    ID_PAD = 19
    ID_EOS = 20
    ID_START = 21

    for i in shuffled_indices:
        route_raw = all_routes[i]
        tokens = tokenizer.tokenization(route_raw.unsqueeze(0), mode="simple").long()[0]
        
        # 長さ判定
        end_mask = (tokens == ID_PAD) | (tokens == ID_EOS)
        if end_mask.any():
            length = end_mask.nonzero(as_tuple=True)[0][0].item()
        else:
            length = len(tokens)
            
        if 8 <= length <= 50:
            target_indices.append(i)
            print(f"  Found Route {i}: Length = {length}")
        
        if len(target_indices) >= 5:
            break
            
    print(f"Selected indices: {target_indices}")

    # 4. 描画処理 (トリミング修正版)
    print("\nGenerating traces...")
    for idx in target_indices:
        raw_route = all_routes[idx].unsqueeze(0)
        tokens = tokenizer.tokenization(raw_route, mode="simple").long()[0].to(device)
        
        # モデルで z の軌跡を取得
        z_seq = trace_fixed_route(model, tokens)
        
        # === トリミング処理 ===
        
        # 1. Start Token (21) の削除
        # 先頭が21なら削除する
        if tokens[0] == ID_START:
            tokens = tokens[1:]
            z_seq = z_seq[1:]
        
        # 2. End/Pad Token (19, 20) 以降の削除
        # トークン列から 19 か 20 を探す
        end_mask = (tokens == ID_PAD) | (tokens == ID_EOS)
        
        if end_mask.any():
            valid_len = end_mask.nonzero(as_tuple=True)[0][0].item()
            
            # もし <e>(20) だったら、その地点までは描画したいので +1
            # もし <p>(19) だったら、そこは描画しなくていいのでそのまま
            if tokens[valid_len] == ID_EOS:
                valid_len += 1
        else:
            valid_len = len(tokens)
            
        # 安全策: データが空にならないように
        if valid_len < 1: valid_len = 1
            
        # 切り出し
        z_seq_valid = z_seq[:valid_len]
        tokens_valid = tokens[:valid_len].cpu().numpy()
        
        print(f"  Plotting Route {idx}: Valid Length = {valid_len}, Tokens = {tokens_valid}")
        
        # 可視化
        title = f"RouteID_{idx}_Len_{valid_len}"
        visualize_fixed_trajectory(z_seq_valid, tokens_valid, title_suffix=title)

if __name__ == "__main__":
    main()