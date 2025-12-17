import torch
import os
import glob
from network import Network
from tokenization import Tokenization
from KP_RF import KoopmanRoutesFormer
from datetime import datetime

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"/home/mizutani/projects/RF/runs/figure"
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

def generate_route(model, network, start_node_id, max_len=50, strategy="greedy", temperature=1.0):
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
    
    # 4. 推論実行 (例: ノード0からスタート)
    for i in range(0,20):
        start_node = i
        generated_route, z_history = generate_route(model, network, start_node, strategy="greedy")    
        
        # 5. 結果表示
        print("-" * 40)
        print(f"Generated Route (Token IDs): {generated_route}")
        print(f"Length: {len(generated_route)}")
        
        # トークンIDをわかりやすく表示 (<b>, <e> などに戻す)
        tokenizer = Tokenization(network)
        readable_route = []
        inv_special_tokens = {v: k for k, v in tokenizer.SPECIAL_TOKENS.items()}
        
        for token in generated_route:
            if token in inv_special_tokens:
                readable_route.append(inv_special_tokens[token])
            else:
                readable_route.append(str(token))
                
        print(f"Readable: {' -> '.join(readable_route)}")
        print("-" * 40)

    # === ▼ ここから可視化コード ▼ ===
        if len(z_history) > 1:
            z_data = np.array(z_history) # shape: [Steps, z_dim]
            
            # PCAで2次元に圧縮
            pca = PCA(n_components=2)
            z_2d = pca.fit_transform(z_data)
            
            plt.figure(figsize=(8, 6))
            
            # 軌跡を描画
            plt.plot(z_2d[:, 0], z_2d[:, 1], marker='o', alpha=0.6, label='Trajectory')
            
            # 開始地点と終了地点を強調
            plt.scatter(z_2d[0, 0], z_2d[0, 1], c='green', s=100, label='Start')
            plt.scatter(z_2d[-1, 0], z_2d[-1, 1], c='red', s=100, label='End')
            
            # 各点に「ノード番号」をラベル表示
            # (generated_routeは <b>, start, next... なのでインデックスに注意)
            route_nodes = generated_route[1:] # <b>を除く
            for i in range(min(len(z_2d), len(route_nodes))):
                plt.text(z_2d[i, 0], z_2d[i, 1], str(route_nodes[i]), fontsize=9)

            plt.title(f"Koopman Latent Dynamics (Start Node: {start_node})")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.legend()
            plt.grid(True)
            
            # 保存
            graph_filename = stamp(f"z_node{start_node}_gr_{run_id}.png")
            plt.savefig(graph_filename)
            print(f"Trajectory plot saved to z_node{start_node}_{run_id}.png")
        # ===============================

if __name__ == "__main__":
    main()