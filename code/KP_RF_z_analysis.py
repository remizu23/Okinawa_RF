import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import networkx as nx
import random
import io
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from datetime import datetime
import matplotlib.cm as cm

# ★重要: KP_RF.py の読み込み
try:
    from KP_RF import KoopmanRoutesFormer
except ImportError:
    raise ImportError("KP_RF.py not found.")

# =========================================================
# 0. 設定 & 保存先
# =========================================================
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"/home/mizutani/projects/RF/runs/z_analysis_{run_id}"
os.makedirs(out_dir, exist_ok=True)

print(f"=== Latent Space Analysis: {run_id} ===")
print(f"Results will be saved to: {out_dir}")

# ★★★ 要修正: モデルパス ★★★
MODEL_PATH_WITH = "/home/mizutani/projects/RF/runs/20251218_033604/model_weights_20251218_033604.pth"
MODEL_PATH_WITHOUT = "/home/mizutani/projects/RF/runs/20251218_034727/model_weights_20251218_034727.pth"

# =========================================================
# 1. 環境設定 (共通)
# =========================================================
csv_data = """
,0,1,2,3,4,5,6,7,8,9,10,11,13,14,16,18
0,1,1,1,0,1,0,0,0,0,0,0,1,0,0,0,0
1,1,1,1,1,1,1,0,0,0,1,0,0,0,0,0,0
2,1,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0
3,0,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0
4,1,1,0,0,1,1,0,0,1,1,0,1,0,0,0,0
5,0,0,0,1,1,1,1,0,0,1,1,0,0,0,0,0
6,0,0,0,1,0,1,1,1,0,0,0,0,0,0,0,0
7,0,0,1,1,0,0,1,1,0,0,0,0,1,1,0,0
8,0,0,0,0,1,0,0,0,1,1,0,1,0,0,0,0
9,0,1,0,0,1,0,0,0,1,1,1,1,0,0,0,0
10,0,0,0,0,0,1,1,0,0,1,1,0,1,0,0,0
11,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0
13,0,0,0,0,0,0,0,1,0,0,1,0,1,1,0,0
14,0,0,0,0,0,0,0,1,0,0,0,0,1,1,1,0
16,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1
18,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1
"""
df_adj = pd.read_csv(io.StringIO(csv_data), index_col=0)
df_adj.columns = df_adj.columns.astype(int)
G = nx.from_pandas_adjacency(df_adj)
G.remove_edges_from(nx.selfloop_edges(G))

AREA_SHOP1 = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11]
AREA_SHOP2 = [7, 13, 14, 16, 18]
PAD_TOKEN = 19

# =========================================================
# 2. AgentV2 (データ生成用)
# =========================================================
class AgentV2:
    def __init__(self, agent_id, graph, behavior_type):
        self.id = agent_id
        self.graph = graph
        self.type = behavior_type
        self.trajectory = []
        self.finished = False
        
        if self.type == 'through':
            self.start_node = random.choice([0, 3])
            self.goal_node = random.choice([16, 18])
            self.current_node = self.start_node
            self.state = 'WALK'; self.target = self.goal_node; self.phase = None
        elif self.type == 'stopover':
            self.start_node = random.choice([0, 3])
            self.shop_node = random.choice([2, 4, 8, 10])
            self.current_node = self.start_node
            self.state = 'WALK'; self.target = self.shop_node; self.phase = 'GO_TO_SHOP'
        elif self.type == 'wander':
            self.current_node = random.choice(AREA_SHOP1)
            self.state = 'STAY'; self.stay_counter = random.randint(3, 5); self.target = None; self.phase = None

    def get_shortest_path_step(self, target):
        try:
            path = nx.shortest_path(self.graph, self.current_node, target)
            return path[1] if len(path) > 1 else self.current_node
        except: return self.current_node

    def step(self):
        if self.finished: return PAD_TOKEN
        self.trajectory.append(self.current_node)
        if self.state == 'STAY':
            self.stay_counter -= 1
            if self.stay_counter <= 0:
                if self.type == 'stopover': self.phase = 'GO_HOME'; self.state = 'WALK'; self.target = self.start_node
                elif self.type == 'wander':
                    self.state = 'WALK'; self.target = random.choice(AREA_SHOP1 + AREA_SHOP2)
                    while self.target == self.current_node: self.target = random.choice(AREA_SHOP1 + AREA_SHOP2)
            return self.current_node
        if self.state == 'WALK':
            if self.current_node == self.target:
                if self.type == 'through': self.finished = True; self.state = 'FINISHED'
                elif self.type == 'stopover':
                    if self.phase == 'GO_TO_SHOP': self.state = 'STAY'; self.stay_counter = random.randint(5, 10)
                    elif self.phase == 'GO_HOME': self.finished = True; self.state = 'FINISHED'
                elif self.type == 'wander': self.state = 'STAY'; self.stay_counter = random.randint(5, 15)
                return self.current_node
            self.current_node = self.get_shortest_path_step(self.target)
            return self.current_node
        return PAD_TOKEN

def generate_test_data(num_agents=300, max_steps=60):
    random.seed(1234)
    np.random.seed(1234)
    data = []
    types = ['through', 'stopover', 'wander']
    for i in range(num_agents):
        # 均等に生成
        b_type = types[i % 3]
        agent = AgentV2(i, G, b_type)
        seq = [agent.step() for _ in range(max_steps)]
        # パディング除去（可視化の邪魔になるので有効部分のみ）
        valid_seq = [x for x in seq if x != PAD_TOKEN]
        if len(valid_seq) > 5: # 短すぎるのは除外
            data.append({'id': i, 'type': b_type, 'seq': valid_seq})
    return data

# =========================================================
# 3. 潜在変数抽出 & 可視化関数
# =========================================================
def load_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get('config', {
        'vocab_size': 20, 'token_emb_dim': 64, 'd_model': 64, 
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

def extract_latent_vars(model, data_list, device):
    """
    データリストを入力し、潜在変数zのリストを返す
    returns: List of (type, z_numpy_array[seq_len, z_dim], original_seq)
    """
    results = []
    with torch.no_grad():
        for item in data_list:
            seq = item['seq']
            inp = torch.tensor([seq], dtype=torch.long).to(device)
            
            # モデルから出力を取得
            # 想定: model(inp) -> (logits, z) または (logits, z, ...)
            out = model(inp)
            
            if isinstance(out, tuple):
                z = out[1] # 2番目の要素がzと仮定
            else:
                # もしzが返ってこない実装の場合、ここでエラーになる
                # その場合はKP_RF.pyのforwardで `return x, h` のようにzを返すよう修正が必要
                print("Error: Model does not return latent variable z.")
                return []
                
            # z: [1, seq_len, z_dim] -> [seq_len, z_dim]
            z_np = z[0].cpu().numpy()
            results.append({
                'type': item['type'],
                'z': z_np,
                'seq': seq,
                'id': item['id']
            })
    return results

def visualize_analysis(model_path, model_name, test_data, device):
    print(f"\nAnalyzing {model_name}...")
    model = load_model(model_path, device)
    
    # 1. 潜在変数抽出
    latent_data = extract_latent_vars(model, test_data, device)
    if not latent_data: return

    # 2. PCA学習 (全データの全ステップを結合して学習)
    all_z = np.concatenate([d['z'] for d in latent_data], axis=0)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_z) # [Total_Steps, 2]
    
    # 各データにPCA座標を割り当て直す
    current_idx = 0
    for d in latent_data:
        length = len(d['z'])
        d['z_pca'] = pca_result[current_idx : current_idx + length]
        current_idx += length

    # --- Plot 1: 全体分布 (Global Distribution) ---
    # 各軌跡の「重心(平均)」をプロットして、タイプごとの分離を見る
    plt.figure(figsize=(10, 8))
    colors = {'through': 'red', 'stopover': 'blue', 'wander': 'green'}
    markers = {'through': '^', 'stopover': 'o', 'wander': 's'}
    
    for d in latent_data:
        # 軌跡の平均位置
        center = np.mean(d['z_pca'], axis=0)
        plt.scatter(center[0], center[1], color=colors[d['type']], marker=markers[d['type']], alpha=0.6, s=50)
        
    # 凡例用ダミー
    for t, c in colors.items():
        plt.scatter([], [], color=c, label=t, marker=markers[t])
        
    plt.title(f"Latent Space Distribution (Centroids) - {model_name}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, f"global_dist_{model_name}.png"))
    plt.close()

    # --- Plot 2: 個別軌跡の詳細 (Trajectories with Node IDs) ---
    # 各タイプから1つずつ代表を選んで描画
    selected_samples = {}
    for t in ['through', 'stopover', 'wander']:
        # ランダムに1つ選ぶ (あるいは長めのものを選ぶ)
        candidates = [d for d in latent_data if d['type'] == t and len(d['seq']) > 15]
        if candidates:
            selected_samples[t] = candidates[0] # 先頭を採用

    plt.figure(figsize=(14, 10))
    
    for t, d in selected_samples.items():
        z_pca = d['z_pca']
        seq = d['seq']
        c = colors[t]
        
        # 軌跡の線を描画
        plt.plot(z_pca[:, 0], z_pca[:, 1], color=c, alpha=0.5, linewidth=1, label=f"{t} (Agent {d['id']})")
        
        # 各点にノードIDをテキストで描画
        # 重なりを防ぐため、少し間引くか、全点打つか
        # ユーザー要望:「なるべくプロット点毎にノード番号」
        for i in range(len(seq)):
            node_id = seq[i]
            x, y = z_pca[i]
            
            # 点を打つ
            plt.scatter(x, y, color=c, s=15, alpha=0.8)
            
            # 文字を書く (少しずらす)
            plt.text(x, y, str(node_id), fontsize=8, color='black', alpha=0.8)
            
            # Start/Endを強調
            if i == 0: plt.text(x, y, "Start", fontsize=10, fontweight='bold', color=c)
            if i == len(seq)-1: plt.text(x, y, "End", fontsize=10, fontweight='bold', color=c)

    plt.title(f"Trajectory Evolution with Node IDs - {model_name}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, f"trajectory_detail_{model_name}.png"))
    plt.close()

# =========================================================
# 実行
# =========================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. テストデータ生成
    test_data = generate_test_data(num_agents=300, max_steps=60)
    print(f"Generated {len(test_data)} test trajectories.")

    # 2. With Koopman の分析
    visualize_analysis(MODEL_PATH_WITH, "With_Koopman", test_data, device)
    
    # 3. Without Koopman の分析
    visualize_analysis(MODEL_PATH_WITHOUT, "Without_Koopman", test_data, device)
    
    print("Done.")