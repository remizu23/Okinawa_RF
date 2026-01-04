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

# ★ KP_RF.py の読み込み
try:
    from KP_RF import KoopmanRoutesFormer
except ImportError:
    raise ImportError("KP_RF.py not found.")

# =========================================================
# 0. 設定 & 保存先
# =========================================================
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"/home/mizutani/projects/RF/runs/z_analysis_v2_{run_id}"
os.makedirs(out_dir, exist_ok=True)

print(f"=== Latent Analysis V2: {run_id} ===")
print(f"Results will be saved to: {out_dir}")

# ★★★ 要修正: モデルパス ★★★
MODEL_PATH_WITH = "/home/mizutani/projects/RF/runs/20251218_033604/model_weights_20251218_033604.pth"
MODEL_PATH_WITHOUT = "/home/mizutani/projects/RF/runs/20251218_034727/model_weights_20251218_034727.pth"

# =========================================================
# 1. 環境設定
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
# 2. データ生成 (AgentV2)
# =========================================================
class AgentV2:
    def __init__(self, agent_id, graph, behavior_type):
        self.id = agent_id; self.graph = graph; self.type = behavior_type
        self.trajectory = []; self.finished = False
        if self.type == 'through':
            self.start_node = random.choice([0, 3]); self.goal_node = random.choice([16, 18])
            self.current_node = self.start_node; self.state = 'WALK'; self.target = self.goal_node; self.phase = None
        elif self.type == 'stopover':
            self.start_node = random.choice([0, 3]); self.shop_node = random.choice([2, 4, 8, 10])
            self.current_node = self.start_node; self.state = 'WALK'; self.target = self.shop_node; self.phase = 'GO_TO_SHOP'
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
    random.seed(1234); np.random.seed(1234)
    data = []
    types = ['through', 'stopover', 'wander']
    for i in range(num_agents):
        b_type = types[i % 3]
        agent = AgentV2(i, G, b_type)
        seq = [agent.step() for _ in range(max_steps)]
        valid_seq = [x for x in seq if x != PAD_TOKEN]
        if len(valid_seq) > 5:
            data.append({'id': i, 'type': b_type, 'seq': valid_seq})
    return data

# =========================================================
# 3. 分析・可視化ロジック
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

def get_latent_data(model, data_list, device):
    results = []
    with torch.no_grad():
        for item in data_list:
            seq = item['seq']
            inp = torch.tensor([seq], dtype=torch.long).to(device)
            out = model(inp)
            z = out[1] if isinstance(out, tuple) else None
            if z is None: return []
            z_np = z[0].cpu().numpy()
            results.append({'type': item['type'], 'z': z_np, 'seq': seq, 'id': item['id']})
    
    # PCA
    if not results: return []
    all_z = np.concatenate([d['z'] for d in results], axis=0)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_z)
    
    current_idx = 0
    for d in results:
        length = len(d['z'])
        d['z_pca'] = pca_result[current_idx : current_idx + length]
        current_idx += length
        
    return results

def get_plot_limits(data_list_1, data_list_2):
    """2つのデータセットから共通のX, Yの範囲を計算する"""
    all_x, all_y = [], []
    for d in data_list_1 + data_list_2:
        all_x.extend(d['z_pca'][:, 0])
        all_y.extend(d['z_pca'][:, 1])
    
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    
    # 少し余裕を持たせる
    margin_x = (x_max - x_min) * 0.1
    margin_y = (y_max - y_min) * 0.1
    return (x_min - margin_x, x_max + margin_x), (y_min - margin_y, y_max + margin_y)

def visualize_comparison(data_with, data_without):
    colors = {'through': 'red', 'stopover': 'blue', 'wander': 'green'}
    markers = {'through': '^', 'stopover': 'o', 'wander': 's'}
    
    # 共通スケールの計算
    xlim, ylim = get_plot_limits(data_with, data_without)
    print(f"Unified Scale -- X: {xlim}, Y: {ylim}")

    # --- 1. Global Distribution (重心プロット) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    
    for ax, data, title in zip(axes, [data_with, data_without], ["With Koopman", "Without Koopman"]):
        for d in data:
            center = np.mean(d['z_pca'], axis=0)
            ax.scatter(center[0], center[1], color=colors[d['type']], marker=markers[d['type']], alpha=0.5, s=60)
        
        # 凡例
        for t, c in colors.items():
            ax.scatter([], [], color=c, label=t, marker=markers[t])
            
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
    plt.suptitle("Global Latent Distribution (Centroids) - Unified Scale")
    plt.savefig(os.path.join(out_dir, "comparison_global_dist.png"))
    plt.close()

    # --- 2. Trajectory Details (3行2列) ---
    types = ['through', 'stopover', 'wander']
    fig, axes = plt.subplots(3, 2, figsize=(16, 18)) # 行:Type, 列:Model
    
    # 表示するエージェントIDを固定する（比較のため同じIDを表示）
    # 各タイプごとに、データが長いエージェントを5人選ぶ
    sample_ids = {}
    for t in types:
        candidates = [d['id'] for d in data_with if d['type'] == t and len(d['seq']) > 10]
        sample_ids[t] = candidates[:5] # 最初の5人
    
    cmap = plt.get_cmap('tab10') # エージェントごとに色を変える用

    for i, t in enumerate(types):
        target_ids = sample_ids[t]
        
        # --- With Koopman (左列) ---
        ax = axes[i, 0]
        subset = [d for d in data_with if d['id'] in target_ids]
        for k, d in enumerate(subset):
            z = d['z_pca']
            # 軌跡
            ax.plot(z[:,0], z[:,1], marker='.', markersize=4, label=f"ID {d['id']}", color=cmap(k), alpha=0.7)
            # ノード番号 (間引いて表示)
            for step, (val_x, val_y) in enumerate(z):
                if step % 2 == 0 or step == len(z)-1: # 2歩ごとに表示
                    ax.text(val_x, val_y, str(d['seq'][step]), fontsize=8, color=cmap(k))
            # Start
            ax.text(z[0,0], z[0,1], "S", fontweight='bold', fontsize=12, color=cmap(k))

        ax.set_title(f"With Koopman - {t.capitalize()}")
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # --- Without Koopman (右列) ---
        ax = axes[i, 1]
        subset = [d for d in data_without if d['id'] in target_ids]
        for k, d in enumerate(subset):
            z = d['z_pca']
            ax.plot(z[:,0], z[:,1], marker='.', markersize=4, label=f"ID {d['id']}", color=cmap(k), alpha=0.7)
            for step, (val_x, val_y) in enumerate(z):
                if step % 2 == 0 or step == len(z)-1:
                    ax.text(val_x, val_y, str(d['seq'][step]), fontsize=8, color=cmap(k))
            ax.text(z[0,0], z[0,1], "S", fontweight='bold', fontsize=12, color=cmap(k))

        ax.set_title(f"Without Koopman - {t.capitalize()}")
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_trajectories.png"))
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

    # 2. 潜在変数取得
    print("Processing With Koopman...")
    model_with = load_model(MODEL_PATH_WITH, device)
    data_with = get_latent_data(model_with, test_data, device)
    
    print("Processing Without Koopman...")
    model_without = load_model(MODEL_PATH_WITHOUT, device)
    data_without = get_latent_data(model_without, test_data, device)
    
    # 3. 比較可視化
    visualize_comparison(data_with, data_without)
    
    print("Done.")