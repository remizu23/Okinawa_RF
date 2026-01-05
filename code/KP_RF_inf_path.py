import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import networkx as nx
import random
import io
import os
import matplotlib.pyplot as plt
import Levenshtein
from datetime import datetime

# ★重要: 自作モデル定義ファイルのインポート
try:
    from KP_RF import KoopmanRoutesFormer
except ImportError:
    raise ImportError("KP_RF.py not found.")

# =========================================================
# ★設定変更エリア
# =========================================================
# パディングトークン定義 (v3仕様)
PAD_TOKEN = 38 
STAY_OFFSET = 19
VOCAB_SIZE = 39 # 0-18(Move) + 19-37(Stay) + 38(Pad)

# モデルパス
MODEL_KOOPMAN_PATH = "/home/mizutani/projects/RF/runs/20260104_182356/model_weights_20260104_182356.pth"
MODEL_NORMAL_PATH  = "/home/mizutani/projects/RF/runs/20260105_013224/model_weights_20260105_013224.pth"

# =========================================================
# 0. 保存先設定
# =========================================================
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"/home/mizutani/projects/RF/runs/comparison_eval_v3_{run_id}"
os.makedirs(out_dir, exist_ok=True)

print(f"=== Evaluation Started: {run_id} ===")
def save_log(msg):
    print(msg)
    with open(os.path.join(out_dir, "evaluation_log.txt"), "a") as f:
        f.write(msg + "\n")

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

# =========================================================
# 2. AgentV2 クラス (物理的な動きのみ担当)
# =========================================================
class AgentV2:
    def __init__(self, agent_id, graph, behavior_type):
        self.id = agent_id
        self.graph = graph
        self.type = behavior_type
        self.trajectory = []
        self.finished = False
        
        # 初期設定 (変更なし)
        if self.type == 'through':
            self.start_node = random.choice([0, 3])
            self.goal_node = random.choice([16, 18])
            self.current_node = self.start_node
            self.state = 'WALK'
            self.target = self.goal_node
            self.phase = None
        elif self.type == 'stopover':
            self.start_node = random.choice([0, 3])
            self.shop_node = random.choice([2, 4, 8, 10])
            self.current_node = self.start_node
            self.state = 'WALK'
            self.target = self.shop_node
            self.phase = 'GO_TO_SHOP'
        elif self.type == 'wander':
            self.current_node = random.choice(AREA_SHOP1)
            self.state = 'STAY' 
            self.stay_counter = random.randint(3, 5)
            self.target = None
            self.phase = None

    def get_shortest_path_step(self, target):
        try:
            path = nx.shortest_path(self.graph, self.current_node, target)
            if len(path) > 1: return path[1]
            else: return self.current_node
        except nx.NetworkXNoPath:
            return self.current_node

    def step(self):
        if self.finished:
            return PAD_TOKEN # ここは後で一括置換されるので仮の値でOK
        
        self.trajectory.append(self.current_node)

        # Stay Logic
        if self.state == 'STAY':
            self.stay_counter -= 1
            if self.stay_counter <= 0:
                if self.type == 'stopover':
                    self.phase = 'GO_HOME'
                    self.state = 'WALK'
                    self.target = self.start_node
                elif self.type == 'wander':
                    self.state = 'WALK'
                    self.target = random.choice(AREA_SHOP1 + AREA_SHOP2)
                    while self.target == self.current_node:
                        self.target = random.choice(AREA_SHOP1 + AREA_SHOP2)
            return self.current_node

        # Walk Logic
        if self.state == 'WALK':
            if self.current_node == self.target:
                if self.type == 'through':
                    self.finished = True
                    self.state = 'FINISHED'
                elif self.type == 'stopover':
                    if self.phase == 'GO_TO_SHOP':
                        self.state = 'STAY'
                        self.stay_counter = random.randint(5, 10)
                    elif self.phase == 'GO_HOME':
                        self.finished = True
                        self.state = 'FINISHED'
                elif self.type == 'wander':
                    self.state = 'STAY'
                    self.stay_counter = random.randint(5, 15)
                return self.current_node

            next_node = self.get_shortest_path_step(self.target)
            self.current_node = next_node
            return self.current_node
        
        return PAD_TOKEN

# ★★★ ここが重要：変換ロジック ★★★
def convert_seq_to_stay_format(raw_seq):
    """
    生のノードID列(0-18)を受け取り、v3形式(滞在オフセットあり)に変換する。
    例: [0, 0, 0, 1] -> [0, 19, 19, 1]
    """
    if not raw_seq:
        return []
    
    new_seq = []
    # 最初の1つ目は常に「移動(新規)」扱い
    new_seq.append(raw_seq[0])
    
    for i in range(1, len(raw_seq)):
        curr = raw_seq[i]
        prev = raw_seq[i-1]
        
        # 1. パディング(古い定義の19や新しい38)なら、今回のPAD_TOKEN(38)にする
        # ※Agentからは仮のPADとして何かが返ってくるが、ここで統一
        if curr >= 19: # 既存ロジックではPAD=19が返ってくる
            new_seq.append(PAD_TOKEN)
            continue
            
        # 2. 滞在判定 (前回と同じ場所 かつ パディングではない)
        if curr == prev and prev < 19: # 19は古いPADなので除外
            new_seq.append(curr + STAY_OFFSET)
        else:
            new_seq.append(curr)
            
    return new_seq

def generate_ground_truth_test(num_agents=100, max_steps=60):
    random.seed(999) 
    np.random.seed(999)
    test_data = []
    types = ['through', 'stopover', 'wander']
    
    for i in range(num_agents):
        b_type = random.choice(types)
        agent = AgentV2(i, G, b_type)
        full_seq_raw = [] # 生データ
        
        for _ in range(max_steps):
            node = agent.step()
            full_seq_raw.append(node)
            
        # ★ここで変換を噛ませる★
        converted_seq = convert_seq_to_stay_format(full_seq_raw)
        
        test_data.append({
            'agent_id': i,
            'type': b_type,
            'trajectory': converted_seq
        })
    return test_data

# =========================================================
# 3. モデルロード関数
# =========================================================
def load_model(model_path, device):
    save_log(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # デフォルト設定をv3用に更新
    default_config = {
        'vocab_size': VOCAB_SIZE, # 39
        'token_emb_dim': 64, 'd_model': 64, 'nhead': 4, 
        'num_layers': 6, 'd_ff': 128, 'z_dim': 16, 
        'pad_token_id': PAD_TOKEN # 38
    }
    
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = default_config

    model = KoopmanRoutesFormer(
        vocab_size=config.get('vocab_size', VOCAB_SIZE),
        token_emb_dim=config['token_emb_dim'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        z_dim=config['z_dim'],
        pad_token_id=config.get('pad_token_id', PAD_TOKEN)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, config

# =========================================================
# 4. 推論 & 評価
# =========================================================
def predict_trajectory(model, initial_seq, predict_len, device):
    model.eval()
    current_seq = initial_seq.copy()
    
    with torch.no_grad():
        for _ in range(predict_len):
            input_tensor = torch.tensor([current_seq], dtype=torch.long).to(device)
            output = model(input_tensor)
            
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            
            last_timestep_logits = logits[0, -1, :]
            next_token = torch.argmax(last_timestep_logits).item()
            current_seq.append(next_token)
            
    generated_future = current_seq[len(initial_seq):]
    return generated_future

def evaluate_models(model_koopman, model_normal, test_data, prompt_len=15, device='cuda'):
    results = []
    save_log(f"Evaluating on {len(test_data)} test trajectories...")
    
    # 評価用の文字変換 (0-38をユニークな文字へ)
    # 39文字必要なので、A-Z(26)だけでは足りない。ASCIIコードを使う
    to_str = lambda seq: "".join([chr(x + 48) for x in seq]) # 0='0', 1='1'...

    for i, data in enumerate(test_data):
        full_traj = data['trajectory']
        
        if len(full_traj) <= prompt_len: continue
            
        prompt_seq = full_traj[:prompt_len]
        gt_future = full_traj[prompt_len:]
        pred_len = len(gt_future)
        
        pred_k_future = predict_trajectory(model_koopman, prompt_seq, pred_len, device)
        pred_n_future = predict_trajectory(model_normal, prompt_seq, pred_len, device)
        
        dist_k = Levenshtein.distance(to_str(gt_future), to_str(pred_k_future))
        dist_n = Levenshtein.distance(to_str(gt_future), to_str(pred_n_future))
        
        score_k = dist_k / len(gt_future)
        score_n = dist_n / len(gt_future)
        
        results.append({
            'id': i,
            'type': data['type'],
            'score_k': score_k,
            'score_n': score_n,
            'prompt': prompt_seq,
            'gt': gt_future,
            'pred_k': pred_k_future,
            'pred_n': pred_n_future
        })
        
        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/{len(test_data)}...", end='\r')
            
    return pd.DataFrame(results)

# =========================================================
# 5. 可視化ロジック (デコード対応)
# =========================================================
def decode_for_plot(seq):
    """
    プロット用にトークンを物理的な位置(Y軸)に戻す
    19-37 (Stay) -> 0-18
    38 (Pad) -> NaN
    """
    arr = np.array(seq, dtype=float)
    
    # Stayの処理: 19以上38未満なら -19 する
    # ※論理演算を使って一括処理
    is_stay = (arr >= STAY_OFFSET) & (arr < PAD_TOKEN)
    arr[is_stay] -= STAY_OFFSET
    
    # Padの処理
    arr[arr == PAD_TOKEN] = np.nan
    return arr

def plot_results(df_res):
    df_res['diff'] = df_res['score_n'] - df_res['score_k']
    target_types = ['through', 'stopover', 'wander']
    
    for b_type in target_types:
        df_type = df_res[df_res['type'] == b_type]
        if len(df_type) == 0: continue

        # Koopman Wins (Top 1)
        k_wins = df_type[df_type['diff'] > 0.05].sort_values(by='diff', ascending=False)
        if not k_wins.empty:
            plot_single(k_wins.iloc[0], "Koopman_Win")
            
        # Normal Wins (Top 1)
        n_wins = df_type[df_type['diff'] < -0.05].sort_values(by='diff', ascending=True)
        if not n_wins.empty:
            plot_single(n_wins.iloc[0], "Normal_Win")

def plot_single(row, title_prefix):
    plt.figure(figsize=(12, 5))
    prompt_len = len(row['prompt'])
    
    # 結合してデコード
    full_gt = decode_for_plot(row['prompt'] + row['gt'])
    full_k  = decode_for_plot(row['prompt'] + row['pred_k'])
    full_n  = decode_for_plot(row['prompt'] + row['pred_n'])
    
    # GT
    plt.plot(full_gt, label='Ground Truth', color='black', linewidth=3, alpha=0.3)
    plt.axvline(x=prompt_len-0.5, color='gray', linestyle='--')
    
    # Predictions
    x_range = range(prompt_len, len(full_gt))
    plt.plot(x_range, full_k[prompt_len:], label=f'Koopman ({row["score_k"]:.2f})', color='red')
    plt.plot(x_range, full_n[prompt_len:], label=f'Normal ({row["score_n"]:.2f})', color='blue', linestyle=':')
    
    plt.title(f"{title_prefix} | {row['type']} (Agent {row['id']})")
    plt.xlabel("Time")
    plt.ylabel("Node ID (Decoded)")
    plt.yticks(range(0, 19))
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(out_dir, f"{title_prefix}_{row['type']}_{row['id']}.png"))
    plt.close()

# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. モデルロード
    try:
        model_koopman, _ = load_model(MODEL_KOOPMAN_PATH, device)
        model_normal, _ = load_model(MODEL_NORMAL_PATH, device)
    except Exception as e:
        save_log(f"Model Load Error: {e}")
        exit()
        
    # 2. データ生成 (AgentV2 -> convert_seq_to_stay_format)
    gt_data = generate_ground_truth_test(num_agents=500, max_steps=60)
    save_log(f"Generated {len(gt_data)} trajectories (Stay Extended Format).")
    
    # 3. 評価
    df_res = evaluate_models(model_koopman, model_normal, gt_data, prompt_len=15, device=device)
    
    # 4. 集計
    csv_path = os.path.join(out_dir, "evaluation_results.csv")
    df_res.to_csv(csv_path, index=False)
    
    save_log("\n=== Scores ===")
    save_log(df_res[['score_k', 'score_n']].mean().to_string())
    save_log("\n=== By Type ===")
    save_log(df_res.groupby('type')[['score_k', 'score_n']].mean().to_string())
    
    # 5. プロット
    plot_results(df_res)
    save_log("\nDone.")