import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import networkx as nx
import random
import io
import os
import matplotlib.pyplot as plt
import Levenshtein  # pip install python-Levenshtein
from datetime import datetime

# ★重要: 自作モデル定義ファイルのインポート
try:
    from KP_RF import KoopmanRoutesFormer
except ImportError:
    # ファイルがない場合はエラー
    raise ImportError("KP_RF.py not found. Please place this script in the same directory as KP_RF.py")

# ★★★ 要修正: モデルパス ★★★
# 学習し直した新しいモデルのパスを指定してください
MODEL_KOOPMAN_PATH = "/home/mizutani/projects/RF/runs/20251218_033604/model_weights_20251218_033604.pth"
MODEL_NORMAL_PATH  = "/home/mizutani/projects/RF/runs/20251218_034727/model_weights_20251218_034727.pth"


# =========================================================
# 0. 保存先設定
# =========================================================
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"/home/mizutani/projects/RF/runs/comparison_eval_v2_{run_id}"
os.makedirs(out_dir, exist_ok=True)

print(f"=== Evaluation Started: {run_id} ===")
print(f"Results will be saved to: {out_dir}")

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
PAD_TOKEN = 19

# =========================================================
# 2. AgentV2 クラス (synth_gen_v2.py と同じロジック)
# =========================================================
class AgentV2:
    def __init__(self, agent_id, graph, behavior_type):
        self.id = agent_id
        self.graph = graph
        self.type = behavior_type
        self.trajectory = []
        self.finished = False
        
        if self.type == 'through':
            # Start -> Goal (最短)
            self.start_node = random.choice([0, 3])
            self.goal_node = random.choice([16, 18])
            self.current_node = self.start_node
            self.state = 'WALK'
            self.target = self.goal_node
            self.phase = None
            
        elif self.type == 'stopover':
            # Start -> Shop (Stay) -> Start
            self.start_node = random.choice([0, 3])
            self.shop_node = random.choice([2, 4, 8, 10])
            self.current_node = self.start_node
            self.state = 'WALK'
            self.target = self.shop_node
            self.phase = 'GO_TO_SHOP'
            
        elif self.type == 'wander':
            # Shop -> Shop (Stay) -> ...
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
            return PAD_TOKEN
        
        self.trajectory.append(self.current_node)

        # --- Stay ---
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

        # --- Walk ---
        if self.state == 'WALK':
            if self.current_node == self.target:
                if self.type == 'through':
                    self.finished = True
                    self.state = 'FINISHED'
                elif self.type == 'stopover':
                    if self.phase == 'GO_TO_SHOP':
                        self.state = 'STAY'
                        self.stay_counter = random.randint(5, 10) # 固定滞在
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

def generate_ground_truth_test(num_agents=100, max_steps=60):
    # シードを変えてテストデータを生成
    random.seed(999) 
    np.random.seed(999)
    test_data = []
    types = ['through', 'stopover', 'wander']
    
    for i in range(num_agents):
        b_type = random.choice(types)
        agent = AgentV2(i, G, b_type)
        full_seq = []
        for _ in range(max_steps):
            node = agent.step()
            full_seq.append(node)
            
        test_data.append({
            'agent_id': i,
            'type': b_type,
            'trajectory': full_seq
        })
    return test_data

# =========================================================
# 3. モデルロード関数
# =========================================================
def load_model(model_path, device):
    save_log(f"Loading model from: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except FileNotFoundError:
        save_log(f"Error: Model file not found at {model_path}")
        raise

    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # デフォルト設定 (学習コードに合わせて変更してください)
        config = {'vocab_size': 20, 'token_emb_dim': 64, 'd_model': 64, 'nhead': 4, 
                  'num_layers': 6, 'd_ff': 128, 'z_dim': 16, 'pad_token_id': 19}

    model = KoopmanRoutesFormer(
        vocab_size=config['vocab_size'],
        token_emb_dim=config['token_emb_dim'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        z_dim=config['z_dim'],
        pad_token_id=config.get('pad_token_id', 19)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, config

# =========================================================
# 4. 推論ロジック
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
            
            # ★注意: 今回のテストでは「パディングも正しく予測できるか」を見たいので
            # 予測ループを途中でbreakせず、最後まで回します。
            # ただし、next_tokenがPADだった場合の処理はモデルの挙動に任せます。
            current_seq.append(next_token)
            
    generated_future = current_seq[len(initial_seq):]
    return generated_future

def evaluate_models(model_koopman, model_normal, test_data, prompt_len=15, device='cuda'):
    results = []
    save_log(f"Evaluating on {len(test_data)} test trajectories...")
    to_str = lambda seq: "".join([chr(x + 65) for x in seq])

    for i, data in enumerate(test_data):
        full_traj = data['trajectory']
        
        # プロンプトより短いデータはあり得ない(paddingが入るため)が念のため
        if len(full_traj) <= prompt_len: continue
            
        prompt_seq = full_traj[:prompt_len]
        gt_future = full_traj[prompt_len:]
        pred_len = len(gt_future)
        
        # 推論
        pred_k_future = predict_trajectory(model_koopman, prompt_seq, pred_len, device)
        pred_n_future = predict_trajectory(model_normal, prompt_seq, pred_len, device)
        
        # 距離計算 (Padding=19 も文字 'T' などとして比較に含まれるので、停止予測も評価される)
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
# 5. メイン実行スクリプト
# =========================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_log(f"Device: {device}")


    try:
        model_koopman, _ = load_model(MODEL_KOOPMAN_PATH, device)
        model_normal, _ = load_model(MODEL_NORMAL_PATH, device)
    except Exception as e:
        save_log(f"Failed to load models: {e}")
        # テスト実行用にダミーで進める場合はここをコメントアウト
        exit(1)

    # テストデータ生成
    TEST_AGENTS = 500
    MAX_STEPS = 60
    PROMPT_STEPS = 15

    gt_test_data = generate_ground_truth_test(num_agents=TEST_AGENTS, max_steps=MAX_STEPS)
    save_log(f"Generated {len(gt_test_data)} ground truth trajectories (V2 Logic).")

    # 評価
    df_res = evaluate_models(model_koopman, model_normal, gt_test_data, prompt_len=PROMPT_STEPS, device=device)

    # 保存
    csv_path = os.path.join(out_dir, "evaluation_results.csv")
    df_res.to_csv(csv_path, index=False)
    
    mean_scores = df_res[['score_k', 'score_n']].mean()
    save_log("\n=== Evaluation Summary (Lower is Better) ===")
    save_log(f"Koopman: {mean_scores['score_k']:.4f}, Normal: {mean_scores['score_n']:.4f}")
    
    # タイプ別スコア
    type_scores = df_res.groupby('type')[['score_k', 'score_n']].mean()
    save_log("\n=== By Behavior Type ===")
    save_log(type_scores.to_string())

    # =========================================================
    # F. 可視化 (パディング対応版)
    # =========================================================
    save_log("\n=== Selecting Plots ===")
    df_res['diff'] = df_res['score_n'] - df_res['score_k'] # 正ならKoopman勝ち
    
    target_types = ['through', 'stopover', 'wander']
    
    def clean_seq(seq):
        """パディング(19)を除外したリストを返す（描画用）"""
        return [x for x in seq if x != PAD_TOKEN]

    def plot_trajectory(row, title_prefix=""):
        plt.figure(figsize=(12, 5))
        prompt_len = len(row['prompt'])
        
        # パディングを含む全系列を作成
        full_gt_raw = row['prompt'] + row['gt']
        pred_k_raw = row['prompt'] + row['pred_k']
        pred_n_raw = row['prompt'] + row['pred_n']
        
        # 描画用にパディングを除去するが、インデックス(x軸)はずらさないように工夫する
        # 方法: 値が19の箇所は np.nan にしてプロットする
        def to_plot_array(seq):
            arr = np.array(seq, dtype=float)
            arr[arr == PAD_TOKEN] = np.nan
            return arr

        gt_plot = to_plot_array(full_gt_raw)
        k_plot = to_plot_array(pred_k_raw)
        n_plot = to_plot_array(pred_n_raw)
        
        # Ground Truth
        plt.plot(gt_plot, label='Ground Truth', color='black', linewidth=2, marker='o', alpha=0.3)
        plt.axvline(x=prompt_len-0.5, color='gray', linestyle='--', label='Input End')
        
        # Predictions (Input以降のみプロット)
        x_range = range(prompt_len, len(full_gt_raw))
        
        # Koopman
        plt.plot(x_range, k_plot[prompt_len:], label=f'Koopman (Score: {row["score_k"]:.2f})', color='red', linestyle='-', linewidth=2)
        
        # Normal
        plt.plot(x_range, n_plot[prompt_len:], label=f'Normal (Score: {row["score_n"]:.2f})', color='blue', linestyle=':', linewidth=2)
        
        plt.title(f"{title_prefix} | Agent {row['id']} ({row['type']})\nDiff: {row['diff']:.3f} (Pos=Koopman Win)")
        plt.xlabel("Time Step")
        plt.ylabel("Node ID")
        plt.ylim(-1, 20)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        fname = f"{title_prefix.replace(' ', '_')}_{row['type']}_{row['id']}.png"
        save_path = os.path.join(out_dir, fname)
        plt.savefig(save_path)
        plt.close()

    # 勝っているケースを抽出してプロット
    for b_type in target_types:
        df_type = df_res[df_res['type'] == b_type]
        if len(df_type) == 0: continue

        # Koopman Wins (Top 2)
        df_k_win = df_type[df_type['diff'] > 0.05].sort_values(by='diff', ascending=False)
        for _, row in df_k_win.head(2).iterrows():
            plot_trajectory(row, title_prefix="Koopman_Win")
            
        # Normal Wins (Top 2)
        df_n_win = df_type[df_type['diff'] < -0.05].sort_values(by='diff', ascending=True)
        for _, row in df_n_win.head(2).iterrows():
            plot_trajectory(row, title_prefix="Normal_Win")

    save_log("Done.")