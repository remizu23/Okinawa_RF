import torch
import torch.nn as nn
import torch.nn.functional as F
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
VOCAB_SIZE = 39 

# モデルパス (適宜変更してください)
# ※一方がカウント対応、もう一方が非対応でも動くように作るのが理想です
MODEL_KOOPMAN_PATH = "/home/mizutani/projects/RF/runs/20260107_124130/model_weights_20260107_124130.pth"
MODEL_NORMAL_PATH  = "/home/mizutani/projects/RF/runs/20260105_234558/model_weights_20260105_234558.pth"

# =========================================================
# 0. 保存先設定
# =========================================================
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"/home/mizutani/projects/RF/runs/comparison_eval_v3_simpleLoss_{run_id}"
os.makedirs(out_dir, exist_ok=True)

print(f"=== Evaluation Started: {run_id} ===")
def save_log(msg):
    print(msg)
    with open(os.path.join(out_dir, "evaluation_log.txt"), "a") as f:
        f.write(msg + "\n")

# =========================================================
# 1. 環境設定 & マップ構築 (変更なし)
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

# --- 定数定義 ---
AREA_SHOP = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16] 
ENTRY_EXIT_NODES = [0, 18]
NUM_NODES = 19

# --- 満足度パラメータ設定 (β) ---
rng_env = np.random.default_rng(42)
NODE_BETAS = {i: 0.0 for i in range(NUM_NODES)}
for node in AREA_SHOP:
    NODE_BETAS[node] = round(rng_env.uniform(0.5, 1.5), 2)

# =========================================================
# 2. AgentV3 クラス (活動欲求モデル) (変更なし)
# =========================================================
class AgentV3:
    def __init__(self, agent_id, graph, behavior_type):
        self.id = agent_id
        self.graph = graph
        self.type = behavior_type
        
        self.history_tokens = []    
        self.finished = False
        
        self.satisfaction_threshold = random.uniform(20.0, 35.0)    
        self.current_satisfaction = 0.0
        
        self.start_node = random.choice(ENTRY_EXIT_NODES)
        self.current_node = self.start_node
        self.history_tokens.append(self.current_node)
        
        self.state = 'WALK' 
        self.target = self._choose_next_shop()
        self.local_sat_goal = 0.0 

    def _choose_next_shop(self):
        candidates = [n for n in AREA_SHOP if n != self.current_node]
        if not candidates: return self.start_node
        return random.choice(candidates)

    def _choose_exit(self):
        return self.start_node 

    def get_shortest_path_step(self, target):
        try:
            path = nx.shortest_path(self.graph, self.current_node, target)
            if len(path) > 1: return path[1]
            return self.current_node
        except nx.NetworkXNoPath:
            return self.current_node

    def step(self):
        if self.finished:
            return PAD_TOKEN
        
        if self.state == 'STAY':
            beta = NODE_BETAS.get(self.current_node, 0)
            self.current_satisfaction += beta
            token = self.current_node + STAY_OFFSET
            self.history_tokens.append(token)
            if self.current_satisfaction >= self.satisfaction_threshold:
                self.state = 'WALK'
                self.target = self._choose_exit()
            elif self.current_satisfaction >= self.local_sat_goal:
                self.state = 'WALK'
                self.target = self._choose_next_shop()
            return token

        elif self.state == 'WALK':
            if self.current_node == self.target:
                if self.current_satisfaction >= self.satisfaction_threshold:
                    self.finished = True
                    self.state = 'FINISHED'
                    return PAD_TOKEN
                else:
                    self.state = 'STAY'
                    remaining = self.satisfaction_threshold - self.current_satisfaction
                    gain = remaining * random.uniform(0.3, 0.6)
                    gain = max(gain, 1.0)
                    self.local_sat_goal = self.current_satisfaction + gain
                    beta = NODE_BETAS.get(self.current_node, 0)
                    self.current_satisfaction += beta
                    token = self.current_node
                    self.history_tokens.append(token)
                    return token

            next_node = self.get_shortest_path_step(self.target)
            self.current_node = next_node
            self.history_tokens.append(self.current_node)
            return self.current_node
        
        return PAD_TOKEN

def generate_ground_truth_test(num_agents=100, max_steps=60):
    random.seed(999) 
    np.random.seed(999)
    test_data = []
    types = ['stopover', 'wander'] 
    
    for i in range(num_agents):
        b_type = random.choice(types)
        agent = AgentV3(i, G, b_type)
        
        for _ in range(max_steps - 1):
            token = agent.step()
            if token == PAD_TOKEN and agent.finished:
                break
        
        raw_traj = [t for t in agent.history_tokens if t != PAD_TOKEN]
        
        if len(raw_traj) > 5:
            test_data.append({
                'agent_id': i,
                'type': b_type,
                'trajectory': raw_traj
            })
            
    return test_data

# =========================================================
# 3. モデルロード関数 (★修正: 新パラメータ対応)
# =========================================================
def load_model(model_path, device):
    save_log(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    default_config = {
        'vocab_size': VOCAB_SIZE, 
        'token_emb_dim': 64, 'd_model': 64, 'nhead': 4, 
        'num_layers': 6, 'd_ff': 128, 'z_dim': 16, 
        'pad_token_id': PAD_TOKEN
    }
    
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = default_config

    # KoopmanRoutesFormerの初期化
    # configに新しいパラメータがある場合はそれを使う、なければデフォルト(1)
    # これにより新旧モデル両対応が可能
    model = KoopmanRoutesFormer(
        vocab_size=config.get('vocab_size', VOCAB_SIZE),
        token_emb_dim=config['token_emb_dim'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        z_dim=config['z_dim'],
        pad_token_id=config.get('pad_token_id', PAD_TOKEN),
        # ★追加: カウント・AgentID対応
        num_agents=config.get('num_agents', 1),
        agent_emb_dim=config.get('agent_emb_dim', 16),
        max_stay_count=config.get('max_stay_count', 100),
        stay_emb_dim=config.get('stay_emb_dim', 16)
    )
    
    # モデルがAgentID対応かどうかを判定するためのフラグを仕込んでおく
    model.has_extra_inputs = ('agent_emb_dim' in config) or (hasattr(model, 'agent_embedding'))

    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        # 念のため、足りなかったキーを表示して確認できるようにする
        missing_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False).missing_keys
        if missing_keys:
            print(f"Warning: Missing keys in state_dict (Safe to ignore if these are new layers): {missing_keys}")
    except RuntimeError as e:
        # strict=Falseでもサイズ不一致などで落ちる場合は報告
        save_log(f"Critical Error in load_state_dict: {e}")
        raise e    
    model.to(device)
    model.eval()
    return model, config

# =========================================================
# 4. 推論 & 評価 (★修正: カウント計算と3引数対応)
# =========================================================
def calculate_stay_counts_seq(seq):
    """
    系列リストを受け取り、滞在カウントリストを返すヘルパー関数
    """
    counts = []
    current_val = -1
    counter = 0
    for val in seq:
        if val == current_val:
            counter += 1
        else:
            counter = 1
            current_val = val
        counts.append(counter)
    return counts

def predict_trajectory(model, initial_seq, predict_len, agent_id, device):
    """
    model: 学習済みモデル
    initial_seq: プロンプト系列 (List[int])
    predict_len: 予測したい長さ
    agent_id: エージェントID (int) ★追加
    """
    model.eval()
    current_seq = initial_seq.copy()
    
    # 3引数が必要かどうかの判定 (load_modelでフラグ管理、またはtry-exceptでも可)
    # ここではモデルの属性を見て判断する簡易実装
    needs_extra = getattr(model, 'has_extra_inputs', False)
    # あるいは config の値を見て判断しても良いが、モデルインスタンスに属性があると安全
    
    with torch.no_grad():
        for _ in range(predict_len):
            input_tensor = torch.tensor([current_seq], dtype=torch.long).to(device)
            
            if needs_extra:
                # ★追加: 滞在カウントの計算 (毎回最初から計算しなおすのが確実)
                current_counts = calculate_stay_counts_seq(current_seq)
                stay_tensor = torch.tensor([current_counts], dtype=torch.long).to(device)
                
                # ★追加: Agent ID (バッチサイズ1)
                # モデル学習時のnum_agentsを超えないようにクリップするか、剰余をとる等の安全策
                # ここではそのまま渡すが、学習時より大きいIDが来るとエラーになるので注意
                # 学習データの最大IDを確認し、それ以下にする処理が必要なら入れる
                safe_agent_id = agent_id % model.agent_embedding.num_embeddings 
                agent_tensor = torch.tensor([safe_agent_id], dtype=torch.long).to(device)
                
                # 3引数でコール
                output = model(input_tensor, stay_tensor, agent_tensor)
            else:
                # 旧モデル (1引数)
                output = model(input_tensor)
            
            # タプル展開 (logits, z_hat, z_pred)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            
            last_timestep_logits = logits[0, -1, :]
            
            # 決定論的(argmax)か確率的(sample)か
            # 評価用なのでargmax推奨だが、分布を見たいならsample
            next_token = torch.argmax(last_timestep_logits).item()
            
            current_seq.append(next_token)
            
    generated_future = current_seq[len(initial_seq):]
    return generated_future

# =========================================================
# DTW / Evaluate Models (★修正: predict呼び出し引数)
# =========================================================
def calc_dtw(seq1, seq2):
    n, m = len(seq1), len(seq2)
    dtw = [[float('inf')] * (m + 1) for _ in range(n + 1)]
    dtw[0][0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq1[i-1] == seq2[j-1] else 1
            dtw[i][j] = cost + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])
    return dtw[n][m]

def evaluate_models(model_koopman, model_normal, test_data, prompt_len=15, device='cuda'):
    results = []
    print(f"Evaluating on {len(test_data)} test trajectories...")
    to_str = lambda seq: "".join([chr(x + 48) for x in seq]) 
    
    # ★追加: 評価したい最小のデータ長 (これより短いと捨てて、長いデータだけでテストする)
    MIN_TRAJ_LEN = 50

    for i, data in enumerate(test_data):
        full_traj = data['trajectory']
        agent_id = data['agent_id'] # ★取得

        if len(full_traj) <= MIN_TRAJ_LEN: continue
            
        prompt_seq = full_traj[:prompt_len]
        gt_future = full_traj[prompt_len:]
        pred_len = len(gt_future)
        
        # ★修正: agent_idを渡す
        pred_k_future = predict_trajectory(model_koopman, prompt_seq, pred_len, agent_id, device)
        pred_n_future = predict_trajectory(model_normal, prompt_seq, pred_len, agent_id, device)
        
        # Levenshtein
        dist_k_lev = Levenshtein.distance(to_str(gt_future), to_str(pred_k_future))
        dist_n_lev = Levenshtein.distance(to_str(gt_future), to_str(pred_n_future))
        
        # DTW
        dist_k_dtw = calc_dtw(gt_future, pred_k_future)
        dist_n_dtw = calc_dtw(gt_future, pred_n_future)
        
        results.append({
            'id': agent_id,
            'type': data['type'],
            'score_k_lev': dist_k_lev / pred_len,
            'score_n_lev': dist_n_lev / pred_len,
            'score_k_dtw': dist_k_dtw / pred_len,
            'score_n_dtw': dist_n_dtw / pred_len,
            'prompt': prompt_seq,
            'gt': gt_future,
            'pred_k': pred_k_future,
            'pred_n': pred_n_future
        })
        
        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/{len(test_data)}...", end='\r')
            
    return pd.DataFrame(results)

# =========================================================
# 5. 可視化ロジック (変更なし)
# =========================================================
def decode_for_plot(seq):
    arr = np.array(seq, dtype=float)
    is_stay = (arr >= STAY_OFFSET) & (arr < PAD_TOKEN)
    arr[is_stay] -= STAY_OFFSET
    arr[arr == PAD_TOKEN] = np.nan
    return arr

def plot_results(df_res):
    if df_res.empty:
        print("No results to plot.")
        return

    df_res['diff'] = df_res['score_n_dtw'] - df_res['score_k_dtw']
    target_types = ['stopover', 'wander']
    
    for b_type in target_types:
        df_type = df_res[df_res['type'] == b_type]
        if len(df_type) == 0: continue

        k_wins = df_type[df_type['diff'] > 0.05].sort_values(by='diff', ascending=False)
        if not k_wins.empty:
            plot_single(k_wins.iloc[0], "Koopman_Win")
            
        n_wins = df_type[df_type['diff'] < -0.05].sort_values(by='diff', ascending=True)
        if not n_wins.empty:
            plot_single(n_wins.iloc[0], "Normal_Win")

def plot_single(row, title_prefix):
    plt.figure(figsize=(12, 5))
    prompt_len = len(row['prompt'])
    full_gt = decode_for_plot(row['prompt'] + row['gt'])
    full_k  = decode_for_plot(row['prompt'] + row['pred_k'])
    full_n  = decode_for_plot(row['prompt'] + row['pred_n'])
    
    plt.plot(full_gt, label='Ground Truth', color='black', linewidth=3, alpha=0.3)
    plt.axvline(x=prompt_len-0.5, color='gray', linestyle='--')
    
    x_range = range(prompt_len, len(full_gt))
    len_k = min(len(x_range), len(full_k[prompt_len:]))
    len_n = min(len(x_range), len(full_n[prompt_len:]))
    
    plt.plot(list(x_range)[:len_k], full_k[prompt_len:][:len_k], label=f'Koopman ({row["score_k_dtw"]:.2f})', color='red')
    plt.plot(list(x_range)[:len_n], full_n[prompt_len:][:len_n], label=f'Normal ({row["score_n_dtw"]:.2f})', color='blue', linestyle=':')
    
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
        # exit() 
        
    # 2. データ生成 (AgentV3)
    # 学習時より多いエージェント数(250)で生成し、汎化性能も見れる
    gt_data = generate_ground_truth_test(num_agents=250, max_steps=60)
    save_log(f"Generated {len(gt_data)} trajectories (Activity Need Model).")
    
    # 3. 評価
    if gt_data:
        df_res = evaluate_models(model_koopman, model_normal, gt_data, prompt_len=15, device=device)
        
        csv_path = os.path.join(out_dir, "evaluation_results.csv")
        df_res.to_csv(csv_path, index=False)
        
        save_log("\n=== Scores (Lower is Better) ===")
        save_log(df_res[['score_k_lev', 'score_n_lev']].mean().to_string())
        save_log(df_res[['score_k_dtw', 'score_n_dtw']].mean().to_string())

        save_log("\n=== By Type ===")
        save_log(df_res.groupby('type')[['score_k_lev', 'score_n_lev']].mean().to_string())
        save_log(df_res.groupby('type')[['score_k_dtw', 'score_n_dtw']].mean().to_string())
        
        plot_results(df_res)
        save_log("\nDone.")
    else:
        save_log("No valid test data generated.")