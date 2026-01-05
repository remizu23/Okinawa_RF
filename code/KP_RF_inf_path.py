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

# モデルパス (適宜変更してください)
MODEL_KOOPMAN_PATH = "/home/mizutani/projects/RF/runs/20260105_164148/model_weights_20260105_164148.pth"
MODEL_NORMAL_PATH  = "/home/mizutani/projects/RF/runs/20260105_164632/model_weights_20260105_164632.pth"

# =========================================================
# 0. 保存先設定
# =========================================================
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"/home/mizutani/projects/RF/runs/comparison_eval_v3_Arandom_{run_id}"
os.makedirs(out_dir, exist_ok=True)

print(f"=== Evaluation Started: {run_id} ===")
def save_log(msg):
    print(msg)
    with open(os.path.join(out_dir, "evaluation_log.txt"), "a") as f:
        f.write(msg + "\n")

# =========================================================
# 1. 環境設定 & マップ構築
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
# シード固定で毎回同じ環境にする
rng_env = np.random.default_rng(42)
NODE_BETAS = {i: 0.0 for i in range(NUM_NODES)}
for node in AREA_SHOP:
    NODE_BETAS[node] = round(rng_env.uniform(0.5, 1.5), 2)

# =========================================================
# 2. AgentV3 クラス (活動欲求モデル)
# =========================================================
class AgentV3:
    def __init__(self, agent_id, graph, behavior_type):
        self.id = agent_id
        self.graph = graph
        self.type = behavior_type
        
        self.history_tokens = []    # トークン化された履歴
        self.finished = False
        
        # --- 活動欲求パラメータ ---
        # Stopoverはすぐ満足、Wanderは長く回る
        # if self.type == 'stopover':
        #     self.satisfaction_threshold = random.uniform(5.0, 15.0)
        # else: # wander
        #     self.satisfaction_threshold = random.uniform(20.0, 35.0)
    
        self.satisfaction_threshold = random.uniform(20.0, 35.0)    
        self.current_satisfaction = 0.0
        
        # --- 初期位置 ---
        self.start_node = random.choice(ENTRY_EXIT_NODES)
        self.current_node = self.start_node
        self.history_tokens.append(self.current_node)
        
        # --- 状態管理 ---
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
        
        # --- 1. 滞在中 (STAY) ---
        if self.state == 'STAY':
            beta = NODE_BETAS.get(self.current_node, 0)
            self.current_satisfaction += beta
            
            # トークン生成: 滞在継続中は +19
            token = self.current_node + STAY_OFFSET
            self.history_tokens.append(token)
            
            # 離脱判定
            if self.current_satisfaction >= self.satisfaction_threshold:
                self.state = 'WALK'
                self.target = self._choose_exit()
            elif self.current_satisfaction >= self.local_sat_goal:
                self.state = 'WALK'
                self.target = self._choose_next_shop()
            
            return token

        # --- 2. 移動中 (WALK) ---
        elif self.state == 'WALK':
            if self.current_node == self.target:
                # 帰宅完了
                if self.current_satisfaction >= self.satisfaction_threshold:
                    self.finished = True
                    self.state = 'FINISHED'
                    return PAD_TOKEN
                
                # 店に到着 -> 滞在開始
                else:
                    self.state = 'STAY'
                    remaining = self.satisfaction_threshold - self.current_satisfaction
                    gain = remaining * random.uniform(0.3, 0.6)
                    gain = max(gain, 1.0)
                    self.local_sat_goal = self.current_satisfaction + gain
                    
                    # ★到着したこのターンは「移動の終わり」かつ「滞在の初手」
                    # トークンは生ID (滞在初手)
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
    """
    AgentV3を使用してテストデータを生成する。
    すでに正しいトークン列(滞在分離済み)が得られるため、変換処理は不要。
    """
    random.seed(999) 
    np.random.seed(999)
    test_data = []
    types = ['stopover', 'wander'] # Throughは除外
    
    for i in range(num_agents):
        b_type = random.choice(types)
        agent = AgentV3(i, G, b_type)
        
        # シミュレーション実行 (MAX_STEPS回)
        # 初期位置はinitで入っているので、残りステップ分回す
        for _ in range(max_steps - 1):
            token = agent.step()
            if token == PAD_TOKEN and agent.finished:
                break
        
        # エージェントが生成したトークン列をそのまま使用
        # パディングは除去して、純粋な軌跡を評価用データとする
        raw_traj = [t for t in agent.history_tokens if t != PAD_TOKEN]
        
        # パディング除去の結果、短すぎるデータは除外してもよいが今回は含める
        if len(raw_traj) > 5:
            test_data.append({
                'agent_id': i,
                'type': b_type,
                'trajectory': raw_traj
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
        'vocab_size': VOCAB_SIZE, 
        'token_emb_dim': 64, 'd_model': 64, 'nhead': 4, 
        'num_layers': 6, 'd_ff': 128, 'z_dim': 16, 
        'pad_token_id': PAD_TOKEN
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
    to_str = lambda seq: "".join([chr(x + 48) for x in seq]) 

    for i, data in enumerate(test_data):
        full_traj = data['trajectory']
        
        # プロンプトより短いデータはスキップ
        if len(full_traj) <= prompt_len: continue
            
        prompt_seq = full_traj[:prompt_len]
        gt_future = full_traj[prompt_len:]
        pred_len = len(gt_future)
        
        # 推論
        pred_k_future = predict_trajectory(model_koopman, prompt_seq, pred_len, device)
        pred_n_future = predict_trajectory(model_normal, prompt_seq, pred_len, device)
        
        # 編集距離計算
        dist_k = Levenshtein.distance(to_str(gt_future), to_str(pred_k_future))
        dist_n = Levenshtein.distance(to_str(gt_future), to_str(pred_n_future))
        
        score_k = dist_k / len(gt_future)
        score_n = dist_n / len(gt_future)
        
        results.append({
            'id': data['agent_id'],
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
# 5. 可視化ロジック
# =========================================================
def decode_for_plot(seq):
    """
    プロット用にトークンを物理的な位置(Y軸)に戻す
    """
    arr = np.array(seq, dtype=float)
    
    # Stay(19-37) -> 元のID(0-18)
    is_stay = (arr >= STAY_OFFSET) & (arr < PAD_TOKEN)
    arr[is_stay] -= STAY_OFFSET
    
    # Pad -> NaN
    arr[arr == PAD_TOKEN] = np.nan
    return arr

def plot_results(df_res):
    if df_res.empty:
        print("No results to plot.")
        return

    df_res['diff'] = df_res['score_n'] - df_res['score_k']
    target_types = ['stopover', 'wander']
    
    for b_type in target_types:
        df_type = df_res[df_res['type'] == b_type]
        if len(df_type) == 0: continue

        # Koopman Wins (差が大きい順に上位1つ)
        k_wins = df_type[df_type['diff'] > 0.05].sort_values(by='diff', ascending=False)
        if not k_wins.empty:
            plot_single(k_wins.iloc[0], "Koopman_Win")
            
        # Normal Wins
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
    # 長さが合わない場合のガード（可視化でエラーにならないように）
    len_k = min(len(x_range), len(full_k[prompt_len:]))
    len_n = min(len(x_range), len(full_n[prompt_len:]))
    
    plt.plot(list(x_range)[:len_k], full_k[prompt_len:][:len_k], label=f'Koopman ({row["score_k"]:.2f})', color='red')
    plt.plot(list(x_range)[:len_n], full_n[prompt_len:][:len_n], label=f'Normal ({row["score_n"]:.2f})', color='blue', linestyle=':')
    
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
    # ※パスは適宜自分の環境に合わせてください
    try:
        model_koopman, _ = load_model(MODEL_KOOPMAN_PATH, device)
        model_normal, _ = load_model(MODEL_NORMAL_PATH, device)
    except Exception as e:
        save_log(f"Model Load Error: {e}")
        # モデルがない場合はダミーで進める（動作確認用）
        # exit() 
        
    # 2. データ生成 (AgentV3)
    # 500エージェント分生成
    gt_data = generate_ground_truth_test(num_agents=500, max_steps=60)
    save_log(f"Generated {len(gt_data)} trajectories (Activity Need Model).")
    
    # 3. 評価
    # prompt_len: 最初の15ステップを与えて、残りを予測させる
    if gt_data:
        df_res = evaluate_models(model_koopman, model_normal, gt_data, prompt_len=15, device=device)
        
        # 4. 集計
        csv_path = os.path.join(out_dir, "evaluation_results.csv")
        df_res.to_csv(csv_path, index=False)
        
        save_log("\n=== Scores (Lower is Better) ===")
        save_log(df_res[['score_k', 'score_n']].mean().to_string())
        save_log("\n=== By Type ===")
        save_log(df_res.groupby('type')[['score_k', 'score_n']].mean().to_string())
        
        # 5. プロット
        plot_results(df_res)
        save_log("\nDone.")
    else:
        save_log("No valid test data generated.")