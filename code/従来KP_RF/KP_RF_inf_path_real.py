import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import Levenshtein
from tqdm import tqdm
from datetime import datetime
import sys
from types import ModuleType
from KP_RF import KoopmanRoutesFormer
from network import Network, expand_adjacency_matrix

# =========================================================
# 1. Configuration
# =========================================================
CONFIG = {
    "gpu_id": 0,
    "pad_token": 38,
    "vocab_size": 42, 
    "stay_offset": 19,
    
    # ★★★ 評価モード設定 ★★★
    # "VAL_SPLIT": 学習用ファイルから検証データ(20%)を抽出して評価
    # "TEST_FILE": テスト用ファイルを全件評価
    "eval_mode": "VAL_SPLIT",  # <--- ここを変更してください ("VAL_SPLIT" or "TEST_FILE")

    # (A) "VAL_SPLIT" モード用のファイルパス (学習に使ったもの)
    # "train_npz_path": "/home/mizutani/projects/RF/data/input_real_m4_emb.npz", 
    "train_npz_path": "/home/mizutani/projects/RF/data/input_real_m5.npz", 
    
    # (B) "TEST_FILE" モード用のファイルパス (テスト専用)
    "test_npz_path": "/home/mizutani/projects/RF/data/input_real_test_m4_emb.npz",

    # モデルパス
    "model_koopman_path": "/home/mizutani/projects/RF/runs/20260124_214854/model_weights_20260124_214854.pth",
    "model_ablation_path": "/home/mizutani/projects/RF/runs/20260124_184039/model_weights_20260124_184039.pth",
    # 出力設定⭐︎変更忘れない！
    "output_dir": "/home/mizutani/projects/RF/runs/20260124_214854/eval_m5val_自己回帰",
    "plot_max_samples": 1000,

    "val_indices_path": "/home/mizutani/projects/RF/data/common_val_m5.npy",

    # Context Logic
    "holidays": [20240928, 20240929, 20251122, 20251123],
    "night_start": 19, 
    "night_end": 2,
    "events": [
        (20240929, 9, 16, [14]),
        (20251122, 10, 19, [2, 11]),
        (20251123, 10, 16, [2])
    ],
}

# 2-hop Adjacency Map
ADJACENCY_MAP = {
    0: [1, 2, 4, 11], 1: [0, 2, 4, 5, 9], 2: [0, 1, 5, 6, 7],
    4: [0, 1, 5, 8, 9, 10, 11], 5: [1, 2, 4, 6, 10], 6: [2, 5, 7, 10, 14],
    7: [2, 6, 13, 14, 15], 8: [4, 9, 11], 9: [1, 4, 8, 10, 12],
    10: [4, 5, 6, 9, 12, 13], 11: [0, 4, 8], 12: [9, 10, 13],
    13: [7, 10, 12, 14, 15], 14: [6, 7, 13, 15, 16], 15: [7, 13, 14],
    16: [14, 17, 18], 17: [16, 18], 18: [16, 17]
}

# =========================================================
# 2. Helper Functions
# =========================================================
# =========================================================
# 追加: 滞在評価用ヘルパー関数
# =========================================================
def get_stay_events(seq, stay_offset=19, pad_token=38):
    """
    数列から滞在イベントを抽出する
    Returns: list of dict {'start': int, 'end': int, 'node': int, 'dur': int}
    """
    events = []
    n = len(seq)
    i = 0
    while i < n:
        token = seq[i]
        # 定義A: トークンIDが stay_offset(19)以上 pad_token(38)未満なら滞在
        if stay_offset <= token < pad_token:
            start = i
            node_id = token - stay_offset
            # 同じ滞在トークンが続く限りループ
            while i < n and seq[i] == token:
                i += 1
            end = i # endはexclusive
            duration = end - start
            events.append({
                'start': start,
                'end': end,
                'node': node_id,
                'dur': duration
            })
        else:
            i += 1
    return events

def calc_stay_metrics_pair(gt_seq, pred_seq, node_dists):
    """
    GTと予測の滞在イベントをマッチングし、指標を計算する (定義B & C)
    """
    gt_events = get_stay_events(gt_seq)
    pred_events = get_stay_events(pred_seq)
    
    results = []
    
    # GTの各滞在に対して、予測側で重複しているものを探す
    for gt_e in gt_events:
        best_match = None
        max_overlap = 0
        
        for pred_e in pred_events:
            # 時間的重複 (Overlap) の計算
            overlap_start = max(gt_e['start'], pred_e['start'])
            overlap_end = min(gt_e['end'], pred_e['end'])
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > 0:
                # 最も長く重複しているものを採用
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_match = pred_e
        
        # マッチ結果の記録
        metric = {
            'detected': False,
            'len_diff': None,    # 定義C: Pred - GT
            'loc_dist': None,    # 地理的距離 (Hop数)
            'gt_dur': gt_e['dur'],
            'gt_node': gt_e['node']
        }
        
        if best_match:
            metric['detected'] = True
            metric['len_diff'] = best_match['dur'] - gt_e['dur'] # 数値(差分)
            
            # 場所の距離計算 (NODE_DISTANCESを使用)
            u, v = gt_e['node'], best_match['node']
            if u == v:
                dist = 0
            elif u in node_dists and v in node_dists[u]:
                dist = node_dists[u][v]
            else:
                # 辞書にない場合は最大ペナルティまたは遠方扱い(便宜上10とするかinf)
                dist = 999 
            metric['loc_dist'] = dist
        else:
            # 検出できなかった場合 (見逃し)
            # 差分は「完全に短かった」として扱うなら -gt_dur ですが、
            # ここでは集計時に区別できるよう None または 特殊値を推奨
            metric['len_diff'] = -gt_e['dur'] # 0 - GT = 負の値
            metric['loc_dist'] = None       # 場所は評価不能
            
        results.append(metric)
        
    return results
    
def build_distance_matrix(adj_map):
    G = nx.Graph()
    for u, neighbors in adj_map.items():
        for v in neighbors:
            G.add_edge(u, v)
    return dict(nx.all_pairs_shortest_path_length(G))

NODE_DISTANCES = build_distance_matrix(ADJACENCY_MAP)

def get_node_id(token):
    if token == CONFIG["pad_token"]: return -1
    if token >= CONFIG["stay_offset"] and token < CONFIG["pad_token"]:
        return token - CONFIG["stay_offset"]
    if token < CONFIG["stay_offset"]:
        return token
    return -1

def get_geo_cost(t1, t2):
    n1, n2 = get_node_id(t1), get_node_id(t2)
    if n1 == -1 or n2 == -1: return 1.0
    if n1 == n2: return 0.0
    try:
        dist = NODE_DISTANCES[n1][n2]
    except KeyError:
        return 1.0
    if dist == 1: return 0.3
    elif dist == 2: return 0.6
    else: return 1.0

class ContextDeterminer:
    def __init__(self, config):
        self.config = config
    
    def get_holiday(self, timestamp_int):
        date_int = timestamp_int // 10000
        return 1 if date_int in self.config["holidays"] else 0

    def get_timezone(self, timestamp_int):
        hour = (timestamp_int // 100) % 100
        if hour >= self.config["night_start"] or hour < self.config["night_end"]:
            return 1
        return 0

    def get_event(self, timestamp_int, current_token):
        current_node = get_node_id(current_token)
        if current_node == -1: return 0
        date_int = timestamp_int // 10000
        hour = (timestamp_int // 100) % 100
        for (e_date, s_h, e_h, nodes) in self.config["events"]:
            if date_int == e_date:
                if s_h <= hour < e_h:
                    if current_node in nodes:
                        return 1
        return 0

# =========================================================
# 3. Evaluator Class
# =========================================================
class ModelEvaluator:
    def __init__(self, model, device, context_logic, is_koopman=True):
        self.model = model
        self.device = device
        self.context_logic = context_logic
        self.is_koopman = is_koopman
        self.model.eval()

    def calculate_stay_counts(self, tokens):
        b, t = tokens.shape
        out = torch.zeros_like(tokens)
        tokens_np = tokens.cpu().numpy()
        pad_id = CONFIG["pad_token"]
        special_ids = [pad_id, pad_id+1, pad_id+2, pad_id+3]
        for i in range(b):
            cnt = 0
            curr = -1
            for j in range(t):
                val = tokens_np[i, j]
                if val in special_ids:
                    cnt = 0; curr = -1
                else:
                    if val == curr: cnt += 1
                    else: cnt = 1; curr = val
                out[i, j] = cnt
        return out.to(self.device)

    def get_embeddings(self, token, stay_count, agent_id, holiday, timezone, event):
        token_t = torch.tensor([[token]], device=self.device)
        stay_t  = torch.tensor([[stay_count]], device=self.device)
        agent_t = torch.tensor([agent_id], device=self.device).unsqueeze(0)
        hol_t   = torch.tensor([[holiday]], device=self.device)
        tz_t    = torch.tensor([[timezone]], device=self.device)
        evt_t   = torch.tensor([[event]], device=self.device)
        u_vec = torch.cat([
            self.model.token_embedding(token_t),
            self.model.stay_embedding(stay_t),
            self.model.agent_embedding(agent_t),
            self.model.holiday_embedding(hol_t),
            self.model.time_zone_embedding(tz_t),
            self.model.event_embedding(evt_t)
        ], dim=-1)
        return u_vec

    def predict_next_step_metrics(self, prompt_seq, prompt_contexts, target_token):
        tokens = torch.tensor([prompt_seq], dtype=torch.long, device=self.device)
        h_b = torch.tensor([prompt_contexts['h']], dtype=torch.long, device=self.device)
        tz_b = torch.tensor([prompt_contexts['tz']], dtype=torch.long, device=self.device)
        e_b = torch.tensor([prompt_contexts['e']], dtype=torch.long, device=self.device)
        stay_counts = self.calculate_stay_counts(tokens)
        
        a_val = prompt_contexts['agent']
        num_agents_model = self.model.agent_embedding.num_embeddings
        a_val = a_val % num_agents_model 
        agent_id = torch.tensor([a_val], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits, _, _, _ = self.model(
                tokens, stay_counts, agent_id, h_b, tz_b, e_b
            )
        
        last_logits = logits[0, -1, :]
        probs = F.softmax(last_logits, dim=0)
        pred_token = torch.argmax(probs).item()
        
        is_correct = (pred_token == target_token)
        likelihood = probs[target_token].item()
        return is_correct, likelihood

    ###=====================###
    ### 0125 koopman側も逐次transformerに入れて自己回帰生成するver作成 ###
    ###=====================###
    def generate_trajectory(self, prompt_seq, start_time, agent_id_raw, gen_len):
        # 1. 初期プロンプトのコンテキスト作成
        # プロンプト系列全体の時刻依存コンテキストを用意する
        h_seq = []
        tz_seq = []
        e_seq = []
        
        # 時刻計算用 (1分刻みと仮定するか、データの仕様に合わせる)
        # ※簡易的に「系列長分だけ進む」あるいは「固定」など、Ablation側のロジックに準拠します
        # 厳密には prompt_seq の各時点の時刻が必要ですが、
        # ここでは start_time からの相対経過、あるいは固定値(h_val, tz_val)を使っている既存ロジックを踏襲します。
        
        h_val = self.context_logic.get_holiday(start_time)
        tz_val = self.context_logic.get_timezone(start_time)
        
        # プロンプト部分のコンテキスト配列作成
        prompt_h = [h_val] * len(prompt_seq)
        prompt_tz = [tz_val] * len(prompt_seq)
        prompt_e = [self.context_logic.get_event(start_time, t) for t in prompt_seq]
        
        # 2. Tensor化 (Batchサイズ 1)
        curr_seq = torch.tensor([prompt_seq], dtype=torch.long, device=self.device)
        curr_h = torch.tensor([prompt_h], dtype=torch.long, device=self.device)
        curr_tz = torch.tensor([prompt_tz], dtype=torch.long, device=self.device)
        curr_e = torch.tensor([prompt_e], dtype=torch.long, device=self.device)
        
        # エージェントID
        num_agents_model = self.model.agent_embedding.num_embeddings
        safe_agent_id = agent_id_raw % num_agents_model
        agent_in = torch.tensor([safe_agent_id], dtype=torch.long, device=self.device)
        
        generated_seq = []
        
        # 3. 生成ループ (Autoregressive)
        # KoopmanモデルもAblationモデルも、ここで「逐次Transformer」を行う
        for _ in range(gen_len):
            # 現在の系列から滞在カウントを計算
            curr_stay = self.calculate_stay_counts(curr_seq)
            
            # Forward Pass
            # Koopmanモデルの場合: 
            #   内部で Transformer -> z_t -> A z_t (+ B u_t) -> z_t+1 -> logits
            #   という計算が行われ、"次の1ステップ" が予測される。
            logits, _, _, _ = self.model(
                curr_seq, curr_stay, agent_in, curr_h, curr_tz, curr_e
            )
            
            # 次のトークンを決定 (Greedy)
            # ※確率的サンプリングが必要ならここを修正
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            generated_seq.append(next_token)
            
            # 4. 次のステップへの入力作成
            # トークン追加
            next_tens = torch.tensor([[next_token]], device=self.device)
            curr_seq = torch.cat([curr_seq, next_tens], dim=1)
            
            # コンテキスト追加
            # 厳密な時刻更新が必要ならここで start_time を更新するが、
            # 既存コードに従い、Holiday/Timezoneは固定、Eventはノード依存で判定
            e_val = self.context_logic.get_event(start_time, next_token)
            
            curr_h = torch.cat([curr_h, torch.tensor([[h_val]], device=self.device)], dim=1)
            curr_tz = torch.cat([curr_tz, torch.tensor([[tz_val]], device=self.device)], dim=1)
            curr_e = torch.cat([curr_e, torch.tensor([[e_val]], device=self.device)], dim=1)

        return generated_seq
    # def generate_trajectory(self, prompt_seq, start_time, agent_id_raw, gen_len):
    #     h_val = self.context_logic.get_holiday(start_time)
    #     tz_val = self.context_logic.get_timezone(start_time)
        
    #     prompt_h = [h_val] * len(prompt_seq)
    #     prompt_tz = [tz_val] * len(prompt_seq)
    #     prompt_e = [self.context_logic.get_event(start_time, t) for t in prompt_seq]
        
    #     tokens_in = torch.tensor([prompt_seq], dtype=torch.long, device=self.device)
    #     h_in = torch.tensor([prompt_h], dtype=torch.long, device=self.device)
    #     tz_in = torch.tensor([prompt_tz], dtype=torch.long, device=self.device)
    #     e_in = torch.tensor([prompt_e], dtype=torch.long, device=self.device)
    #     stay_counts_in = self.calculate_stay_counts(tokens_in)

    #     num_agents_model = self.model.agent_embedding.num_embeddings
    #     safe_agent_id = agent_id_raw % num_agents_model
    #     agent_in = torch.tensor([safe_agent_id], dtype=torch.long, device=self.device)
        
    #     generated_seq = []
        
    #     with torch.no_grad():
    #         logits, z_hat_seq, _, _ = self.model(
    #             tokens_in, stay_counts_in, agent_in, h_in, tz_in, e_in
    #         )
            
    #         if self.is_koopman:
    #             z_curr = z_hat_seq[:, -1, :]
    #             current_token = prompt_seq[-1]
    #             current_stay_count = stay_counts_in[0, -1].item()
                
    #             for _ in range(gen_len):
    #                 e_val = self.context_logic.get_event(start_time, current_token)
    #                 u_t = self.get_embeddings(
    #                     current_token, current_stay_count, safe_agent_id,
    #                     h_val, tz_val, e_val
    #                 ).squeeze(1)
                    
    #                 ###=====================###
    #                 ###=Butの入力省略変更箇所1=###
    #                 ###=====================###

    #                 term_A = torch.matmul(z_curr, self.model.A.t())
    #                 term_B = torch.matmul(u_t, self.model.B.t())
    #                 z_next = term_A + term_B
                    
    #                 logits_next = self.model.to_logits(z_next)
    #                 next_token = torch.argmax(logits_next, dim=-1).item()
    #                 generated_seq.append(next_token)
                    
    #                 if next_token == current_token:
    #                     current_stay_count += 1
    #                     if current_stay_count >= self.model.stay_embedding.num_embeddings:
    #                         current_stay_count = self.model.stay_embedding.num_embeddings - 1
    #                 else:
    #                     current_stay_count = 1
    #                 current_token = next_token
    #                 z_curr = z_next
    #         else:
    #             curr_seq = tokens_in
    #             curr_h = h_in
    #             curr_tz = tz_in
    #             curr_e = e_in
    #             curr_stay = stay_counts_in
                
    #             for _ in range(gen_len):
    #                 logits, _, _, _ = self.model(
    #                     curr_seq, curr_stay, agent_in, curr_h, curr_tz, curr_e
    #                 )
    #                 next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
    #                 generated_seq.append(next_token)
                    
    #                 next_tens = torch.tensor([[next_token]], device=self.device)
    #                 curr_seq = torch.cat([curr_seq, next_tens], dim=1)
                    
    #                 e_val = self.context_logic.get_event(start_time, next_token)
    #                 curr_h = torch.cat([curr_h, torch.tensor([[h_val]], device=self.device)], dim=1)
    #                 curr_tz = torch.cat([curr_tz, torch.tensor([[tz_val]], device=self.device)], dim=1)
    #                 curr_e = torch.cat([curr_e, torch.tensor([[e_val]], device=self.device)], dim=1)
    #                 curr_stay = self.calculate_stay_counts(curr_seq)
    #     return generated_seq

def calc_dtw(seq1, seq2, geo=False):
    n, m = len(seq1), len(seq2)
    dtw = np.full((n + 1, m + 1), float('inf'))
    dtw[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if geo: cost = get_geo_cost(seq1[i-1], seq2[j-1])
            else: cost = 0.0 if seq1[i-1] == seq2[j-1] else 1.0
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    return dtw[n, m]

def calc_levenshtein(seq1, seq2, geo=False):
    if not geo:
        s1 = "".join([chr(x+100) for x in seq1])
        s2 = "".join([chr(x+100) for x in seq2])
        return Levenshtein.distance(s1, s2)
    else:
        n, m = len(seq1), len(seq2)
        dp = np.zeros((n + 1, m + 1))
        for i in range(n + 1): dp[i, 0] = i
        for j in range(m + 1): dp[0, j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                sub_cost = get_geo_cost(seq1[i-1], seq2[j-1])
                dp[i, j] = min(dp[i-1, j]+1, dp[i, j-1]+1, dp[i-1, j-1]+sub_cost)
        return dp[n, m]

# =========================================================
# 5. Main Execution
# =========================================================
def load_weights(path, device, is_koopman):
    checkpoint = torch.load(path, map_location=device)
    conf = checkpoint['config']
    state_dict = checkpoint['model_state_dict']
    
    if 'agent_embedding.weight' in state_dict:
        detected_num_agents = state_dict['agent_embedding.weight'].shape[0]
        print(f"[{os.path.basename(path)}] Detected num_agents: {detected_num_agents}")
    else:
        detected_num_agents = 1
        print(f"[{os.path.basename(path)}] Agent embedding not found, using default: 1")

    model = KoopmanRoutesFormer(
        vocab_size=conf.get('vocab_size', CONFIG["vocab_size"]),
        token_emb_dim=conf['token_emb_dim'],
        d_model=conf['d_model'],
        nhead=conf['nhead'],
        num_layers=conf['num_layers'],
        d_ff=conf['d_ff'],
        z_dim=conf['z_dim'],
        pad_token_id=conf.get('pad_token_id', CONFIG["pad_token"]),
        num_agents=detected_num_agents, 
        agent_emb_dim=conf.get('agent_emb_dim', 16),
        stay_emb_dim=conf.get('stay_emb_dim', 16),
        holiday_emb_dim=conf.get('holiday_emb_dim', 4),
        time_zone_emb_dim=conf.get('time_zone_emb_dim', 4),
        event_emb_dim=conf.get('event_emb_dim', 4),
        dist_mat_base=None 
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model

def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    device = torch.device(f"cuda:{CONFIG['gpu_id']}" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Started: {CONFIG['output_dir']}")
    print(f"Mode: {CONFIG['eval_mode']}")
    print(f"Using Device: {device}")

    # =====================================================
    # 1. Data Loading Switch Logic
    # =====================================================
    route_arr, time_arr, agent_ids = None, None, None
    
    if CONFIG["eval_mode"] == "TEST_FILE":
        # (A) テスト用ファイルを直接使用
        dpath = CONFIG["test_npz_path"]
        print(f"Loading TEST file: {dpath}")
        if not os.path.exists(dpath):
            print(f"Error: File not found {dpath}")
            return
        
        data = np.load(dpath)
        route_arr = data['route_arr']
        time_arr = data['time_arr']
        if 'agent_ids' in data:
            agent_ids = data['agent_ids']
        else:
            agent_ids = np.zeros(len(route_arr), dtype=int)
            
        print(f"Loaded {len(route_arr)} test samples.")
        
    elif CONFIG["eval_mode"] == "VAL_SPLIT":
        # (B) 学習用ファイルからValidation Splitを再現
        dpath = CONFIG["train_npz_path"]
        print(f"Loading TRAIN file for validation split: {dpath}")
        if not os.path.exists(dpath):
            print(f"Error: File not found {dpath}")
            return
            
        data = np.load(dpath)
        route_arr_full = data['route_arr']
        time_arr_full = data['time_arr']
        if 'agent_ids' in data:
            agent_ids_full = data['agent_ids']
        else:
            agent_ids_full = np.zeros(len(route_arr_full), dtype=int)

        val_indices_path = CONFIG.get("val_indices_path", None)

        # チェックポイントからインデックスを取得
        # print(f"Retrieving validation indices from checkpoint...")
        # checkpoint = torch.load(CONFIG["model_koopman_path"], map_location=device)
        # if "val_indices" not in checkpoint:
        #     print("Error: 'val_indices' not found in checkpoint.")
        #     return
        
        # val_indices = checkpoint["val_indices"]
        print(f"Retrieving validation indices from checkpoint...")

        # 1) まず固定ファイルがあればそれを使う
        if val_indices_path is not None and os.path.exists(val_indices_path):
            val_indices = np.load(val_indices_path)
            print(f"Loaded fixed val_indices from: {val_indices_path} (N={len(val_indices)})")

        # 2) なければ Koopman ckpt から取得して保存する
        else:
            checkpoint = torch.load(CONFIG["model_koopman_path"], map_location=device)
            if "val_indices" not in checkpoint:
                print("Error: 'val_indices' not found in checkpoint.")
                return

            val_indices = checkpoint["val_indices"]
            if isinstance(val_indices, torch.Tensor):
                val_indices = val_indices.cpu().numpy()

            if val_indices_path is not None:
                os.makedirs(os.path.dirname(val_indices_path), exist_ok=True)
                np.save(val_indices_path, val_indices)
                print(f"Saved fixed val_indices to: {val_indices_path} (N={len(val_indices)})")

            
        route_arr = route_arr_full[val_indices]
        time_arr = time_arr_full[val_indices]
        agent_ids = agent_ids_full[val_indices]
        
        print(f"Extracted {len(route_arr)} validation samples.")
        
    else:
        print(f"Error: Unknown eval_mode {CONFIG['eval_mode']}")
        return

    # =====================================================
    # 2. Model & Evaluator Setup
    # =====================================================
    try:
        model_koopman = load_weights(CONFIG["model_koopman_path"], device, is_koopman=True)
        model_ablation = load_weights(CONFIG["model_ablation_path"], device, is_koopman=False)
    except Exception as e:
        print(f"Model Load Error: {e}")
        return

    ctx_logic = ContextDeterminer(CONFIG)
    eval_k = ModelEvaluator(model_koopman, device, ctx_logic, is_koopman=True)
    eval_a = ModelEvaluator(model_ablation, device, ctx_logic, is_koopman=False)
    
    metrics_list = []
    test_indices = range(len(route_arr))
    prompt_len = 5
    
    # =====================================================
    # 3. Evaluation Loop
    # =====================================================
    print("Running evaluation loop...")
    for idx in tqdm(test_indices):
        full_seq = [int(x) for x in route_arr[idx] if x != CONFIG["pad_token"]]
        
        if len(full_seq) <= prompt_len + 3: 
            continue
            
        start_time = int(time_arr[idx])
        agent_id = int(agent_ids[idx])
        
        prompt_seq = full_seq[:prompt_len]
        gt_future = full_seq[prompt_len:]
        
        # Task 1
        prompt_ctx = {
            'h': [ctx_logic.get_holiday(start_time)] * prompt_len,
            'tz': [ctx_logic.get_timezone(start_time)] * prompt_len,
            'e': [ctx_logic.get_event(start_time, t) for t in prompt_seq],
            'agent': agent_id
        }
        target_token = gt_future[0]
        
        acc_k, like_k = eval_k.predict_next_step_metrics(prompt_seq, prompt_ctx, target_token)
        acc_a, like_a = eval_a.predict_next_step_metrics(prompt_seq, prompt_ctx, target_token)
        
        # Task 2
        gen_len = len(gt_future)
        pred_k = eval_k.generate_trajectory(prompt_seq, start_time, agent_id, gen_len)
        pred_a = eval_a.generate_trajectory(prompt_seq, start_time, agent_id, gen_len)
        
        # ===============================================
        # 追加: 滞在指標の計算
        # ===============================================
        # GT系列全体における滞在を評価対象とするか、生成部分のみとするか？
        # 一般的に長期生成テストなら「生成部分(gt_future)」と比較します。
        
        stay_metrics_k = calc_stay_metrics_pair(gt_future, pred_k, NODE_DISTANCES)
        stay_metrics_a = calc_stay_metrics_pair(gt_future, pred_a, NODE_DISTANCES)

        # 1つのサンプルに複数の滞在が含まれる可能性があるため、リスト構造で保持するか、
        # ここで「平均」を取ってDataFrameに入れるかですが、詳細分析のため
        # フラットなリスト（stay_metrics_list）を別途作るのがおすすめです。

        # ここではDataFrame用のmetrics_listに、サンプルごとの平均値として埋め込みます
        # (詳細な分布を見たい場合は別途 stay_results を保存してください)
        
        # ===============================================
        # 修正: 統合コスト計算を含む集計関数
        # ===============================================
        def summarize_stay(metrics_list, alpha=1.5, dist_thresh=3):
            """
            alpha: 空間誤差1ホップを、時間誤差何ステップ分に換算するか (1.5)
            dist_thresh: これを超えて場所が離れていたら、時間誤差を無視してペナルティ化 (Pattern Y)
            """
            if not metrics_list: return None, None, None, None, None
            
            # 検出できたものだけを対象
            detected = [m for m in metrics_list if m['detected']]
            if not detected: return 0.0, None, None, None, None
            
            det_rate = len(detected) / len(metrics_list)
            
            # 1. 既存指標
            diffs = [m['len_diff'] for m in detected]
            dists = [m['loc_dist'] for m in detected]
            
            mean_len_diff = np.mean(diffs)          # Bias
            mean_len_abs  = np.mean(np.abs(diffs))  # Abs Error
            mean_loc_dist = np.mean(dists)          # Loc Error
            
            # 2. ★新規: 統合コスト (Integrated Cost) の計算
            costs = []
            for m in detected:
                d = m['loc_dist']
                abs_l = abs(m['len_diff'])
                
                # パターンY: 足切りロジック
                if d > dist_thresh:
                    # 場所が遠すぎる(>2hop)場合、時間の正確さは無意味とみなす。
                    # 時間誤差として「正解の滞在時間そのもの(全ミス)」をペナルティ採用
                    # これにより「遠い場所で偶然時間が合った」ケースを排除
                    time_cost = m['gt_dur'] 
                else:
                    # 場所が許容範囲なら、実際の時間誤差を採用
                    time_cost = abs_l
                
                # Cost = 時間コスト + (重み * 空間コスト)
                # 例: 距離2hop, 時間誤差0 -> Cost = 0 + 1.5*2 = 3.0
                # 例: 距離3hop, 時間誤差0 -> Cost = gt_dur(例:5) + 1.5*3 = 9.5 (大ペナルティ)
                cost = time_cost + (alpha * d)
                costs.append(cost)
            
            mean_cost = np.mean(costs)
            
            return det_rate, mean_len_diff, mean_len_abs, mean_loc_dist, mean_cost

        # 変数の受け取り (5つに増加)
        k_rate, k_ldiff, k_labs, k_loc, k_cost = summarize_stay(stay_metrics_k)
        a_rate, a_ldiff, a_labs, a_loc, a_cost = summarize_stay(stay_metrics_a)

        metrics_list.append({
            'id': idx, 'len': gen_len,
            'k_acc': 1 if acc_k else 0, 'k_prob': like_k,
            'a_acc': 1 if acc_a else 0, 'a_prob': like_a,
            'k_ed': calc_levenshtein(gt_future, pred_k),
            'a_ed': calc_levenshtein(gt_future, pred_a),
            'k_ged': calc_levenshtein(gt_future, pred_k, geo=True),
            'a_ged': calc_levenshtein(gt_future, pred_a, geo=True),
            'k_dtw': calc_dtw(gt_future, pred_k),
            'a_dtw': calc_dtw(gt_future, pred_a),
            'k_gdtw': calc_dtw(gt_future, pred_k, geo=True),
            'a_gdtw': calc_dtw(gt_future, pred_a, geo=True),
            'prompt': prompt_seq, 'gt': gt_future, 'pred_k': pred_k, 'pred_a': pred_a,
            # ★ 滞在指標 (Absを追加)
            'k_stay_rate': k_rate if k_rate is not None else np.nan,
            'k_stay_len_diff': k_ldiff if k_ldiff is not None else np.nan,
            'k_stay_len_abs':  k_labs  if k_labs  is not None else np.nan,
            'k_stay_dist': k_loc if k_loc is not None else np.nan,
            'k_stay_cost': k_cost if k_cost is not None else np.nan, # New
            
            'a_stay_rate': a_rate if a_rate is not None else np.nan,
            'a_stay_len_diff': a_ldiff if a_ldiff is not None else np.nan,
            'a_stay_len_abs':  a_labs  if a_labs  is not None else np.nan,
            'a_stay_dist': a_loc if a_loc is not None else np.nan,
            'a_stay_cost': a_cost if a_cost is not None else np.nan, # New
            'prompt': prompt_seq, 'gt': gt_future, 'pred_k': pred_k, 'pred_a': pred_a
        })

    # Save Metrics
    df = pd.DataFrame(metrics_list)
    csv_path = os.path.join(CONFIG["output_dir"], "metrics.csv")
    df.to_csv(csv_path, index=False)
    
    # =====================================================
    # 4. Result Summarization
    # =====================================================
    def print_metrics(label, d):
        print(f"\n>>> {label} (N={len(d)})")
        if len(d) == 0:
            print("  No samples found.")
            return
        print(f"  [Next Token] Acc:     Koopman={d['k_acc'].mean():.4f} | Ablation={d['a_acc'].mean():.4f}")
        print(f"  [Next Token] Prob:    Koopman={d['k_prob'].mean():.4f} | Ablation={d['a_prob'].mean():.4f}")
        
        k_ed = (d['k_ed'] / d['len']).mean()
        a_ed = (d['a_ed'] / d['len']).mean()
        k_ged = (d['k_ged'] / d['len']).mean()
        a_ged = (d['a_ged'] / d['len']).mean()
        k_dtw = (d['k_dtw'] / d['len']).mean()
        a_dtw = (d['a_dtw'] / d['len']).mean()
        k_gdtw = (d['k_gdtw'] / d['len']).mean()
        a_gdtw = (d['a_gdtw'] / d['len']).mean()

        print(f"  [Gen] ED (norm):      Koopman={k_ed:.4f} | Ablation={a_ed:.4f}")
        print(f"  [Gen] Geo-ED (norm):  Koopman={k_ged:.4f} | Ablation={a_ged:.4f}")
        print(f"  [Gen] DTW (norm):     Koopman={k_dtw:.4f} | Ablation={a_dtw:.4f}")
        print(f"  [Gen] Geo-DTW (norm): Koopman={k_gdtw:.4f} | Ablation={a_gdtw:.4f}")

        # ★ 追加: 滞在指標の表示 (NaNを除去して平均)
        print("  [Stay Metrics] (Detected Stays Only)")
        print(f"    Detection Rate:     Koopman={d['k_stay_rate'].mean():.4f} | Ablation={d['a_stay_rate'].mean():.4f}")
        print(f"    Length Diff (avg):  Koopman={d['k_stay_len_diff'].mean():.4f} | Ablation={d['a_stay_len_diff'].mean():.4f}")
        print(f"    Length Diff (Abs):  Koopman={d['k_stay_len_abs'].mean():.4f}  | Ablation={d['a_stay_len_abs'].mean():.4f}")
        print(f"    Loc Dist (hops):    Koopman={d['k_stay_dist'].mean():.4f}     | Ablation={d['a_stay_dist'].mean():.4f}")
        print(f"    Integrated Cost:    Koopman={d['k_stay_cost'].mean():.4f}     | Ablation={d['a_stay_cost'].mean():.4f}")
        print("    (Cost: Lower is better. 0=Perfect. Includes penalty for dist > 3 hops)")

    df_short = df[df['len'] <= 8]
    df_long  = df[df['len'] > 8]

    print("\n" + "="*50)
    print("Evaluation Summary")
    print("="*50)
    print_metrics("Overall", df)
    print_metrics("Short Term (<= 8 steps)", df_short)
    print_metrics("Long Term (> 8 steps)", df_long)
    print("-" * 50)
    
    # =====================================================
    # 5. Plotting
    # =====================================================
    plot_pdf_path = os.path.join(CONFIG["output_dir"], "trajectories_sample.pdf")
    max_plot = CONFIG["plot_max_samples"]
    print(f"\nGenerating PDF plots for first {max_plot} samples...")
    
    with PdfPages(plot_pdf_path) as pdf:
        rows, cols = 5, 4
        per_page = rows * cols
        df_plot = df.head(max_plot)
        num_plots = len(df_plot)
        num_pages = (num_plots + per_page - 1) // per_page
        
        for p in range(num_pages):
            fig, axes = plt.subplots(rows, cols, figsize=(20, 24))
            axes = axes.flatten()
            start_i = p * per_page
            for i in range(per_page):
                curr_i = start_i + i
                ax = axes[i]
                if curr_i < num_plots:
                    row = df_plot.iloc[curr_i]
                    full_gt = row['prompt'] + row['gt']
                    full_k = row['prompt'] + row['pred_k']
                    full_a = row['prompt'] + row['pred_a']
                    ax.plot(full_gt, 'k-', alpha=0.3, label='GT', linewidth=2)
                    start_x = len(row['prompt'])
                    end_x = start_x + len(row['pred_k'])

                    ax.plot(range(start_x, end_x), row['pred_k'], 'r.-', label='Koopman', alpha=0.8)
                    ax.plot(range(start_x, end_x), row['pred_a'], 'b.--', label='Ablation', alpha=0.6)

                    ax.axvline(x=len(row['prompt'])-0.5, color='gray', linestyle=':')
                    ax.set_title(f"ID:{row['id']}\nAcc:{row['k_acc']}/{row['a_acc']} GeoDTW:{row['k_gdtw']:.2f}", fontsize=8)
                    ax.set_yticks(range(0, 38, 2))
                    ax.grid(True, alpha=0.3)
                    if i == 0: ax.legend(fontsize=6)
                else:
                    ax.axis('off')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
            
    print(f"Completed. Saved to {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()