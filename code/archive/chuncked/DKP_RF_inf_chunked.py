"""
Inference and Evaluation Script (Chunked K+L Multi-phase Rollout版)
Koopman (Multi-phase) と Ablation (Autoregressive) の性能比較

従来のDKP_RF_inf.pyを踏襲し、K+L Multi-phase rolloutに対応
- Koopman: K+Lずつ再入力してLステップrollout
- Ablation: 1ステップずつ自己回帰（従来通り）
- 全評価指標（ED, Geo-ED, Geo-ED2, DTW, Geo-DTW, Geo-DTW2, Stay metrics）
- 短期/長期分類
- ターミナル出力をファイルにも自動保存
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import Levenshtein
from tqdm import tqdm
from datetime import datetime

from DKP_RF import KoopmanRoutesFormer
from Transformer_Ablation import TransformerAblation
from network import Network, expand_adjacency_matrix
from tokenization import Tokenization
from datetime import datetime


run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
# =========================================================
# 1. Configuration
# =========================================================
common_split_path = "/home/mizutani/projects/RF/data/common_split_indices_m5.npz"

CONFIG = {
    "gpu_id": 0,
    "pad_token": 38,
    "end_token": 39, 
    "vocab_size": 42,
    "stay_offset": 19,
    
    # ★★★ 評価データ設定 ★★★
    "eval_data": "TEST",  # "TRAIN", "VAL", "TEST" のいずれか
    
    # データパス
    "data_npz_path": "/home/mizutani/projects/RF/data/input_real_m5.npz",
    "adj_matrix_path": "/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt",
    
    # モデルパス
    "model_koopman_path": "/home/mizutani/projects/RF/runs/20260128_180332_K4_L4/model_weights_20260128_180332.pth",
    "model_ablation_path": "/home/mizutani/projects/RF/runs/20260127_014847/ablation_weights_20260127_014847.pth",
    "output_dir": f"/home/mizutani/projects/RF/runs/20260128_180332_K4_L4/eval_{run_id}",

    "plot_max_samples": 1000,
    
    # ★★★ Chunked Rollout設定 ★★★
    "K": 4,  # Prefix length（学習時のKと同じ値を推奨）
    "L": 4,  # Rollout length（学習時のLと同じ値を推奨）
    
    # ★★★ Prefix設定（評価開始位置、実験的に変更可能） ★★★
    "prefix_lengths": [4],  # 複数のPrefix長で評価可能（原則はKと同じ）
    
    # ★★★ 短期/長期の閾値 ★★★
    "short_long_threshold": 9,  # これ以下が短期、より大きいが長期
    
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

class Logger(object):
    """ターミナル出力とファイル出力を同時に行うためのヘルパークラス"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def get_geo_cost(token1, token2, base_N=19, node_distances=None):
    """地理的コスト（最短距離）"""
    if token1 == token2:
        return 0
    
    # ノードIDに変換
    if 0 <= token1 < base_N:
        n1 = token1
    elif base_N <= token1 < base_N * 2:
        n1 = token1 - base_N
    else:
        return 1
    
    if 0 <= token2 < base_N:
        n2 = token2
    elif base_N <= token2 < base_N * 2:
        n2 = token2 - base_N
    else:
        return 1
    
    if node_distances is not None and n1 in node_distances and n2 in node_distances[n1]:
        return node_distances[n1][n2]
    
    return 1


def get_geo_cost_relaxed(token1, token2, base_N=19, adjacency_map=None):
    """Relaxed地理的コスト（2hop以内なら0）"""
    if token1 == token2:
        return 0
    
    # ノードIDに変換
    if 0 <= token1 < base_N:
        n1 = token1
    elif base_N <= token1 < base_N * 2:
        n1 = token1 - base_N
    else:
        return 1
    
    if 0 <= token2 < base_N:
        n2 = token2
    elif base_N <= token2 < base_N * 2:
        n2 = token2 - base_N
    else:
        return 1
    
    # 2hop以内なら0
    if adjacency_map is not None and n1 in adjacency_map and n2 in adjacency_map[n1]:
        return 0
    
    return 1


def calc_levenshtein(seq1, seq2, geo=False, relaxed=False, base_N=19, node_distances=None, adjacency_map=None):
    """Edit Distance (通常 / Geo / Geo-Relaxed)"""
    n, m = len(seq1), len(seq2)
    
    if n == 0 or m == 0:
        return max(n, m)
    
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i-1] == seq2[j-1]:
                cost = 0
            else:
                if geo:
                    if relaxed:
                        cost = get_geo_cost_relaxed(seq1[i-1], seq2[j-1], base_N, adjacency_map)
                    else:
                        cost = get_geo_cost(seq1[i-1], seq2[j-1], base_N, node_distances)
                else:
                    cost = 1
            
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )
    
    return dp[n][m]


def calc_dtw(seq1, seq2, geo=False, relaxed=False, base_N=19, node_distances=None, adjacency_map=None):
    """DTW (通常 / Geo / Geo-Relaxed)"""
    n, m = len(seq1), len(seq2)
    dtw_matrix = [[float('inf')] * (m + 1) for _ in range(n + 1)]
    dtw_matrix[0][0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if geo:
                if relaxed:
                    cost = get_geo_cost_relaxed(seq1[i-1], seq2[j-1], base_N, adjacency_map)
                else:
                    cost = get_geo_cost(seq1[i-1], seq2[j-1], base_N, node_distances)
            else:
                cost = 0 if seq1[i-1] == seq2[j-1] else 1
            
            dtw_matrix[i][j] = cost + min(
                dtw_matrix[i-1][j],
                dtw_matrix[i][j-1],
                dtw_matrix[i-1][j-1]
            )
    
    return dtw_matrix[n][m]


def extract_stays(seq, stay_offset=19):
    """滞在の抽出"""
    stays = []
    i = 0
    while i < len(seq):
        token = seq[i]
        if token >= stay_offset and token < stay_offset * 2:
            node_id = token - stay_offset
            length = 1
            j = i + 1
            while j < len(seq) and seq[j] == token:
                length += 1
                j += 1
            stays.append({'node': node_id, 'start': i, 'length': length})
            i = j
        else:
            i += 1
    return stays


def calc_stay_metrics_pair(gt_seq, pred_seq, node_distances, stay_offset=19):
    """
    滞在指標のペア計算（修正版）
    - 場所の一致ではなく、開始時刻の近さ(Time Proximity)でマッチングを行います。
    - これにより、場所が間違っている場合でも距離(Loc Dist)が正しく計算されます。
    """
    gt_stays = extract_stays(gt_seq, stay_offset)
    pred_stays = extract_stays(pred_seq, stay_offset)
    
    if len(gt_stays) == 0:
        return None
    
    # 予測された滞在がない場合は、検出なしとして空リストを返す
    if len(pred_stays) == 0:
        return []
    
    matched = []
    for gs in gt_stays:
        best_match = None
        best_time_diff = float('inf')
        
        # 1. 最も開始時刻が近い予測滞在を探す
        for ps in pred_stays:
            # 時間差（絶対値）
            time_diff = abs(ps['start'] - gs['start'])
            
            if time_diff < best_time_diff:
                best_time_diff = time_diff
                best_match = ps
            elif time_diff == best_time_diff:
                # 時間差が同じ場合、より長さが近い方などを選ぶ（タイブレーク）
                if abs(ps['length'] - gs['length']) < abs(best_match['length'] - gs['length']):
                    best_match = ps

        # 2. マッチングしたペアで指標を計算
        if best_match is not None:
            # 場所が違う場合の距離を取得
            if best_match['node'] == gs['node']:
                dist_val = 0
            else:
                # 辞書から距離を取得。見つからない場合(連結なし)はペナルティとして大きめの値(例: 10)
                # node_distances[u][v] の形式に合わせて取得
                n_gt = gs['node']
                n_pred = best_match['node']
                dist_val = node_distances.get(n_gt, {}).get(n_pred, 10) 

            matched.append({
                'gt': gs,
                'pred': best_match,
                'len_diff': best_match['length'] - gs['length'], # 長さの誤差
                'loc_dist': dist_val  # 距離の誤差（これが0以外になるようになります）
            })
    
    return matched

def summarize_stay(matched):
    """滞在指標のサマリー"""
    if matched is None or len(matched) == 0:
        return None, None, None, None, None
    
    rate = len(matched)  # 検出数
    len_diff_avg = np.mean([m['len_diff'] for m in matched])
    len_diff_abs = np.mean([abs(m['len_diff']) for m in matched])
    loc_dist = np.mean([m['loc_dist'] for m in matched])
    cost = (1 - rate) + len_diff_abs + loc_dist
    
    return rate, len_diff_avg, len_diff_abs, loc_dist, cost


class ContextDeterminer:
    """コンテキスト情報の判定"""
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
    
    def get_event_nodes(self, timestamp_int):
        """このタイムスタンプで有効なイベントノードのリストを返す"""
        date_int = timestamp_int // 10000
        hour = (timestamp_int // 100) % 100
        
        event_nodes = []
        for (ev_date, ev_start, ev_end, ev_nodes) in self.config["events"]:
            if date_int == ev_date and ev_start <= hour < ev_end:
                event_nodes.extend(ev_nodes)
        
        return event_nodes
    
    def get_event_flag(self, token_id, timestamp_int, base_N=19):
        """指定トークンがイベント対象かどうか"""
        event_nodes = self.get_event_nodes(timestamp_int)
        if not event_nodes:
            return 0
        
        # トークンIDからノードIDを取得
        if 0 <= token_id < base_N:
            node_id = token_id
        elif base_N <= token_id < base_N * 2:
            node_id = token_id - base_N
        else:
            return 0
        
        return 1 if node_id in event_nodes else 0


def token_to_node(token: int, base_N: int) -> int | None:
    # move: 0..base_N-1, stay: base_N..2*base_N-1
    if 0 <= token < base_N:
        return token
    if base_N <= token < 2 * base_N:
        return token - base_N
    return None  # <e>, <pad> など

def allowed_next_tokens(last_token: int, base_N: int, adjacency_map: dict[int, list[int]],
                        end_token: int) -> set[int]:
    # <e> の後は <e> 固定でも良いが、ここでは end は常に許可
    allowed = {end_token}

    last_node = token_to_node(last_token, base_N)
    if last_node is None:
        return allowed  # 特殊トークン直後は end だけ許可（保守的）

    # 「移動」先ノード候補（2-hop）＋「同一ノード」（滞在のため）
    neigh = set(adjacency_map.get(last_node, []))
    neigh.add(last_node)

    # move/stay の両方を許可
    for n in neigh:
        allowed.add(n)              # move token
        allowed.add(n + base_N)     # stay token
    return allowed

def greedy_with_mask(logits_1d: torch.Tensor, allowed: set[int]) -> int:
    # logits_1d: [vocab]
    masked = logits_1d.clone()
    vocab = masked.numel()
    disallowed = torch.ones(vocab, dtype=torch.bool, device=masked.device)
    idx = torch.tensor(sorted(list(allowed)), dtype=torch.long, device=masked.device)
    idx = idx[(0 <= idx) & (idx < vocab)]
    disallowed[idx] = False
    masked[disallowed] = -1e9
    return int(torch.argmax(masked).item())


# =========================================================
# 3. Evaluator Classes
# =========================================================

class KoopmanEvaluatorChunked:
    """
    Koopmanモデルの評価クラス（Multi-phase rollout版）
    K+Lずつ再入力してLステップrollout
    """
    def __init__(self, model, tokenizer, ctx_det, device, K, L, base_N=19):
        self.model = model
        self.tokenizer = tokenizer
        self.ctx_det = ctx_det
        self.device = device
        self.K = K
        self.L = L
        self.base_N = base_N
        self.model.eval()
    
    def generate_trajectory(self, prompt_seq, start_time, agent_id, gen_len):
        """
        Multi-phase rollout生成
        K+Lずつ再入力してLステップ予測を繰り返す
        """
        with torch.no_grad():
            # Context情報（Phase1で固定）
            holiday = self.ctx_det.get_holiday(start_time)
            timezone = self.ctx_det.get_timezone(start_time)
            
            # 現在の系列（初期はprompt）
            current_seq = list(prompt_seq)
            generated = []
            
            while len(generated) < gen_len:
                # 現在の系列をテンソルに
                tokens = torch.tensor([current_seq], dtype=torch.long).to(self.device)
                stay_counts = self.tokenizer.calculate_stay_counts(tokens).to(self.device)
                
                # コンテキスト情報（Phase間で固定）
                seq_len = len(current_seq)
                holidays = torch.full((1, seq_len), holiday, dtype=torch.long).to(self.device)
                time_zones = torch.full((1, seq_len), timezone, dtype=torch.long).to(self.device)
                
                # イベントフラグ（各トークンごとに判定、予測トークンも含む）
                events = torch.zeros((1, seq_len), dtype=torch.long).to(self.device)
                for i, token in enumerate(current_seq):
                    events[0, i] = self.ctx_det.get_event_flag(token, start_time, self.base_N)
                
                agent_ids = torch.tensor([agent_id], dtype=torch.long).to(self.device)
                
                # 残りステップ数
                remaining = gen_len - len(generated)
                rollout_steps = min(self.L, remaining)
                
                # Koopmanでrollout
                outputs = self.model.forward_rollout(
                    prefix_tokens=tokens,
                    prefix_stay_counts=stay_counts,
                    prefix_agent_ids=agent_ids,
                    prefix_holidays=holidays,
                    prefix_time_zones=time_zones,
                    prefix_events=events,
                    K=rollout_steps,
                )
                
                pred_logits = outputs['pred_logits'][0]  # [rollout_steps, vocab]
                
                # Greedyサンプリング
                end_token = CONFIG["end_token"]
                pad_token = CONFIG["pad_token"]

                # Greedyサンプリング（隣接マスク + <e> stop）
                for step in range(rollout_steps):
                    last_token = current_seq[-1]

                    allowed = allowed_next_tokens(
                        last_token=last_token,
                        base_N=self.base_N,
                        adjacency_map=ADJACENCY_MAP,
                        end_token=end_token,
                    )
                    next_token = greedy_with_mask(pred_logits[step], allowed)

                    generated.append(next_token)
                    current_seq.append(next_token)

                    # <e> が出たら終了（残りは pad で埋める）
                    if next_token == end_token:
                        remaining2 = gen_len - len(generated)
                        if remaining2 > 0:
                            generated.extend([pad_token] * remaining2)
                        return generated[:gen_len]

                # gen_lenに達したら終了
                if len(generated) >= gen_len:
                    break
            
            return generated[:gen_len]
    
    def calc_next_token_metrics(self, prompt_seq, gt_next, start_time, agent_id):
        """
        次トークン予測の精度・確率を計算
        """
        with torch.no_grad():
            tokens = torch.tensor([prompt_seq], dtype=torch.long).to(self.device)
            stay_counts = self.tokenizer.calculate_stay_counts(tokens).to(self.device)
            
            seq_len = len(prompt_seq)
            holidays = torch.full((1, seq_len), self.ctx_det.get_holiday(start_time), dtype=torch.long).to(self.device)
            time_zones = torch.full((1, seq_len), self.ctx_det.get_timezone(start_time), dtype=torch.long).to(self.device)
            
            events = torch.zeros((1, seq_len), dtype=torch.long).to(self.device)
            for i, token in enumerate(prompt_seq):
                events[0, i] = self.ctx_det.get_event_flag(token, start_time, self.base_N)
            
            agent_ids = torch.tensor([agent_id], dtype=torch.long).to(self.device)
            
            outputs = self.model.forward_rollout(
                prefix_tokens=tokens,
                prefix_stay_counts=stay_counts,
                prefix_agent_ids=agent_ids,
                prefix_holidays=holidays,
                prefix_time_zones=time_zones,
                prefix_events=events,
                K=1,
            )
            
            pred_logits = outputs['pred_logits'][0, 0, :]  # [vocab]
            pred_token = torch.argmax(pred_logits).item()
            
            probs = F.softmax(pred_logits, dim=0)
            gt_prob = probs[gt_next].item()
            
            accuracy = (pred_token == gt_next)
            
            return accuracy, gt_prob


class AblationEvaluator:
    """Ablationモデルの評価クラス（1ステップずつ自己回帰）"""
    def __init__(self, model, tokenizer, ctx_det, device, base_N=19):
        self.model = model
        self.tokenizer = tokenizer
        self.ctx_det = ctx_det
        self.device = device
        self.base_N = base_N
        self.model.eval()
    
    def generate_trajectory(self, prompt_seq, start_time, agent_id, gen_len):
        """
        自己回帰生成（Greedy decoding）
        """
        with torch.no_grad():
            current_seq = list(prompt_seq)
            
            for step in range(gen_len):
                # 1. 入力系列のTensor化 [1, T]
                tokens = torch.tensor([current_seq], dtype=torch.long).to(self.device)
                
                # 2. Stay Count計算
                stay_counts = self.tokenizer.calculate_stay_counts(tokens).to(self.device)
                
                # 3. コンテキスト情報の準備
                seq_len = len(current_seq)
                
                # Holiday & Timezone (全体で固定)
                h_val = self.ctx_det.get_holiday(start_time)
                tz_val = self.ctx_det.get_timezone(start_time)
                
                holidays = torch.full((1, seq_len), h_val, dtype=torch.long).to(self.device)
                time_zones = torch.full((1, seq_len), tz_val, dtype=torch.long).to(self.device)
                
                # Events (トークンごとに判定)
                events = torch.zeros((1, seq_len), dtype=torch.long).to(self.device)
                for i, token in enumerate(current_seq):
                    events[0, i] = self.ctx_det.get_event_flag(token, start_time, self.base_N)
                
                # Agent ID [1]
                agent_ids = torch.tensor([agent_id], dtype=torch.long).to(self.device)
                
                # 4. モデル推論
                outputs = self.model(
                    tokens=tokens,
                    stay_counts=stay_counts,
                    agent_ids=agent_ids,
                    holidays=holidays,
                    time_zones=time_zones,
                    events=events,
                )
                
                # 5. 次トークン決定
                end_token = CONFIG["end_token"]
                pad_token = CONFIG["pad_token"]

                # 最後のステップのロジットを使用
                pred_logits = outputs[0, -1, :]  # [vocab]

                allowed = allowed_next_tokens(
                    last_token=current_seq[-1],
                    base_N=self.base_N,
                    adjacency_map=ADJACENCY_MAP,
                    end_token=end_token,
                )
                next_token = greedy_with_mask(pred_logits, allowed)
                current_seq.append(next_token)

                if next_token == end_token:
                    # <e>が出たら残りはpadで埋める
                    remaining2 = gen_len - (step + 1)
                    if remaining2 > 0:
                        current_seq.extend([pad_token] * remaining2)
                    break
            
            generated = current_seq[len(prompt_seq):]
            return generated
    
    def calc_next_token_metrics(self, prompt_seq, gt_next, start_time, agent_id):
        """
        次トークン予測の精度・確率を計算
        """
        with torch.no_grad():
            # 1. 入力系列のTensor化 [1, T]
            # prompt_seq は list[int] である前提
            if isinstance(prompt_seq, list):
                tokens = torch.tensor([prompt_seq], dtype=torch.long).to(self.device)
            elif torch.is_tensor(prompt_seq):
                if prompt_seq.dim() == 1:
                    tokens = prompt_seq.unsqueeze(0).to(self.device)
                else:
                    tokens = prompt_seq.to(self.device)
            else:
                raise TypeError("prompt_seq must be list or tensor")

            # 2. Stay Count計算
            stay_counts = self.tokenizer.calculate_stay_counts(tokens).to(self.device)
            
            # 3. コンテキスト情報の準備
            seq_len = tokens.size(1)
            
            # Holiday & Timezone
            h_val = self.ctx_det.get_holiday(start_time)
            tz_val = self.ctx_det.get_timezone(start_time)
            
            holidays = torch.full((1, seq_len), h_val, dtype=torch.long).to(self.device)
            time_zones = torch.full((1, seq_len), tz_val, dtype=torch.long).to(self.device)
            
            # Events (リストに戻してイテレーション)
            events = torch.zeros((1, seq_len), dtype=torch.long).to(self.device)
            current_seq_list = tokens[0].cpu().tolist()
            for i, token in enumerate(current_seq_list):
                events[0, i] = self.ctx_det.get_event_flag(token, start_time, self.base_N)
            
            # Agent ID [1]
            # agent_id が Tensorの場合と intの場合を吸収
            if torch.is_tensor(agent_id):
                aid = agent_id.view(1).long().to(self.device)
            else:
                aid = torch.tensor([int(agent_id)], dtype=torch.long).to(self.device)
            
            # 4. モデル推論 (パディングはない前提なのでmaskは不要)
            # TransformerAblation.forward の引数に合わせて渡す
            logits = self.model(
                tokens=tokens,
                stay_counts=stay_counts,
                agent_ids=aid,
                holidays=holidays,
                time_zones=time_zones,
                events=events
            )
            # logits: [1, T, vocab]
            
            # 5. 評価
            # 最後のステップのロジットを取得
            pred_logits = logits[0, -1, :]  # [vocab]
            
            pred_token = torch.argmax(pred_logits).item()
            
            probs = F.softmax(pred_logits, dim=0)
            
            # gt_next の型吸収
            if torch.is_tensor(gt_next):
                gt_val = gt_next.item()
            elif isinstance(gt_next, list):
                gt_val = gt_next[0]
            else:
                gt_val = int(gt_next)

            gt_prob = probs[gt_val].item()
            accuracy = (pred_token == gt_val)
            
            return accuracy, gt_prob


# =========================================================
# 4. Main Evaluation
# =========================================================

def main():
    print("="*60)
    print("Koopman vs Ablation - Evaluation Script (Chunked Multi-phase)")
    print("="*60)
    
    # 出力ディレクトリ作成
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # ログファイル設定
    log_path = os.path.join(CONFIG["output_dir"], "terminal_log.txt")
    sys.stdout = Logger(log_path)
    print(f"Log saved to: {log_path}")
    print("="*60)
    
    # デバイス設定
    device = torch.device(f"cuda:{CONFIG['gpu_id']}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # =====================================================
    # データ読み込み
    # =====================================================
    print("\n=== Loading Data ===")
    
    trip_arrz = np.load(CONFIG["data_npz_path"])
    trip_arr = trip_arrz['route_arr']
    time_arr = trip_arrz['time_arr']
    agent_ids_arr = trip_arrz.get('agent_ids', np.zeros(len(trip_arr)))
    
    # 分割インデックス読み込み
    print("\n=== Loading Common Split Indices ===")
    split_data = np.load(common_split_path)
    train_seq_indices = split_data['train_sequences']
    val_seq_indices = split_data['val_sequences']
    test_seq_indices = split_data['test_sequences']
    
    print(f"Train sequences: {len(train_seq_indices)}")
    print(f"Val sequences: {len(val_seq_indices)}")
    print(f"Test sequences: {len(test_seq_indices)}")
    
    # 評価データ選択
    eval_data_name = CONFIG["eval_data"].upper()
    if eval_data_name == "TRAIN":
        eval_seq_indices = train_seq_indices
    elif eval_data_name == "VAL":
        eval_seq_indices = val_seq_indices
    elif eval_data_name == "TEST":
        eval_seq_indices = test_seq_indices
    else:
        raise ValueError(f"Invalid eval_data: {CONFIG['eval_data']}")
    
    print(f"\nEvaluating on: {eval_data_name} ({len(eval_seq_indices)} sequences)")
    
    # ノード距離行列の計算
    print("\n=== Computing Node Distances ===")
    adj_matrix = torch.load(CONFIG["adj_matrix_path"], weights_only=True)

    # Tokenizer初期化
    expanded_adj = expand_adjacency_matrix(adj_matrix)
    dummy_node_features = torch.zeros((len(adj_matrix), 1))
    expanded_features = torch.cat([dummy_node_features, dummy_node_features], dim=0)
    network = Network(expanded_adj, expanded_features)
    tokenizer = Tokenization(network)

    def compute_shortest_path_distance_matrix(adj, directed=False):
        if not isinstance(adj, torch.Tensor):
            adj = torch.tensor(adj)
        adj_cpu = adj.detach().cpu()
        N = adj_cpu.shape[0]
        G = nx.DiGraph() if directed else nx.Graph()
        G.add_nodes_from(range(N))
        rows, cols = torch.nonzero(adj_cpu, as_tuple=True)
        edges = [(int(r), int(c)) for r, c in zip(rows, cols) if int(r) != int(c)]
        G.add_edges_from(edges)
        
        lengths = dict(nx.all_pairs_shortest_path_length(G))
        dist_dict = {}
        for i in range(N):
            dist_dict[i] = {}
            for j in range(N):
                if i == j:
                    dist_dict[i][j] = 0
                elif j in lengths.get(i, {}):
                    dist_dict[i][j] = lengths[i][j]
                else:
                    dist_dict[i][j] = 999
        
        return dist_dict
    
    NODE_DISTANCES = compute_shortest_path_distance_matrix(expanded_adj, directed=False)
    base_N = CONFIG["stay_offset"]
    
    # Context Determiner
    ctx_det = ContextDeterminer(CONFIG)
    
    # =====================================================
    # モデル読み込み
    # =====================================================
    print("\n=== Loading Models ===")
    
    # Koopmanモデル
    print("Loading Koopman model...")
    koopman_ckpt = torch.load(CONFIG["model_koopman_path"], map_location=device, weights_only=False)
    koopman_config = koopman_ckpt.get("config", {})
    
    model_koopman = KoopmanRoutesFormer(
        vocab_size=koopman_config.get("vocab_size", 39),
        token_emb_dim=koopman_config.get("token_emb_dim", 64),
        d_model=koopman_config.get("d_model", 64),
        nhead=koopman_config.get("nhead", 4),
        num_layers=koopman_config.get("num_layers", 3),
        d_ff=koopman_config.get("d_ff", 128),
        z_dim=koopman_config.get("z_dim", 16),
        pad_token_id=CONFIG["pad_token"],
        base_N=base_N,
        num_agents=1,
        agent_emb_dim=16,
        max_stay_count=500,
        stay_emb_dim=16,
        holiday_emb_dim=4,
        time_zone_emb_dim=4,
        event_emb_dim=4,
        use_aux_loss=True,
    ).to(device)
    
    model_koopman.load_state_dict(koopman_ckpt["model_state_dict"])
    print(f"Koopman model loaded. Parameters: {sum(p.numel() for p in model_koopman.parameters()):,}")
    
    # Ablationモデル
    print("Loading Ablation model...")
    ablation_ckpt = torch.load(CONFIG["model_ablation_path"], map_location=device, weights_only=False)
    ablation_config = ablation_ckpt.get("config", {})
    
    model_ablation = TransformerAblation(
        vocab_size=ablation_config.get("vocab_size", 39),
        token_emb_dim=ablation_config.get("token_emb_dim", 64),
        d_model=ablation_config.get("d_model", 64),
        nhead=ablation_config.get("nhead", 4),
        num_layers=ablation_config.get("num_layers", 3),
        d_ff=ablation_config.get("d_ff", 128),
        pad_token_id=CONFIG["pad_token"],
        base_N=base_N,
        num_agents=1,
        agent_emb_dim=16,
        max_stay_count=500,
        stay_emb_dim=16,
        holiday_emb_dim=4,
        time_zone_emb_dim=4,
        event_emb_dim=4,
    ).to(device)
    
    model_ablation.load_state_dict(ablation_ckpt["model_state_dict"])
    print(f"Ablation model loaded. Parameters: {sum(p.numel() for p in model_ablation.parameters()):,}")
    
    # Evaluator初期化
    K = CONFIG["K"]
    L = CONFIG["L"]
    
    eval_koopman = KoopmanEvaluatorChunked(model_koopman, tokenizer, ctx_det, device, K, L, base_N)
    eval_ablation = AblationEvaluator(model_ablation, tokenizer, ctx_det, device, base_N)
    
    # =====================================================
    # 評価ループ
    # =====================================================
    print("\n=== Starting Evaluation ===")
    print(f"Prefix lengths: {CONFIG['prefix_lengths']}")
    print(f"Short/Long threshold: {CONFIG['short_long_threshold']}")
    print(f"Koopman Multi-phase: K={K}, L={L}")
    
    all_results = []
    
    for prefix_len in CONFIG["prefix_lengths"]:
        print(f"\n--- Evaluating with Prefix Length = {prefix_len} ---")
        
        metrics_list = []
        
        for idx in tqdm(eval_seq_indices, desc=f"Prefix={prefix_len}"):
            seq = trip_arr[idx]
            start_time = time_arr[idx]
            agent_id = agent_ids_arr[idx]
            
            # パディングを除去
            pad_indices = np.where(seq == CONFIG["pad_token"])[0]
            seq_len = pad_indices[0] if len(pad_indices) > 0 else len(seq)
            seq = seq[:seq_len]
            
            # Prefix長が十分か確認
            if seq_len <= prefix_len:
                continue
            
            # Prefix と Future に分割
            prompt_seq = seq[:prefix_len].tolist()
            gt_future = seq[prefix_len:].tolist()
            
            if len(gt_future) == 0:
                continue
            
            # Task 1: 次トークン予測
            gt_next = gt_future[0]
            
            acc_k, prob_k = eval_koopman.calc_next_token_metrics(prompt_seq, gt_next, start_time, agent_id)
            acc_a, prob_a = eval_ablation.calc_next_token_metrics(prompt_seq, gt_next, start_time, agent_id)
            
            # Task 2: 系列生成
            gen_len = len(gt_future)
            
            pred_k = eval_koopman.generate_trajectory(prompt_seq, start_time, agent_id, gen_len)
            pred_a = eval_ablation.generate_trajectory(prompt_seq, start_time, agent_id, gen_len)
            
            # 滞在指標
            stay_metrics_k = calc_stay_metrics_pair(gt_future, pred_k, NODE_DISTANCES, CONFIG["stay_offset"])
            stay_metrics_a = calc_stay_metrics_pair(gt_future, pred_a, NODE_DISTANCES, CONFIG["stay_offset"])
            
            k_rate, k_ldiff, k_labs, k_loc, k_cost = summarize_stay(stay_metrics_k)
            a_rate, a_ldiff, a_labs, a_loc, a_cost = summarize_stay(stay_metrics_a)
            
            # メトリクス保存
            metrics_list.append({
                'id': idx,
                'prefix_len': prefix_len,
                'gen_len': gen_len,
                
                # 次トークン予測
                'k_acc': 1 if acc_k else 0,
                'k_prob': prob_k,
                'a_acc': 1 if acc_a else 0,
                'a_prob': prob_a,
                
                # Edit Distance (Normal & Geo)
                'k_ed': calc_levenshtein(gt_future, pred_k, base_N=base_N),
                'a_ed': calc_levenshtein(gt_future, pred_a, base_N=base_N),
                'k_ged': calc_levenshtein(gt_future, pred_k, geo=True, base_N=base_N, node_distances=NODE_DISTANCES),
                'a_ged': calc_levenshtein(gt_future, pred_a, geo=True, base_N=base_N, node_distances=NODE_DISTANCES),
                
                # Relaxed Geo-ED
                'k_ged2': calc_levenshtein(gt_future, pred_k, geo=True, relaxed=True, base_N=base_N, adjacency_map=ADJACENCY_MAP),
                'a_ged2': calc_levenshtein(gt_future, pred_a, geo=True, relaxed=True, base_N=base_N, adjacency_map=ADJACENCY_MAP),
                
                # DTW (Normal & Geo)
                'k_dtw': calc_dtw(gt_future, pred_k, base_N=base_N),
                'a_dtw': calc_dtw(gt_future, pred_a, base_N=base_N),
                'k_gdtw': calc_dtw(gt_future, pred_k, geo=True, base_N=base_N, node_distances=NODE_DISTANCES),
                'a_gdtw': calc_dtw(gt_future, pred_a, geo=True, base_N=base_N, node_distances=NODE_DISTANCES),
                
                # Relaxed Geo-DTW
                'k_gdtw2': calc_dtw(gt_future, pred_k, geo=True, relaxed=True, base_N=base_N, adjacency_map=ADJACENCY_MAP),
                'a_gdtw2': calc_dtw(gt_future, pred_a, geo=True, relaxed=True, base_N=base_N, adjacency_map=ADJACENCY_MAP),
                
                # 滞在指標
                'k_stay_rate': k_rate if k_rate is not None else np.nan,
                'k_stay_len_diff': k_ldiff if k_ldiff is not None else np.nan,
                'k_stay_len_abs': k_labs if k_labs is not None else np.nan,
                'k_stay_dist': k_loc if k_loc is not None else np.nan,
                'k_stay_cost': k_cost if k_cost is not None else np.nan,
                
                'a_stay_rate': a_rate if a_rate is not None else np.nan,
                'a_stay_len_diff': a_ldiff if a_ldiff is not None else np.nan,
                'a_stay_len_abs': a_labs if a_labs is not None else np.nan,
                'a_stay_dist': a_loc if a_loc is not None else np.nan,
                'a_stay_cost': a_cost if a_cost is not None else np.nan,
                
                # 系列保存
                'prompt': prompt_seq,
                'gt': gt_future,
                'pred_k': pred_k,
                'pred_a': pred_a,
            })
        
        # DataFrameに変換
        df = pd.DataFrame(metrics_list)
        
        # CSV保存
        csv_path = os.path.join(CONFIG["output_dir"], f"metrics_prefix{prefix_len}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved metrics to: {csv_path}")
        
        all_results.append({
            'prefix_len': prefix_len,
            'df': df
        })
    
    # =====================================================
    # 結果の集計・表示
    # =====================================================
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    
    def print_metrics(label, d):
        """メトリクスを表示"""
        print(f"\n>>> {label} (N={len(d)})")
        if len(d) == 0:
            print("  No samples found.")
            return
        
        print(f"  [Next Token] Acc:      Koopman={d['k_acc'].mean():.4f} | Ablation={d['a_acc'].mean():.4f}")
        print(f"  [Next Token] Prob:     Koopman={d['k_prob'].mean():.4f} | Ablation={d['a_prob'].mean():.4f}")
        
        k_ed = (d['k_ed'] / d['gen_len']).mean()
        a_ed = (d['a_ed'] / d['gen_len']).mean()
        k_ged = (d['k_ged'] / d['gen_len']).mean()
        a_ged = (d['a_ged'] / d['gen_len']).mean()
        k_ged2 = (d['k_ged2'] / d['gen_len']).mean()
        a_ged2 = (d['a_ged2'] / d['gen_len']).mean()
        
        k_dtw = (d['k_dtw'] / d['gen_len']).mean()
        a_dtw = (d['a_dtw'] / d['gen_len']).mean()
        k_gdtw = (d['k_gdtw'] / d['gen_len']).mean()
        a_gdtw = (d['a_gdtw'] / d['gen_len']).mean()
        k_gdtw2 = (d['k_gdtw2'] / d['gen_len']).mean()
        a_gdtw2 = (d['a_gdtw2'] / d['gen_len']).mean()
        
        print(f"  [Gen] ED (norm):       Koopman={k_ed:.4f} | Ablation={a_ed:.4f}")
        print(f"  [Gen] Geo-ED (norm):   Koopman={k_ged:.4f} | Ablation={a_ged:.4f}")
        print(f"  [Gen] Geo-ED2 (norm):  Koopman={k_ged2:.4f} | Ablation={a_ged2:.4f}  (Relaxed: 0 if <=2hop)")
        print(f"  [Gen] DTW (norm):      Koopman={k_dtw:.4f} | Ablation={a_dtw:.4f}")
        print(f"  [Gen] Geo-DTW (norm):  Koopman={k_gdtw:.4f} | Ablation={a_gdtw:.4f}")
        print(f"  [Gen] Geo-DTW2 (norm): Koopman={k_gdtw2:.4f} | Ablation={a_gdtw2:.4f}  (Relaxed: 0 if <=2hop)")
        
        print("  [Stay Metrics] (Detected Stays Only)")
        print(f"    Detection Rate:      Koopman={d['k_stay_rate'].mean():.4f} | Ablation={d['a_stay_rate'].mean():.4f}")
        print(f"    Length Diff (avg):   Koopman={d['k_stay_len_diff'].mean():.4f} | Ablation={d['a_stay_len_diff'].mean():.4f}")
        print(f"    Length Diff (Abs):   Koopman={d['k_stay_len_abs'].mean():.4f} | Ablation={d['a_stay_len_abs'].mean():.4f}")
        print(f"    Loc Dist (hops):     Koopman={d['k_stay_dist'].mean():.4f} | Ablation={d['a_stay_dist'].mean():.4f}")
        print(f"    Integrated Cost:     Koopman={d['k_stay_cost'].mean():.4f} | Ablation={d['a_stay_cost'].mean():.4f}")
    
    # 各Prefix長ごとに表示
    for result in all_results:
        prefix_len = result['prefix_len']
        df = result['df']
        
        print(f"\n{'='*60}")
        print(f"Prefix Length = {prefix_len}")
        print(f"{'='*60}")
        
        # 全体
        print_metrics("Overall", df)
        
        # 短期・長期
        threshold = CONFIG["short_long_threshold"]
        df_short = df[df['gen_len'] <= threshold]
        df_long = df[df['gen_len'] > threshold]
        
        print_metrics(f"Short Term (<= {threshold} steps)", df_short)
        print_metrics(f"Long Term (> {threshold} steps)", df_long)
    
    # =====================================================
    # 可視化（PDF）
    # =====================================================
    print("\n=== Generating Plots ===")
    
    # ヘルパー: ノード変換関数
    def convert_to_nodes(seq, base_N, specials):
        """
        トークン列をノード列に変換
        - 滞在(base_N..2*base_N-1) -> ノード(0..base_N-1)
        - 移動(0..base_N-1) -> そのまま
        - 特殊(specials) -> 除外 (リストに含めない = 描画しない)
        """
        new_seq = []
        for t in seq:
            if t in specials:
                continue
            if base_N <= t < base_N * 2:
                new_seq.append(t - base_N)
            else:
                new_seq.append(t)
        return new_seq

    # 描画モード設定
    PLOT_MODES = [
        # ① トークンそのまま版
        {
            "name": "tokens",
            "y_min": -1,
            "y_max": 40,
            "y_ticks": range(0, 40, 5),
            "convert": lambda s: s,  # 変換なし
            "title_suffix": "(Raw Tokens)"
        },
        # ② ノード集約版
        {
            "name": "nodes",
            "y_min": -1,
            "y_max": 20,  # 0~19
            "y_ticks": range(0, 20, 1),
            "convert": lambda s: convert_to_nodes(s, CONFIG["stay_offset"], {CONFIG["pad_token"], CONFIG["end_token"]}),
            "title_suffix": "(Nodes Only)"
        }
    ]
    
    for result in all_results:
        prefix_len = result['prefix_len']
        df_origin = result['df']
        
        # 経路が長い順にソート
        df_sorted = df_origin.sort_values(by='gen_len', ascending=False).reset_index(drop=True)
        max_plot = len(df_sorted)
        
        # 文字列 -> リスト変換ヘルパー
        def to_list(val):
            if isinstance(val, (np.ndarray, list)):
                return list(val)
            if isinstance(val, str):
                try: return eval(val)
                except: return []
            return []

        # 各モードで描画
        for mode in PLOT_MODES:
            suffix = mode["name"]
            plot_pdf_path = os.path.join(CONFIG["output_dir"], f"trajectories_prefix{prefix_len}_sorted_{suffix}.pdf")
            print(f"Generating PDF ({suffix}): {plot_pdf_path} ...")
            
            with PdfPages(plot_pdf_path) as pdf:
                rows, cols = 5, 4
                per_page = rows * cols
                num_plots = len(df_sorted)
                num_pages = (num_plots + per_page - 1) // per_page
                
                for p in range(num_pages):
                    fig, axes = plt.subplots(rows, cols, figsize=(20, 24))
                    axes = axes.flatten()
                    start_i = p * per_page
                    
                    for i in range(per_page):
                        curr_i = start_i + i
                        ax = axes[i]
                        
                        if curr_i < num_plots:
                            row = df_sorted.iloc[curr_i]
                            
                            # 元データをリスト化
                            raw_prompt = to_list(row['prompt'])
                            raw_gt = to_list(row['gt'])
                            raw_pred_k = to_list(row['pred_k'])
                            raw_pred_a = to_list(row['pred_a'])
                            
                            # モードに応じてデータ変換
                            prompt = mode["convert"](raw_prompt)
                            gt = mode["convert"](raw_gt)
                            pred_k = mode["convert"](raw_pred_k)
                            pred_a = mode["convert"](raw_pred_a)
                            
                            # 描画処理 (データが空になっていないか確認)
                            if len(prompt) > 0:
                                # 1. GT (Prefix + Future)
                                full_gt = prompt + gt
                                ax.plot(range(len(full_gt)), full_gt, color='gray', linewidth=3, alpha=0.4, label='GT')
                                
                                # 2. 予測線の描画 (Prefix末尾から繋げる)
                                start_x = len(prompt) - 1
                                start_y = prompt[-1]
                                
                                # Koopman
                                if len(pred_k) > 0:
                                    x_range_k = range(start_x, start_x + 1 + len(pred_k))
                                    y_vals_k = [start_y] + pred_k
                                    ax.plot(x_range_k, y_vals_k, color='red', linestyle='-', linewidth=1.5, marker='.', markersize=4, alpha=0.8, label='Koopman')
                                
                                # Ablation
                                if len(pred_a) > 0:
                                    x_range_a = range(start_x, start_x + 1 + len(pred_a))
                                    y_vals_a = [start_y] + pred_a
                                    ax.plot(x_range_a, y_vals_a, color='blue', linestyle='--', linewidth=1.5, marker='.', markersize=4, alpha=0.8, label='Ablation')
                                
                                # 境界線
                                ax.axvline(x=start_x, color='black', linestyle=':', alpha=0.6, linewidth=1.0)
                            
                            # タイトル
                            title_str = (f"ID:{row['id']} | Len:{len(raw_gt)} {mode['title_suffix']}\n"
                                         f"Acc(K/A): {row['k_acc']}/{row['a_acc']} | "
                                         f"GeoDTW(K/A): {row['k_gdtw']:.2f}/{row['a_gdtw']:.2f}")
                            ax.set_title(title_str, fontsize=9)
                            
                            # 軸設定
                            ax.set_ylim(mode["y_min"], mode["y_max"])
                            ax.set_yticks(mode["y_ticks"])
                            ax.grid(True, alpha=0.3)
                            
                            # 凡例 (ページの最初のみ)
                            if i == 0:
                                ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
                        else:
                            ax.axis('off')
                    
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)
        
        print(f"Saved: {plot_pdf_path}")

    print("\n" + "="*60)
    print("Evaluation Complete!")
    print(f"Results saved to: {CONFIG['output_dir']}")
    print("="*60)

if __name__ == "__main__":
    main()