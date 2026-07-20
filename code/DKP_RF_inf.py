"""
Inference and Evaluation Script
KoopmanとAblationモデルの性能比較（反実仮想なし）

元のKP_RF_inf_path_real.pyをベースに、以下に対応：
- 3分割データ（Train/Val/Test）の選択
- 可変Prefix長
- 短期/長期の閾値可変
- 全評価指標
- ★追加: ターミナル出力をファイルにも自動保存
- ★★NEW: DwellU指標（圧縮版+実時間版）を追加
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse
import networkx as nx
import os
import sys  # 追加: 出力保存用
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import Levenshtein
from tqdm import tqdm
from datetime import datetime

from DKP_RF import KoopmanRoutesFormer
from Transformer_Ablation import TransformerAblation
from network import Network, expand_adjacency_matrix
from tokenization import Tokenization

# ★NEW: DwellU関数をインポート
from utils_prism import summarize_dwell_metrics


def sample_from_topk(probs_1d: torch.Tensor, k: int) -> int:
    """Top-k のみから確率サンプリングして token id を返す。
    probs_1d: [V] (softmax 後)
    """
    k = int(k)
    if k <= 0:
        raise ValueError(f"top-k must be >=1, got {k}")
    V = probs_1d.numel()
    k = min(k, V)
    topk_probs, topk_idx = torch.topk(probs_1d, k=k, dim=0)
    topk_probs = topk_probs / (topk_probs.sum() + 1e-12)
    sampled_local = torch.multinomial(topk_probs, num_samples=1).item()
    return topk_idx[sampled_local].item()



# =========================================================
# 1. Configuration
# =========================================================
common_split_path = "/home/mizutani/projects/RF/data/common_split_indices_m5.npz"

CONFIG = {
    "gpu_id": 0,
    "pad_token": 38,
    "vocab_size": 42,
    # 生成停止トークン（<end>）。あなたの設定では <e> が 39 なのでそれに合わせる
    "end_token": 39,
    # <end> が出ない場合の無限ループ防止（必要なら大きくしてください）
    "max_gen_steps": 200,
    # Top-k サンプリングの k
    "sample_topk_k": 39,

    "stay_offset": 19,
    
    # ★★★ 評価データ設定 ★★★
    "eval_data": "TEST",  # "TRAIN", "VAL", "TEST" のいずれか
    
    # データパス
    "data_npz_path": "/home/mizutani/projects/RF/data/input_real_m5.npz",
    "adj_matrix_path": "/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt",
    
    # モデルパス
    "model_koopman_path": "/home/mizutani/projects/RF/runs/20260127_014201/model_weights_20260127_014201.pth",
    "model_ablation_path": "/home/mizutani/projects/RF/runs/20260127_014847/ablation_weights_20260127_014847.pth",
    "output_dir": "/home/mizutani/projects/RF/runs/20260127_014201/evaluation_0304_thr6",


    "plot_max_samples": 1000,
    
    # ★★★ Prefix設定（可変） ★★★
    "prefix_lengths": [5],  # 複数のPrefix長で評価
    
    # ★★★ 短期/長期の閾値 ★★★
    "short_long_threshold": 6,  # gen_lenの長期・短期の閾値
    
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

parser = argparse.ArgumentParser(description="Inference/Eval for DKP_RF with optional overrides")
parser.add_argument("--model-koopman-path", type=str, default=None)
parser.add_argument("--model-ablation-path", type=str, default=None)
parser.add_argument("--output-dir", type=str, default=None)
parser.add_argument("--eval-data", type=str, default=None, choices=["TRAIN", "VAL", "TEST"])
parser.add_argument("--prefix-lengths", type=str, default=None, help="comma separated, e.g. 2,3,4,5")
args = parser.parse_args()

if args.model_koopman_path is not None:
    CONFIG["model_koopman_path"] = args.model_koopman_path
if args.model_ablation_path is not None:
    CONFIG["model_ablation_path"] = args.model_ablation_path
if args.output_dir is not None:
    CONFIG["output_dir"] = args.output_dir
if args.eval_data is not None:
    CONFIG["eval_data"] = args.eval_data
if args.prefix_lengths is not None:
    CONFIG["prefix_lengths"] = [int(x.strip()) for x in args.prefix_lengths.split(",") if x.strip()]


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
        self.log = open(filename, "w", encoding='utf-8') # 上書きモード("w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # 即時書き込み

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def build_distance_matrix(adj_map):  #　adj_mapは ノード：[隣接ノード]　の形の辞書．
    G = nx.Graph()  # network xのグラフを作成
    for u, neighbors in adj_map.items():  # 辞書の（キー(u)，要素(neighbors)）を順に取り出す．
        for v in neighbors:  # uのneighborsの要素vを順に取り出す．
            G.add_edge(u, v)  # グラフの(u,v)に辺を追加する
    return dict(nx.all_pairs_shortest_path_length(G))  # 最短距離の行列を作成．


NODE_DISTANCES = build_distance_matrix(ADJACENCY_MAP) # NODE_DISTANCES[u][v] = uv間の最短距離

# このNODE_DISTANCESを下記の地理コスト計算関数の中で用いる．

def get_node_id(token, stay_offset=19, pad_token=38): # 単純な場合分けにより，node番号を取得
    if token == pad_token:
        return -1
    if token >= stay_offset and token < pad_token:
        return token - stay_offset
    if token < stay_offset:
        return token
    return -1


def get_geo_cost(t1, t2, stay_offset=19, pad_token=38): # token1,2から，距離に基づいた0~1のコストを返す
    """地理的コスト計算（既存: 距離に応じて0.3, 0.6等）"""
    n1 = get_node_id(t1, stay_offset, pad_token)
    n2 = get_node_id(t2, stay_offset, pad_token)
    if n1 == -1 or n2 == -1:
        return 1.0
    if n1 == n2:
        return 0.0
    try:
        dist = NODE_DISTANCES[n1][n2]
    except KeyError:
        return 1.0
    if dist == 1:
        return 0.3
    elif dist == 2:
        return 0.6
    else:
        return 1.0

def get_geo_cost_relaxed(t1, t2, stay_offset=19, pad_token=38): # token1,2から，距離に基づいた0~1のコストを返す
    """
    ★変更: 緩めの地理的コスト計算
    1hopまで(隣接ノードまで)は「コスト0」
    2hop以上は通常のミスマッチ（コスト1）
    """
    n1 = get_node_id(t1, stay_offset, pad_token)
    n2 = get_node_id(t2, stay_offset, pad_token)
    if n1 == -1 or n2 == -1:
        return 1.0
    if n1 == n2:
        return 0.0
    try:
        dist = NODE_DISTANCES[n1][n2]
        # ★ここを変更: dist <= 2 から dist <= 1 へ
        if dist <= 1:
            return 0.0 # 1hop以内ならペナルティなし
        else:
            return 1.0
    except KeyError:
        return 1.0

def get_stay_events(seq, stay_offset=19, pad_token=38): # いつからいつまでどこに滞在していたか，の辞書を返す
    """
    数列から滞在イベントを抽出
    Returns: list of dict {'start': int, 'end': int, 'node': int, 'dur': int}
    """
    events = []
    n = len(seq)
    i = 0
    while i < n:
        token = seq[i]
        if stay_offset <= token < pad_token: # i番目が滞在ならば：
            start = i
            node_id = token - stay_offset
            while i < n and seq[i] == token: # 同じ滞在トークンが続く限りループ
                i += 1
            end = i
            duration = end - start
            events.append({ # 「いつからいつまでどこに」滞在，を追加
                'start': start,
                'end': end,
                'node': node_id,
                'dur': duration
            })
        else: # i番目が移動トークン
            i += 1
    return events

# 滞在のタイミングがoverlapしているものを抽出し，それらの間の持続長やノード間距離などを取得し，
# [{detected:~, len_diff:~, loc_dist:~, gt_dur:~, gt_node:~},...]の辞書を返す

def calc_stay_metrics_pair(gt_seq, pred_seq, node_dists): 
    """
    GTと予測の滞在イベントをマッチングし、指標を計算
    """
    gt_events = get_stay_events(gt_seq)
    pred_events = get_stay_events(pred_seq)
    
    results = []
    
    for gt_e in gt_events:
        best_match = None
        max_overlap = 0
        
        for pred_e in pred_events:
            overlap_start = max(gt_e['start'], pred_e['start']) # ここの計算で，overlapを計算している．
            overlap_end = min(gt_e['end'], pred_e['end'])
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > 0:
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_match = pred_e  # ここにbest_matchが入ることで，次の指標計算がされる
        
        metric = {
            'detected': False,
            'len_diff': None,
            'loc_dist': None,
            'gt_dur': gt_e['dur'],
            'gt_node': gt_e['node']
        }
        
        if best_match:
            metric['detected'] = True
            metric['len_diff'] = best_match['dur'] - gt_e['dur']
            
            u, v = gt_e['node'], best_match['node']
            if u == v:
                dist = 0  # 同じノードなら距離0
            elif u in node_dists and v in node_dists[u]:
                dist = node_dists[u][v]  # ノード間距離を取得
            else:
                dist = 999
            metric['loc_dist'] = dist
        else:
            metric['len_diff'] = -gt_e['dur'] # ????
            metric['loc_dist'] = None
        
        results.append(metric)
    
    return results


def summarize_stay(metrics_list, alpha=1.5, dist_thresh=3):
    """
    滞在指標の集計
    alpha: 空間誤差1ホップを時間誤差何ステップ分に換算するか
    dist_thresh: これを超えて場所が離れていたら、時間誤差を無視してペナルティ化
    """
    if not metrics_list:
        return None, None, None, None, None
    
    detected = [m for m in metrics_list if m['detected']] # m：metrics_listの要素{detected:~,...}の組のうち，detectedがTrueのもの
    if not detected:  # if notとかあるのか．
        return 0.0, None, None, None, None
    
    det_rate = len(detected) / len(metrics_list)
    # 出力①
    # det_rate：滞在イベントのうちどれくらい検出できたか．ここで，検出＝時間的なoverlap．
    # →場所にはゆるく，時間的タイミングの一致度には厳しい指標．
    
    diffs = [m['len_diff'] for m in detected] # 検出した滞在イベントのlen_diffのリスト
    dists = [m['loc_dist'] for m in detected] # 検出した滞在イベントのloc_distのリスト
    
    # 出力②〜④
    mean_len_diff = np.mean(diffs)
    mean_len_abs = np.mean(np.abs(diffs)) # len_diffのプラマイ（＝過大/過小評価）関係なし．
    mean_loc_dist = np.mean(dists)
    
    # 統合コスト
    costs = []
    for m in detected:
        d = m['loc_dist']
        abs_l = abs(m['len_diff'])
        
        if d > dist_thresh:
            time_cost = m['gt_dur']  # 場所が遠すぎるなら，time_costはdurationになる(=実質，丸々不一致ということになる)
        else:
            time_cost = abs_l  # 場所距離3以下なら，abs(len_diff)がtime_costとなる．
        
        cost = time_cost + (alpha * d) # ここがキモ．統合コストの計算
        costs.append(cost)
    
    # 出力⑤
    mean_cost = np.mean(costs)  # metrics_listの全ての滞在イベントについて計算し平均する．

    return det_rate, mean_len_diff, mean_len_abs, mean_loc_dist, mean_cost


def calc_levenshtein(seq1, seq2, geo=False, relaxed=False):
    """
    Edit Distance
    geo=True: 地理的コストを使用
    relaxed=True: 2hop以内ならコスト0とする (geo=True時のみ有効)
    """
    if not geo:
        return Levenshtein.distance(seq1, seq2) # 普通のバージョンは既存のものを使う．
    else:
        # 地理的コストを使ったEdit Distance
        n, m = len(seq1), len(seq2)
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
                    # ★変更: relaxedモードか通常geoモードかでコスト関数を切り替え
                    if relaxed:
                        cost = get_geo_cost_relaxed(seq1[i-1], seq2[j-1])
                    else:
                        cost = get_geo_cost(seq1[i-1], seq2[j-1])
                
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # deletion
                    dp[i][j-1] + 1,      # insertion
                    dp[i-1][j-1] + cost  # substitution
                )
        
        return dp[n][m]


def calc_dtw(seq1, seq2, geo=False, relaxed=False):
    """
    DTW
    geo=True: 地理的コストを使用
    relaxed=True: 2hop以内ならコスト0とする (geo=True時のみ有効)
    """
    n, m = len(seq1), len(seq2)
    dtw_matrix = [[float('inf')] * (m + 1) for _ in range(n + 1)]
    dtw_matrix[0][0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if geo:
                # ★変更: relaxedモードか通常geoモードかでコスト関数を切り替え
                if relaxed:
                    cost = get_geo_cost_relaxed(seq1[i-1], seq2[j-1])
                else:
                    cost = get_geo_cost(seq1[i-1], seq2[j-1])
            else:
                cost = 0 if seq1[i-1] == seq2[j-1] else 1
            
            dtw_matrix[i][j] = cost + min(
                dtw_matrix[i-1][j],      # insertion
                dtw_matrix[i][j-1],      # deletion
                dtw_matrix[i-1][j-1]     # match
            )
    
    return dtw_matrix[n][m]


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


# =========================================================
# 3. Evaluator Classes
# =========================================================

class KoopmanEvaluator:
    """Koopmanモデルの評価クラス"""
    def __init__(self, model, tokenizer, ctx_det, device, base_N=19):
        self.model = model
        self.tokenizer = tokenizer
        self.ctx_det = ctx_det
        self.device = device
        self.base_N = base_N
        self.model.eval()
    
    def generate_trajectory(self, prompt_seq, start_time, agent_id, gen_len):
        """
        prefix(=prompt_seq) を1回だけ transformer に入れて z0 を作り、
        以降は Koopman (A,B,u) で K=gen_len ステップ自律生成（Greedy）
        """
        with torch.no_grad():
            # prefix だけをテンソル化（ここから先は prefix を伸ばさない）
            tokens = torch.tensor([prompt_seq], dtype=torch.long).to(self.device)  # [1, P]
            stay_counts = self.tokenizer.calculate_stay_counts(tokens).to(self.device)

            P = len(prompt_seq)
            holidays = torch.full((1, P), self.ctx_det.get_holiday(start_time), dtype=torch.long).to(self.device)
            time_zones = torch.full((1, P), self.ctx_det.get_timezone(start_time), dtype=torch.long).to(self.device)

            events = torch.zeros((1, P), dtype=torch.long).to(self.device)
            for i, token in enumerate(prompt_seq):
                events[0, i] = self.ctx_det.get_event_flag(token, start_time, self.base_N)

            agent_ids = torch.tensor([agent_id], dtype=torch.long).to(self.device)

            prefix_times = torch.tensor([int(start_time)], dtype=torch.long, device=self.device)  # [1]

            outputs = self.model.forward_rollout(
                prefix_tokens=tokens,
                prefix_stay_counts=stay_counts,
                prefix_agent_ids=agent_ids,
                prefix_holidays=holidays,
                prefix_time_zones=time_zones,
                prefix_events=events,
                prefix_times=prefix_times,
                K=int(gen_len),
                # future_tokens は渡さない（= 自律生成）
            )

            # outputs['pred_logits']: [1, K, vocab]
            pred_tokens = torch.argmax(outputs["pred_logits"], dim=-1)[0].tolist()  # 長さKのlist[int]
            return pred_tokens
        
    
    def generate_trajectory_until_end(self, prompt_seq, start_time, agent_id, end_token_id, max_steps, mode="greedy", topk_k=5):
        """
        自己回帰生成（<end> が出るまで）
        mode:
          - "greedy": argmax
          - "topk": Top-k から確率サンプリング
        """
        with torch.no_grad():
            current_seq = list(prompt_seq)

            for _ in range(int(max_steps)):
                tokens = torch.tensor([current_seq], dtype=torch.long).to(self.device)
                stay_counts = self.tokenizer.calculate_stay_counts(tokens).to(self.device)

                seq_len = len(current_seq)
                holidays = torch.full((1, seq_len), self.ctx_det.get_holiday(start_time), dtype=torch.long).to(self.device)
                time_zones = torch.full((1, seq_len), self.ctx_det.get_timezone(start_time), dtype=torch.long).to(self.device)

                events = torch.zeros((1, seq_len), dtype=torch.long).to(self.device)
                for i, token in enumerate(current_seq):
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
                probs = F.softmax(pred_logits, dim=0)

                if mode == "greedy":
                    next_token = torch.argmax(pred_logits).item()
                elif mode == "topk":
                    next_token = sample_from_topk(probs, k=topk_k)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                current_seq.append(next_token)

                if next_token == int(end_token_id):
                    break

            generated = current_seq[len(prompt_seq):]
            return generated


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

            prefix_times = torch.tensor([int(start_time)], dtype=torch.long, device=self.device)  # [1]
            
            outputs = self.model.forward_rollout(
                prefix_tokens=tokens,
                prefix_stay_counts=stay_counts,
                prefix_agent_ids=agent_ids,
                prefix_holidays=holidays,
                prefix_time_zones=time_zones,
                prefix_events=events,
                K=1,
                prefix_times=prefix_times,   
            )
            
            pred_logits = outputs['pred_logits'][0, 0, :]  # [vocab]
            pred_token = torch.argmax(pred_logits).item()
            
            probs = F.softmax(pred_logits, dim=0)
            gt_prob = probs[gt_next].item()
            
            accuracy = (pred_token == gt_next)
            
            return accuracy, gt_prob


class AblationEvaluator:
    """Ablationモデルの評価クラス"""
    def __init__(self, model, tokenizer, ctx_det, device, base_N=19):
        self.model = model
        self.tokenizer = tokenizer
        self.ctx_det = ctx_det
        self.device = device
        self.base_N = base_N
        self.model.eval()
    
    def generate_trajectory(self, prompt_seq, start_time, agent_id, gen_len):
        """自己回帰生成（Greedy decoding）"""
        with torch.no_grad():
            current_seq = list(prompt_seq)
            
            for step in range(gen_len):
                tokens = torch.tensor([current_seq], dtype=torch.long).to(self.device)
                stay_counts = self.tokenizer.calculate_stay_counts(tokens).to(self.device)
                
                seq_len = len(current_seq)
                holidays = torch.full((1, seq_len), self.ctx_det.get_holiday(start_time), dtype=torch.long).to(self.device)
                time_zones = torch.full((1, seq_len), self.ctx_det.get_timezone(start_time), dtype=torch.long).to(self.device)
                
                events = torch.zeros((1, seq_len), dtype=torch.long).to(self.device)
                for i, token in enumerate(current_seq):
                    events[0, i] = self.ctx_det.get_event_flag(token, start_time, self.base_N)
                
                agent_ids = torch.tensor([agent_id], dtype=torch.long).to(self.device)
                
                # Ablationモデルで次トークン予測
                next_logits = self.model.generate_next_token(
                    tokens, stay_counts, agent_ids,
                    holidays, time_zones, events
                )
                
                next_token = torch.argmax(next_logits[0]).item()
                current_seq.append(next_token)
            
            generated = current_seq[len(prompt_seq):]
            return generated
    
    

    def generate_trajectory_until_end(
        self, prompt_seq, start_time, agent_id,
        end_token_id, max_steps,
        mode="greedy", topk_k=5, rng=None
        ):
        """自己回帰生成（Greedy / Top-k sampling, <end> が出るまで）"""
        if rng is None:
            rng = np.random.default_rng(0)  # 再現性が欲しくないなら seed を変える/Noneに

        with torch.no_grad():
            current_seq = list(prompt_seq)

            for _ in range(int(max_steps)):
                tokens = torch.tensor([current_seq], dtype=torch.long).to(self.device)
                stay_counts = self.tokenizer.calculate_stay_counts(tokens).to(self.device)

                seq_len = len(current_seq)
                holidays = torch.full((1, seq_len), self.ctx_det.get_holiday(start_time),
                                    dtype=torch.long).to(self.device)
                time_zones = torch.full((1, seq_len), self.ctx_det.get_timezone(start_time),
                                        dtype=torch.long).to(self.device)

                events = torch.zeros((1, seq_len), dtype=torch.long).to(self.device)
                for i, token in enumerate(current_seq):
                    events[0, i] = self.ctx_det.get_event_flag(token, start_time, self.base_N)

                agent_ids = torch.tensor([agent_id], dtype=torch.long).to(self.device)

                next_logits = self.model.generate_next_token(
                    tokens, stay_counts, agent_ids,
                    holidays, time_zones, events
                )  # shape: [1, vocab] 想定

                logits_1d = next_logits[0]  # [vocab]

                if mode == "greedy":
                    next_token = torch.argmax(logits_1d).item()

                elif mode in ("topk", "topk_sampling", "sample_topk"):
                    k = int(topk_k)
                    vocab = logits_1d.shape[0]
                    k = max(1, min(k, vocab))

                    # 上位kのlogitsを取り出して、その範囲でsoftmax→サンプリング
                    topk_vals, topk_idx = torch.topk(logits_1d, k=k)
                    topk_probs = F.softmax(topk_vals, dim=0).detach().cpu().numpy()

                    # numpyでサンプリング（GPU↔CPU転送はtopkのk個だけ）
                    pick = rng.choice(k, p=topk_probs)
                    next_token = int(topk_idx[pick].item())

                else:
                    raise ValueError(f"Unknown mode: {mode}")

                current_seq.append(next_token)

                if next_token == int(end_token_id):
                    break

            generated = current_seq[len(prompt_seq):]
            return generated


    def calc_next_token_metrics(self, prompt_seq, gt_next, start_time, agent_id):
        """次トークン予測の精度・確率を計算"""
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
            
            next_logits = self.model.generate_next_token(
                tokens, stay_counts, agent_ids,
                holidays, time_zones, events
            )
            
            pred_token = torch.argmax(next_logits[0]).item()
            
            probs = F.softmax(next_logits[0], dim=0)
            gt_prob = probs[gt_next].item()
            
            accuracy = (pred_token == gt_next)
            
            return accuracy, gt_prob
# =========================================================
# 4. Main Evaluation Function
# =========================================================

def main():
    # ★★★ ここで出力フォルダ作成とログ保存の設定を行う ★★★
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    log_file_path = os.path.join(CONFIG["output_dir"], "terminal_log.txt")
    sys.stdout = Logger(log_file_path)
    
    print("="*60)
    print("Koopman vs Ablation - Evaluation Script")
    print(f"Log saved to: {log_file_path}")
    print("="*60)
    
    device = torch.device(f"cuda:{CONFIG['gpu_id']}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # =====================================================
    # データロード
    # =====================================================
    print("\n=== Loading Data ===")
    
    trip_arrz = np.load(CONFIG["data_npz_path"])
    adj_matrix = torch.load(CONFIG["adj_matrix_path"], weights_only=True)
    
    if adj_matrix.shape[0] == 38:
        base_N = 19
        base_adj = adj_matrix[:base_N, :base_N]
    else:
        base_adj = adj_matrix
        base_N = int(base_adj.shape[0])
    
    # Network構築
    expanded_adj = expand_adjacency_matrix(adj_matrix)
    dummy_node_features = torch.zeros((len(adj_matrix), 1))
    expanded_features = torch.cat([dummy_node_features, dummy_node_features], dim=0)
    network = Network(expanded_adj, expanded_features)
    
    trip_arr = trip_arrz['route_arr']
    time_arr = trip_arrz['time_arr']
    agent_ids_arr = trip_arrz['agent_ids'] if 'agent_ids' in trip_arrz else np.zeros(len(trip_arr), dtype=int)
    
    # =====================================================
    # 共通分割インデックスのロード（系列レベル）
    # =====================================================
    print("\n=== Loading Common Split Indices ===")
    
    
    if not os.path.exists(common_split_path):
        raise FileNotFoundError(
            f"Common split file not found: {common_split_path}\n"
            "Please run 'python create_common_split.py' first!"
        )
    
    split_data = np.load(common_split_path)
    train_seq_indices = split_data['train_sequences']
    val_seq_indices = split_data['val_sequences']
    test_seq_indices = split_data['test_sequences']
    
    print(f"Train sequences: {len(train_seq_indices)}")
    print(f"Val sequences: {len(val_seq_indices)}")
    print(f"Test sequences: {len(test_seq_indices)}")
    
    # 評価対象の系列を選択
    if CONFIG["eval_data"] == "TRAIN":
        eval_seq_indices = train_seq_indices
    elif CONFIG["eval_data"] == "VAL":
        eval_seq_indices = val_seq_indices
    elif CONFIG["eval_data"] == "TEST":
        eval_seq_indices = test_seq_indices
    else:
        raise ValueError(f"Invalid eval_data: {CONFIG['eval_data']}")
    
    print(f"\nEvaluating on: {CONFIG['eval_data']} ({len(eval_seq_indices)} sequences)")

    
    # =====================================================
    # モデルロード
    # =====================================================
    print("\n=== Loading Models ===")
    
    # Koopmanモデル
    print("Loading Koopman model...")
    koopman_checkpoint = torch.load(CONFIG["model_koopman_path"], map_location=device, weights_only=False)
    koopman_config = koopman_checkpoint["config"]
    
    model_koopman = KoopmanRoutesFormer(
        vocab_size=koopman_config["vocab_size"],
        token_emb_dim=koopman_config["token_emb_dim"],
        d_model=koopman_config["d_model"],
        nhead=koopman_config["nhead"],
        num_layers=koopman_config["num_layers"],
        d_ff=koopman_config["d_ff"],
        z_dim=koopman_config["z_dim"],
        pad_token_id=koopman_config["pad_token_id"],
        base_N=koopman_config["base_N"],
        use_aux_loss=koopman_config.get("use_aux_loss", False),
        encoder_type=koopman_config.get("encoder_type", "transformer"),
        max_prefix_len=int(koopman_config.get("max_prefix_len", max(CONFIG["prefix_lengths"]))),
    ).to(device)
    
    model_koopman.load_state_dict(koopman_checkpoint["model_state_dict"])
    model_koopman.eval()
    print(f"Koopman model loaded. Parameters: {sum(p.numel() for p in model_koopman.parameters()):,}")
    
    # Ablationモデル
    print("Loading Ablation model...")
    ablation_checkpoint = torch.load(CONFIG["model_ablation_path"], map_location=device, weights_only=False)
    ablation_config = ablation_checkpoint["config"]
    
    model_ablation = TransformerAblation(
        vocab_size=ablation_config["vocab_size"],
        token_emb_dim=ablation_config["token_emb_dim"],
        d_model=ablation_config["d_model"],
        nhead=ablation_config["nhead"],
        num_layers=ablation_config["num_layers"],
        d_ff=ablation_config["d_ff"],
        pad_token_id=ablation_config["pad_token_id"],
        base_N=ablation_config["base_N"],
    ).to(device)
    
    model_ablation.load_state_dict(ablation_checkpoint["model_state_dict"])
    model_ablation.eval()
    print(f"Ablation model loaded. Parameters: {sum(p.numel() for p in model_ablation.parameters()):,}")
    
    # =====================================================
    # Evaluatorの初期化
    # =====================================================
    tokenizer = Tokenization(network)
    ctx_det = ContextDeterminer(CONFIG)
    
    eval_koopman = KoopmanEvaluator(model_koopman, tokenizer, ctx_det, device, base_N)
    eval_ablation = AblationEvaluator(model_ablation, tokenizer, ctx_det, device, base_N)
    
    # =====================================================
    # 評価ループ
    # =====================================================
    print("\n=== Starting Evaluation ===")
    print(f"Prefix lengths: {CONFIG['prefix_lengths']}")
    print(f"Short/Long threshold: {CONFIG['short_long_threshold']}")
    
    # 各Prefix長ごとに評価
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

            # --- [NEW] Sequence-level accuracy over the whole predicted future ---
            # We use mean per-step accuracy (not product) to avoid values vanishing with longer sequences.
            gt_future_arr = np.asarray(gt_future, dtype=np.int64)
            pred_k_arr = np.asarray(pred_k, dtype=np.int64)
            pred_a_arr = np.asarray(pred_a, dtype=np.int64)
            _L = len(gt_future_arr)
            _Lk = len(pred_k_arr)
            _La = len(pred_a_arr)
            _Lmin_k = min(_L, _Lk)
            _Lmin_a = min(_L, _La)
            k_acc_seq = float((pred_k_arr[:_Lmin_k] == gt_future_arr[:_Lmin_k]).mean()) if _Lmin_k > 0 else float('nan')
            # ↑こんな書き方できるのか．．．その経路の，未来部分全体の平均トークン正解率（0〜1）
            a_acc_seq = float((pred_a_arr[:_Lmin_a] == gt_future_arr[:_Lmin_a]).mean()) if _Lmin_a > 0 else float('nan')
            k_len_mismatch = int(_Lk != _L)
            a_len_mismatch = int(_La != _L)

            # ★追加: <end>が出るまで最後まで生成（Koopman/Ablation）
            pred_k_full = []
            pred_a_full = []
            pred_k_sample_topk = []
            pred_a_sample_topk = []

            # if idx < 400:
            #     pred_k_full = eval_koopman.generate_trajectory_until_end(
            #         prompt_seq, start_time, agent_id,
            #         end_token_id=CONFIG["end_token"],
            #         max_steps=CONFIG["max_gen_steps"],
            #         mode="greedy",
            #     )
            #     pred_a_full = eval_ablation.generate_trajectory_until_end(
            #         prompt_seq, start_time, agent_id,
            #         end_token_id=CONFIG["end_token"],
            #         max_steps=CONFIG["max_gen_steps"],
            #     )

            #     # ★追加: Top-k 確率サンプリングで <end> まで生成（Koopmanのみ）
            #     pred_k_sample_topk = eval_koopman.generate_trajectory_until_end(
            #         prompt_seq, start_time, agent_id,
            #         end_token_id=CONFIG["end_token"],
            #         max_steps=CONFIG["max_gen_steps"],
            #         mode="topk",
            #         topk_k=CONFIG["sample_topk_k"],
            #     )
                
            #     # ★追加: Top-k 確率サンプリングで <end> まで生成（Ablation）
            #     pred_a_sample_topk = eval_ablation.generate_trajectory_until_end(
            #         prompt_seq, start_time, agent_id,
            #         end_token_id=CONFIG["end_token"],
            #         max_steps=CONFIG["max_gen_steps"],
            #         mode="topk",
            #         topk_k=CONFIG["sample_topk_k"],
            #     )

            # 滞在指標
            stay_metrics_k = calc_stay_metrics_pair(gt_future, pred_k, NODE_DISTANCES)
            stay_metrics_a = calc_stay_metrics_pair(gt_future, pred_a, NODE_DISTANCES)
            
            k_rate, k_ldiff, k_labs, k_loc, k_cost = summarize_stay(stay_metrics_k)
            a_rate, a_ldiff, a_labs, a_loc, a_cost = summarize_stay(stay_metrics_a)
            
            # ★NEW: DwellU指標（圧縮版+実時間版）
            dwell_metrics_k = summarize_dwell_metrics(pred_k, gt_future, base_N=base_N)
            dwell_metrics_a = summarize_dwell_metrics(pred_a, gt_future, base_N=base_N)

            # =====================================================
            # [ADD] CSVに出したい「ルート付帯情報」
            # =====================================================

            # time_arr（このスクリプトでは各系列の start_time = time_arr[idx]）
            time_value = int(start_time)

            # 入力条件（全5列）
            is_night = int(ctx_det.get_timezone(start_time))     # 1:夜, 0:昼
            is_holiday = int(ctx_det.get_holiday(start_time))    # 1:休日, 0:平日

            event_nodes = set(ctx_det.get_event_nodes(start_time))  # その時刻で有効なイベントノード
            plaza_2  = 1 if 2  in event_nodes else 0
            plaza_11 = 1 if 11 in event_nodes else 0
            plaza_14 = 1 if 14 in event_nodes else 0

            # 経路の最初/最後のトークン（padding除去後の seq を使う）
            first_token = int(seq[0])
            last_token  = int(seq[-1])

            
            # メトリクス保存
            k_ed_raw = calc_levenshtein(gt_future, pred_k)
            a_ed_raw = calc_levenshtein(gt_future, pred_a)
            k_ged_raw = calc_levenshtein(gt_future, pred_k, geo=True)
            a_ged_raw = calc_levenshtein(gt_future, pred_a, geo=True)
            k_ged2_raw = calc_levenshtein(gt_future, pred_k, geo=True, relaxed=True)
            a_ged2_raw = calc_levenshtein(gt_future, pred_a, geo=True, relaxed=True)
            k_dtw_raw = calc_dtw(gt_future, pred_k)
            a_dtw_raw = calc_dtw(gt_future, pred_a)
            k_gdtw_raw = calc_dtw(gt_future, pred_k, geo=True)
            a_gdtw_raw = calc_dtw(gt_future, pred_a, geo=True)
            k_gdtw2_raw = calc_dtw(gt_future, pred_k, geo=True, relaxed=True)
            a_gdtw2_raw = calc_dtw(gt_future, pred_a, geo=True, relaxed=True)
            norm_den = float(max(gen_len, 1))

            metrics_list.append({
                'id': idx,
                'prefix_len': prefix_len,
                'gen_len': gen_len,
                # --- [ADD] 付帯情報 ---
                'time_arr': time_value,

                # 入力条件（全5列）
                'is_night': is_night,           # 1:夜, 0:昼
                'is_holiday': is_holiday,       # 1:休日, 0:平日
                'plaza_node2': plaza_2,         # 1:有, 0:無
                'plaza_node11': plaza_11,       # 1:有, 0:無
                'plaza_node14': plaza_14,       # 1:有, 0:無

                # 経路端点
                'first_token': first_token,
                'last_token': last_token,

                
                # 次トークン予測
                # 'k_acc': 1 if acc_k else 0,
                'k_prob': prob_k,
                # 'a_acc': 1 if acc_a else 0,
                'a_prob': prob_a,

                # Accuracy (sequence-level; mean over all future steps)
                'k_acc': k_acc_seq,
                'a_acc': a_acc_seq,
                # (kept for reference) Next-token accuracy/probability at the first future step only
                'k_acc_next1': 1 if acc_k else 0,
                'a_acc_next1': 1 if acc_a else 0,
                'k_len_mismatch': k_len_mismatch,
                'a_len_mismatch': a_len_mismatch,

                # Edit Distance (Normal & Geo)
                'k_ed': k_ed_raw,
                'a_ed': a_ed_raw,
                'k_ged': k_ged_raw,
                'a_ged': a_ged_raw,
                
                # ★追加: Relaxed Geo-ED (geoED2)
                'k_ged2': k_ged2_raw,
                'a_ged2': a_ged2_raw,
                
                # DTW (Normal & Geo)
                'k_dtw': k_dtw_raw,
                'a_dtw': a_dtw_raw,
                'k_gdtw': k_gdtw_raw,
                'a_gdtw': a_gdtw_raw,
                
                # ★追加: Relaxed Geo-DTW (geoDTW2)
                'k_gdtw2': k_gdtw2_raw,
                'a_gdtw2': a_gdtw2_raw,

                # 正規化版（prefix差の比較用）
                'k_ed_norm': k_ed_raw / norm_den,
                'a_ed_norm': a_ed_raw / norm_den,
                'k_ged_norm': k_ged_raw / norm_den,
                'a_ged_norm': a_ged_raw / norm_den,
                'k_ged2_norm': k_ged2_raw / norm_den,
                'a_ged2_norm': a_ged2_raw / norm_den,
                'k_dtw_norm': k_dtw_raw / norm_den,
                'a_dtw_norm': a_dtw_raw / norm_den,
                'k_gdtw_norm': k_gdtw_raw / norm_den,
                'a_gdtw_norm': a_gdtw_raw / norm_den,
                'k_gdtw2_norm': k_gdtw2_raw / norm_den,
                'a_gdtw2_norm': a_gdtw2_raw / norm_den,
                
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
                
                # ★NEW: DwellU指標
                'k_dwell_u': dwell_metrics_k['dwell_u'],
                'k_dwell_u_all': dwell_metrics_k['dwell_u_all'],
                'k_dwell_u_realtime': dwell_metrics_k['dwell_u_realtime'],
                'k_dwell_u_all_realtime': dwell_metrics_k['dwell_u_all_realtime'],
                'k_node_accuracy': dwell_metrics_k['node_accuracy'],
                
                'a_dwell_u': dwell_metrics_a['dwell_u'],
                'a_dwell_u_all': dwell_metrics_a['dwell_u_all'],
                'a_dwell_u_realtime': dwell_metrics_a['dwell_u_realtime'],
                'a_dwell_u_all_realtime': dwell_metrics_a['dwell_u_all_realtime'],
                'a_node_accuracy': dwell_metrics_a['node_accuracy'],
                
                # 系列保存
                'prompt': prompt_seq,
                'gt': gt_future,
                'pred_k': pred_k,
                'pred_a': pred_a,
                # ★追加: 最後まで生成した経路
                'pred_k_full': pred_k_full,
                'pred_a_full': pred_a_full,
                # ★追加: Top-k サンプリングで最後まで生成した経路（Koopman）
                'pred_k_sample_topk': pred_k_sample_topk,
                'pred_a_sample_topk': pred_a_sample_topk,
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
        
        print(f"  [Gen Raw] ED:          Koopman={d['k_ed'].mean():.4f} | Ablation={d['a_ed'].mean():.4f}")
        print(f"  [Gen Raw] Geo-ED:      Koopman={d['k_ged'].mean():.4f} | Ablation={d['a_ged'].mean():.4f}")
        print(f"  [Gen Raw] Geo-ED2:     Koopman={d['k_ged2'].mean():.4f} | Ablation={d['a_ged2'].mean():.4f}")
        print(f"  [Gen Raw] DTW:         Koopman={d['k_dtw'].mean():.4f} | Ablation={d['a_dtw'].mean():.4f}")
        print(f"  [Gen Raw] Geo-DTW:     Koopman={d['k_gdtw'].mean():.4f} | Ablation={d['a_gdtw'].mean():.4f}")
        print(f"  [Gen Raw] Geo-DTW2:    Koopman={d['k_gdtw2'].mean():.4f} | Ablation={d['a_gdtw2'].mean():.4f}")

        print(f"  [Gen Norm] ED:         Koopman={d['k_ed_norm'].mean():.4f} | Ablation={d['a_ed_norm'].mean():.4f}")
        print(f"  [Gen Norm] Geo-ED:     Koopman={d['k_ged_norm'].mean():.4f} | Ablation={d['a_ged_norm'].mean():.4f}")
        print(f"  [Gen Norm] Geo-ED2:    Koopman={d['k_ged2_norm'].mean():.4f} | Ablation={d['a_ged2_norm'].mean():.4f}")
        print(f"  [Gen Norm] DTW:        Koopman={d['k_dtw_norm'].mean():.4f} | Ablation={d['a_dtw_norm'].mean():.4f}")
        print(f"  [Gen Norm] Geo-DTW:    Koopman={d['k_gdtw_norm'].mean():.4f} | Ablation={d['a_gdtw_norm'].mean():.4f}")
        print(f"  [Gen Norm] Geo-DTW2:   Koopman={d['k_gdtw2_norm'].mean():.4f} | Ablation={d['a_gdtw2_norm'].mean():.4f}  (Relaxed: 0 if <=2hop)")
        
        print("  [Stay Metrics] (Detected Stays Only)")
        print(f"    Detection Rate:      Koopman={d['k_stay_rate'].mean():.4f} | Ablation={d['a_stay_rate'].mean():.4f}")
        print(f"    Length Diff (avg):   Koopman={d['k_stay_len_diff'].mean():.4f} | Ablation={d['a_stay_len_diff'].mean():.4f}")
        print(f"    Length Diff (Abs):   Koopman={d['k_stay_len_abs'].mean():.4f} | Ablation={d['a_stay_len_abs'].mean():.4f}")
        print(f"    Loc Dist (hops):     Koopman={d['k_stay_dist'].mean():.4f} | Ablation={d['a_stay_dist'].mean():.4f}")
        print(f"    Integrated Cost:     Koopman={d['k_stay_cost'].mean():.4f} | Ablation={d['a_stay_cost'].mean():.4f}")
        
        # ★NEW: DwellU指標
        print("  [DwellU Metrics] (NEW)")
        print(f"    DwellU (stays only):         Koopman={d['k_dwell_u'].mean():.4f} | Ablation={d['a_dwell_u'].mean():.4f}")
        print(f"    DwellU (all steps):          Koopman={d['k_dwell_u_all'].mean():.4f} | Ablation={d['a_dwell_u_all'].mean():.4f}")
        print(f"    DwellU RealTime (stays):     Koopman={d['k_dwell_u_realtime'].mean():.4f} | Ablation={d['a_dwell_u_realtime'].mean():.4f}")
        print(f"    DwellU RealTime (all):       Koopman={d['k_dwell_u_all_realtime'].mean():.4f} | Ablation={d['a_dwell_u_all_realtime'].mean():.4f}")
        print(f"    Node Accuracy:               Koopman={d['k_node_accuracy'].mean():.4f} | Ablation={d['a_node_accuracy'].mean():.4f}")
    
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
    
    for result in all_results:
        prefix_len = result['prefix_len']
        df = result['df']
        
        plot_pdf_path = os.path.join(CONFIG["output_dir"], f"trajectories_prefix{prefix_len}.pdf")
        max_plot = min(CONFIG["plot_max_samples"], len(df))
        
        print(f"Generating PDF for prefix={prefix_len} (first {max_plot} samples)...")
        
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
                        ax.set_title(f"ID:{row['id']} Pref:{row['prefix_len']}\nAcc:{row['k_acc']}/{row['a_acc']} GeoDTW:{row['k_gdtw']:.2f}/{row['a_gdtw']:.2f}", fontsize=8)
                        ax.set_yticks(range(0, 38, 2))
                        ax.grid(True, alpha=0.3)
                        if i == 0:
                            ax.legend(fontsize=6)
                    else:
                        ax.axis('off')
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()
        
        print(f"Saved to: {plot_pdf_path}")
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print(f"Results saved to: {CONFIG['output_dir']}")
    print("="*60)


if __name__ == "__main__":
    main()