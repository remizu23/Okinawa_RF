"""
Inference and Evaluation Script
KoopmanとAblationモデルの性能比較　＋反実仮想

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
import networkx as nx
import os
import sys  # 追加: 出力保存用
import matplotlib.pyplot as plt
import Levenshtein
from tqdm import tqdm
from datetime import datetime

from DKP_RF import KoopmanRoutesFormer
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
    "output_dir": "/home/mizutani/projects/RF/runs/20260127_014201/evaluation0201_6",


    "plot_max_samples": 1000,
    
    # ★★★ Prefix設定（可変） ★★★
    "prefix_lengths": [5],  # 複数のPrefix長で評価
    
    # ★★★ 短期/長期の閾値 ★★★
    "short_long_threshold": 6,  # これ以下が短期、より大きいが長期
    
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
        self.log = open(filename, "w", encoding='utf-8') # 上書きモード("w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # 即時書き込み

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def build_distance_matrix(adj_map):
    G = nx.Graph()
    for u, neighbors in adj_map.items():
        for v in neighbors:
            G.add_edge(u, v)
    return dict(nx.all_pairs_shortest_path_length(G))

NODE_DISTANCES = build_distance_matrix(ADJACENCY_MAP)


def get_node_id(token, stay_offset=19, pad_token=38):
    if token == pad_token:
        return -1
    if token >= stay_offset and token < pad_token:
        return token - stay_offset
    if token < stay_offset:
        return token
    return -1


def get_geo_cost(t1, t2, stay_offset=19, pad_token=38):
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

def get_geo_cost_relaxed(t1, t2, stay_offset=19, pad_token=38):
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

def get_stay_events(seq, stay_offset=19, pad_token=38):
    """
    数列から滞在イベントを抽出
    Returns: list of dict {'start': int, 'end': int, 'node': int, 'dur': int}
    """
    events = []
    n = len(seq)
    i = 0
    while i < n:
        token = seq[i]
        if stay_offset <= token < pad_token:
            start = i
            node_id = token - stay_offset
            while i < n and seq[i] == token:
                i += 1
            end = i
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
    GTと予測の滞在イベントをマッチングし、指標を計算
    """
    gt_events = get_stay_events(gt_seq)
    pred_events = get_stay_events(pred_seq)
    
    results = []
    
    for gt_e in gt_events:
        best_match = None
        max_overlap = 0
        
        for pred_e in pred_events:
            overlap_start = max(gt_e['start'], pred_e['start'])
            overlap_end = min(gt_e['end'], pred_e['end'])
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > 0:
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_match = pred_e
        
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
                dist = 0
            elif u in node_dists and v in node_dists[u]:
                dist = node_dists[u][v]
            else:
                dist = 999
            metric['loc_dist'] = dist
        else:
            metric['len_diff'] = -gt_e['dur']
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
    
    detected = [m for m in metrics_list if m['detected']]
    if not detected:
        return 0.0, None, None, None, None
    
    det_rate = len(detected) / len(metrics_list)
    
    diffs = [m['len_diff'] for m in detected]
    dists = [m['loc_dist'] for m in detected]
    
    mean_len_diff = np.mean(diffs)
    mean_len_abs = np.mean(np.abs(diffs))
    mean_loc_dist = np.mean(dists)
    
    # 統合コスト
    costs = []
    for m in detected:
        d = m['loc_dist']
        abs_l = abs(m['len_diff'])
        
        if d > dist_thresh:
            time_cost = m['gt_dur']
        else:
            time_cost = abs_l
        
        cost = time_cost + (alpha * d)
        costs.append(cost)
    
    mean_cost = np.mean(costs)
    
    return det_rate, mean_len_diff, mean_len_abs, mean_loc_dist, mean_cost


def calc_levenshtein(seq1, seq2, geo=False, relaxed=False):
    """
    Edit Distance
    geo=True: 地理的コストを使用
    relaxed=True: 2hop以内ならコスト0とする (geo=True時のみ有効)
    """
    if not geo:
        return Levenshtein.distance(seq1, seq2)
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
    
    
    def _build_context_tensors(self, seq_tokens, start_time, override_time_zone=None, override_event_nodes=None):
        """Build (holidays, time_zones, events) tensors with optional counterfactual overrides.
        - override_time_zone: 0/1 or None (None uses ctx_det.get_timezone(start_time))
        - override_event_nodes: list[int] or None (None uses ctx_det.get_event_flag; [] makes all events off)
        """
        seq_len = len(seq_tokens)
        holidays = torch.full((1, seq_len), self.ctx_det.get_holiday(start_time), dtype=torch.long).to(self.device)

        tz = self.ctx_det.get_timezone(start_time) if override_time_zone is None else int(override_time_zone)
        time_zones = torch.full((1, seq_len), tz, dtype=torch.long).to(self.device)

        events = torch.zeros((1, seq_len), dtype=torch.long).to(self.device)
        if override_event_nodes is None:
            for i, token in enumerate(seq_tokens):
                events[0, i] = self.ctx_det.get_event_flag(token, start_time, self.base_N)
        else:
            # Counterfactual: determine event flag by membership in override_event_nodes
            ev_set = set(int(x) for x in override_event_nodes)
            for i, token in enumerate(seq_tokens):
                token_id = int(token)
                if 0 <= token_id < self.base_N:
                    node_id = token_id
                elif self.base_N <= token_id < self.base_N * 2:
                    node_id = token_id - self.base_N
                else:
                    node_id = None
                events[0, i] = 1 if (node_id is not None and node_id in ev_set) else 0

        return holidays, time_zones, events
    def generate_trajectory(self, prompt_seq, start_time, agent_id, gen_len, override_time_zone=None, override_event_nodes=None):
        """
        自己回帰生成（Greedy decoding）
        """
        with torch.no_grad():
            # 初期化
            current_seq = list(prompt_seq)
            
            for step in range(gen_len):
                # 現在の系列をテンソルに
                tokens = torch.tensor([current_seq], dtype=torch.long).to(self.device)
                stay_counts = self.tokenizer.calculate_stay_counts(tokens).to(self.device)
                # コンテキスト情報（反実仮想 override 対応）
                holidays, time_zones, events = self._build_context_tensors(
                    current_seq, start_time,
                    override_time_zone=override_time_zone,
                    override_event_nodes=override_event_nodes,
                )
                agent_ids = torch.tensor([agent_id], dtype=torch.long).to(self.device)
                
                # Koopmanモデルで1ステップ予測
                K = 1
                outputs = self.model.forward_rollout(
                    prefix_tokens=tokens,
                    prefix_stay_counts=stay_counts,
                    prefix_agent_ids=agent_ids,
                    prefix_holidays=holidays,
                    prefix_time_zones=time_zones,
                    prefix_events=events,
                    K=K,
                )
                
                pred_logits = outputs['pred_logits'][0, 0, :]  # [vocab]
                next_token = torch.argmax(pred_logits).item()
                
                current_seq.append(next_token)
            
            # Promptを除いた生成部分のみ返す
            generated = current_seq[len(prompt_seq):]
            return generated
    
    
    def generate_trajectory_until_end(self, prompt_seq, start_time, agent_id, end_token_id, max_steps, mode="greedy", topk_k=5, override_time_zone=None, override_event_nodes=None):
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
                holidays, time_zones, events = self._build_context_tensors(
                    current_seq, start_time,
                    override_time_zone=override_time_zone,
                    override_event_nodes=override_event_nodes,
                )
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


    def calc_next_token_metrics(self, prompt_seq, gt_next, start_time, agent_id, override_time_zone=None, override_event_nodes=None):
        """
        次トークン予測の精度・確率を計算
        """
        with torch.no_grad():
            tokens = torch.tensor([prompt_seq], dtype=torch.long).to(self.device)
            stay_counts = self.tokenizer.calculate_stay_counts(tokens).to(self.device)
            holidays, time_zones, events = self._build_context_tensors(
                prompt_seq, start_time,
                override_time_zone=override_time_zone,
                override_event_nodes=override_event_nodes,
            )
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




# =========================================================
# 4. Main Evaluation Function
# =========================================================

def main():
    # ★★★ ここで出力フォルダ作成とログ保存の設定を行う ★★★
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    log_file_path = os.path.join(CONFIG["output_dir"], "terminal_log.txt")
    sys.stdout = Logger(log_file_path)
    
    print("="*60)
    print("Koopman Evaluation Script (Counterfactual Embedding Tests)")
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
    ).to(device)
    
    model_koopman.load_state_dict(koopman_checkpoint["model_state_dict"])
    model_koopman.eval()
    print(f"Koopman model loaded. Parameters: {sum(p.numel() for p in model_koopman.parameters()):,}")

    # =====================================================
    # Evaluatorの初期化
    # =====================================================
    tokenizer = Tokenization(network)
    ctx_det = ContextDeterminer(CONFIG)
    
    eval_koopman = KoopmanEvaluator(model_koopman, tokenizer, ctx_det, device, base_N)
    
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

            # =====================================================
            # [NEW] 反実仮想（is_night / plaza 埋め込みの効果測定）
            #   - factual: 元の条件
            #   - cf_night0: night=1 のデータを night=0 にして推論
            #   - cf_plazaoff: plaza(イベント)がONのデータを全OFFにして推論
            #  diffは「factual - counterfactual（確率がどれだけ上がったか）」で記録
            # =====================================================
            factual_is_night = int(ctx_det.get_timezone(start_time))
            factual_event_nodes = ctx_det.get_event_nodes(start_time)
            factual_has_plaza = 1 if len(factual_event_nodes) > 0 else 0

            prob_cf_night0 = None
            prob_cf_plazaoff = None
            acc_cf_night0 = None
            acc_cf_plazaoff = None

            # night=1 のときだけ night=0 で推論
            if factual_is_night == 1:
                acc_cf_night0, prob_cf_night0 = eval_koopman.calc_next_token_metrics(
                    prompt_seq, gt_next, start_time, agent_id,
                    override_time_zone=0,
                    override_event_nodes=None
                )

            # plaza(イベント)がONのときだけ 全OFF で推論
            if factual_has_plaza == 1:
                acc_cf_plazaoff, prob_cf_plazaoff = eval_koopman.calc_next_token_metrics(
                    prompt_seq, gt_next, start_time, agent_id,
                    override_time_zone=None,
                    override_event_nodes=[]
                )

            prob_diff_night = (prob_k - prob_cf_night0) if prob_cf_night0 is not None else np.nan
            prob_diff_plaza = (prob_k - prob_cf_plazaoff) if prob_cf_plazaoff is not None else np.nan

            
            # Task 2: 系列生成
            gen_len = len(gt_future)
            
            pred_k = eval_koopman.generate_trajectory(prompt_seq, start_time, agent_id, gen_len)

            # =====================================================
            # [ADD] 生成系列の距離指標（ED/Geo-ED/DTW/Geo-DTW）
            # =====================================================
            k_ed    = calc_levenshtein(gt_future, pred_k, geo=False)
            k_ged   = calc_levenshtein(gt_future, pred_k, geo=True,  relaxed=False)
            k_ged2  = calc_levenshtein(gt_future, pred_k, geo=True,  relaxed=True)
            k_dtw   = calc_dtw(gt_future, pred_k, geo=False)
            k_gdtw  = calc_dtw(gt_future, pred_k, geo=True, relaxed=False)
            k_gdtw2 = calc_dtw(gt_future, pred_k, geo=True, relaxed=True)

            # ★追加: <end>が出るまで最後まで生成（Koopman/Ablation）

            # if idx < 400:
            #         prompt_seq, start_time, agent_id,
            #         end_token_id=CONFIG["end_token"],
            #         max_steps=CONFIG["max_gen_steps"],
            #         mode="greedy",
            #     )
            #         prompt_seq, start_time, agent_id,
            #         end_token_id=CONFIG["end_token"],
            #         max_steps=CONFIG["max_gen_steps"],
            #     )

            #     # ★追加: Top-k 確率サンプリングで <end> まで生成（Koopmanのみ）
            #         prompt_seq, start_time, agent_id,
            #         end_token_id=CONFIG["end_token"],
            #         max_steps=CONFIG["max_gen_steps"],
            #         mode="topk",
            #         topk_k=CONFIG["sample_topk_k"],
            #     )
                
            #     # ★追加: Top-k 確率サンプリングで <end> まで生成（Ablation）
            #         prompt_seq, start_time, agent_id,
            #         end_token_id=CONFIG["end_token"],
            #         max_steps=CONFIG["max_gen_steps"],
            #         mode="topk",
            #         topk_k=CONFIG["sample_topk_k"],
            #     )

            # 滞在指標
            stay_metrics_k = calc_stay_metrics_pair(gt_future, pred_k, NODE_DISTANCES)
            
            k_rate, k_ldiff, k_labs, k_loc, k_cost = summarize_stay(stay_metrics_k)
            
            # ★NEW: DwellU指標（圧縮版+実時間版）
            dwell_metrics_k = summarize_dwell_metrics(pred_k, gt_future, base_N=base_N)
            # =====================================================
            # メトリクス保存（Koopmanのみ / 反実仮想差分つき）
            # =====================================================
            metrics_list.append({
                'id': idx,
                'prefix_len': prefix_len,
                'gen_len': gen_len,

                'scenario': 'factual',
                'is_counterfactual': 0,

                # 付帯情報
                'time_arr': int(start_time),
                'is_night': factual_is_night,          # 1:夜, 0:昼
                'plaza': factual_has_plaza,            # 1:いずれかON, 0:全OFF
                'plaza_node2': 1 if 2 in set(factual_event_nodes) else 0,
                'plaza_node11': 1 if 11 in set(factual_event_nodes) else 0,
                'plaza_node14': 1 if 14 in set(factual_event_nodes) else 0,
                'first_token': int(seq[0]),
                'last_token': int(seq[-1]),

                # 次トークン予測
                'k_acc': 1 if acc_k else 0,
                'k_prob': prob_k,

                # 反実仮想との差分（factual - counterfactual）
                'k_prob_diff_night': prob_diff_night,
                'k_prob_diff_plaza': prob_diff_plaza,

                # [ADD] 生成系列の距離指標
                'k_ed': k_ed,
                'k_ged': k_ged,
                'k_ged2': k_ged2,
                'k_dtw': k_dtw,
                'k_gdtw': k_gdtw,
                'k_gdtw2': k_gdtw2,


                # 系列保存
                'prompt': prompt_seq,
                'gt': gt_future,
                'pred_k': pred_k,
            })
            # =====================================================
            # [NEW] 反実仮想行（IDは同じ、scenarioで区別）
            # =====================================================
            if factual_is_night == 1 and prob_cf_night0 is not None:
                pred_cf_night0 = eval_koopman.generate_trajectory(
                    prompt_seq, start_time, agent_id, gen_len,
                    override_time_zone=0,
                    override_event_nodes=None
                )
                cf_ed    = calc_levenshtein(gt_future, pred_cf_night0, geo=False)
                cf_ged   = calc_levenshtein(gt_future, pred_cf_night0, geo=True,  relaxed=False)
                cf_ged2  = calc_levenshtein(gt_future, pred_cf_night0, geo=True,  relaxed=True)
                cf_dtw   = calc_dtw(gt_future, pred_cf_night0, geo=False)
                cf_gdtw  = calc_dtw(gt_future, pred_cf_night0, geo=True, relaxed=False)
                cf_gdtw2 = calc_dtw(gt_future, pred_cf_night0, geo=True, relaxed=True)

                metrics_list.append({
                    'id': idx,
                    'prefix_len': prefix_len,
                    'gen_len': gen_len,
                    'scenario': 'cf_night0',
                    'is_counterfactual': 1,

                    'time_arr': int(start_time),
                    'is_night': 0,
                    'plaza': factual_has_plaza,
                    'plaza_node2': 1 if 2 in set(factual_event_nodes) else 0,
                    'plaza_node11': 1 if 11 in set(factual_event_nodes) else 0,
                    'plaza_node14': 1 if 14 in set(factual_event_nodes) else 0,
                    'first_token': int(seq[0]),
                    'last_token': int(seq[-1]),

                    'k_acc': 1 if acc_cf_night0 else 0,
                    'k_prob': prob_cf_night0,
                    'k_ed': cf_ed,
                    'k_ged': cf_ged,
                    'k_ged2': cf_ged2,
                    'k_dtw': cf_dtw,
                    'k_gdtw': cf_gdtw,
                    'k_gdtw2': cf_gdtw2,


                    'prompt': prompt_seq,
                    'gt': gt_future,
                    'pred_k': pred_cf_night0,
                })

            if factual_has_plaza == 1 and prob_cf_plazaoff is not None:
                pred_cf_plazaoff = eval_koopman.generate_trajectory(
                    prompt_seq, start_time, agent_id, gen_len,
                    override_time_zone=None,
                    override_event_nodes=[]
                )
                cf_ed    = calc_levenshtein(gt_future, pred_cf_plazaoff, geo=False)
                cf_ged   = calc_levenshtein(gt_future, pred_cf_plazaoff, geo=True,  relaxed=False)
                cf_ged2  = calc_levenshtein(gt_future, pred_cf_plazaoff, geo=True,  relaxed=True)
                cf_dtw   = calc_dtw(gt_future, pred_cf_plazaoff, geo=False)
                cf_gdtw  = calc_dtw(gt_future, pred_cf_plazaoff, geo=True, relaxed=False)
                cf_gdtw2 = calc_dtw(gt_future, pred_cf_plazaoff, geo=True, relaxed=True)

                metrics_list.append({
                    'id': idx,
                    'prefix_len': prefix_len,
                    'gen_len': gen_len,
                    'scenario': 'cf_plazaoff',
                    'is_counterfactual': 1,

                    'time_arr': int(start_time),
                    'is_night': factual_is_night,
                    'plaza': 0,
                    'plaza_node2': 0,
                    'plaza_node11': 0,
                    'plaza_node14': 0,
                    'first_token': int(seq[0]),
                    'last_token': int(seq[-1]),

                    'k_acc': 1 if acc_cf_plazaoff else 0,
                    'k_prob': prob_cf_plazaoff,
                    'k_ed': cf_ed,
                    'k_ged': cf_ged,
                    'k_ged2': cf_ged2,
                    'k_dtw': cf_dtw,
                    'k_gdtw': cf_gdtw,
                    'k_gdtw2': cf_gdtw2,


                    'prompt': prompt_seq,
                    'gt': gt_future,
                    'pred_k': pred_cf_plazaoff,
                })

            # if factual_has_plaza == 1:
            #     pred_cf_plazaoff = eval_koopman.generate_trajectory(
            #         prompt_seq, start_time, agent_id, gen_len,
            #         override_time_zone=None,
            #         override_event_nodes=[]
            #     )
            #     metrics_list.append({
            #         'id': idx,
            #         'prefix_len': prefix_len,
            #         'gen_len': gen_len,
            #         'scenario': 'cf_plazaoff',
            #         'is_counterfactual': 1,

            #         'time_arr': int(start_time),
            #         'is_night': factual_is_night,
            #         'plaza': 0,
            #         'plaza_node2': 0,
            #         'plaza_node11': 0,
            #         'plaza_node14': 0,
            #         'first_token': int(seq[0]),
            #         'last_token': int(seq[-1]),

            #         'k_acc': 1 if acc_cf_plazaoff else 0,
            #         'k_prob': prob_cf_plazaoff,

            #         'prompt': prompt_seq,
            #         'gt': gt_future,
            #         'pred_k': pred_cf_plazaoff,
            #     })

        # DataFrameに変換
        df = pd.DataFrame(metrics_list)
        # =====================================================
        # [NEW] diff のサマリーをターミナル出力（factual行のみ）
        # =====================================================
        try:
            df_factual = df[df.get('scenario', '') == 'factual'] if 'scenario' in df.columns else df
            if 'k_prob_diff_night' in df_factual.columns:
                mean_dn = df_factual['k_prob_diff_night'].dropna().mean()
                cnt_dn = df_factual['k_prob_diff_night'].dropna().shape[0]
                print(f"[Summary] k_prob_diff_night: mean={mean_dn:.6f} (n={cnt_dn})")
            if 'k_prob_diff_plaza' in df_factual.columns:
                mean_dp = df_factual['k_prob_diff_plaza'].dropna().mean()
                cnt_dp = df_factual['k_prob_diff_plaza'].dropna().shape[0]
                print(f"[Summary] k_prob_diff_plaza: mean={mean_dp:.6f} (n={cnt_dp})")
        except Exception as e:
            print(f"[Summary] Failed to compute diff summary: {e}")
        
        # CSV保存
        csv_path = os.path.join(CONFIG["output_dir"], f"metrics_cf_prefix{prefix_len}.csv")
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
        
        
        k_ed = (d['k_ed'] / d['gen_len']).mean()
        k_ged = (d['k_ged'] / d['gen_len']).mean()
        # ★追加: ged2
        k_ged2 = (d['k_ged2'] / d['gen_len']).mean()
        
        k_dtw = (d['k_dtw'] / d['gen_len']).mean()
        k_gdtw = (d['k_gdtw'] / d['gen_len']).mean()
        # ★追加: gdtw2
        k_gdtw2 = (d['k_gdtw2'] / d['gen_len']).mean()
        
        print(f"  [Gen] ED (norm):       Koopman={k_ed:.4f}")
        print(f"  [Gen] Geo-ED (norm):   Koopman={k_ged:.4f}")
        print(f"  [Gen] Geo-ED2 (norm):  Koopman={k_ged2:.4f}  (Relaxed: 0 if <=2hop)")
        print(f"  [Gen] DTW (norm):      Koopman={k_dtw:.4f}")
        print(f"  [Gen] Geo-DTW (norm):  Koopman={k_gdtw:.4f}")
        print(f"  [Gen] Geo-DTW2 (norm): Koopman={k_gdtw2:.4f}  (Relaxed: 0 if <=2hop)")
        
        print("  [Stay Metrics] (Detected Stays Only)")
        
        # ★NEW: DwellU指標
        print("  [DwellU Metrics] (NEW)")
    
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
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print(f"Results saved to: {CONFIG['output_dir']}")
    print("="*60)


if __name__ == "__main__":
    main()
# """
# Inference and Evaluation Script (Counterfactual Analysis)
# Koopmanモデル単体評価 + 反実仮想による埋め込み効果測定

# 変更点:
# - Ablationモデルとの比較機能を削除
# - 軌跡プロット(PDF出力)機能を削除
# - ★追加: Is_night=1 のデータに対し、Is_night=0 (Day) として推論した際の確率差分 (_diff_night) を算出
# - ★追加: Plazaあり のデータに対し、Plazaなし として推論した際の確率差分 (_diff_plaza) を算出
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import pandas as pd
# import networkx as nx
# import os
# import sys
# import Levenshtein
# from tqdm import tqdm
# from datetime import datetime

# # 必要なモジュールのみインポート
# from DKP_RF import KoopmanRoutesFormer
# from network import Network, expand_adjacency_matrix
# from tokenization import Tokenization
# from utils_prism import summarize_dwell_metrics


# # =========================================================
# # 1. Configuration
# # =========================================================
# common_split_path = "/home/mizutani/projects/RF/data/common_split_indices_m5.npz"

# CONFIG = {
#     "gpu_id": 0,
#     "pad_token": 38,
#     "vocab_size": 42,
#     "end_token": 39,
#     "max_gen_steps": 200,
    
#     "stay_offset": 19,
    
#     # ★★★ 評価データ設定 ★★★
#     "eval_data": "TEST",  # "TRAIN", "VAL", "TEST"
    
#     # データパス
#     "data_npz_path": "/home/mizutani/projects/RF/data/input_real_m5.npz",
#     "adj_matrix_path": "/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt",
    
#     # モデルパス (Koopmanのみ)
#     "model_koopman_path": "/home/mizutani/projects/RF/runs/20260127_014201/model_weights_20260127_014201.pth",
#     "output_dir": "/home/mizutani/projects/RF/runs/20260127_014201/evaluation0201_cf",

#     # ★★★ Prefix設定 ★★★
#     "prefix_lengths": [5],
    
#     # ★★★ 短期/長期の閾値 ★★★
#     "short_long_threshold": 5,
    
#     # Context Logic
#     "holidays": [20240928, 20240929, 20251122, 20251123],
#     "night_start": 19,
#     "night_end": 2,
#     "events": [
#         (20240929, 9, 16, [14]),
#         (20251122, 10, 19, [2, 11]),
#         (20251123, 10, 16, [2])
#     ],
# }


# # 2-hop Adjacency Map
# ADJACENCY_MAP = {
#     0: [1, 2, 4, 11], 1: [0, 2, 4, 5, 9], 2: [0, 1, 5, 6, 7],
#     4: [0, 1, 5, 8, 9, 10, 11], 5: [1, 2, 4, 6, 10], 6: [2, 5, 7, 10, 14],
#     7: [2, 6, 13, 14, 15], 8: [4, 9, 11], 9: [1, 4, 8, 10, 12],
#     10: [4, 5, 6, 9, 12, 13], 11: [0, 4, 8], 12: [9, 10, 13],
#     13: [7, 10, 12, 14, 15], 14: [6, 7, 13, 15, 16], 15: [7, 13, 14],
#     16: [14, 17, 18], 17: [16, 18], 18: [16, 17]
# }


# # =========================================================
# # 2. Helper Functions
# # =========================================================

# class Logger(object):
#     def __init__(self, filename):
#         self.terminal = sys.stdout
#         self.log = open(filename, "w", encoding='utf-8')

#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#         self.log.flush()

#     def flush(self):
#         self.terminal.flush()
#         self.log.flush()

# def build_distance_matrix(adj_map):
#     G = nx.Graph()
#     for u, neighbors in adj_map.items():
#         for v in neighbors:
#             G.add_edge(u, v)
#     return dict(nx.all_pairs_shortest_path_length(G))

# NODE_DISTANCES = build_distance_matrix(ADJACENCY_MAP)


# def get_node_id(token, stay_offset=19, pad_token=38):
#     if token == pad_token:
#         return -1
#     if token >= stay_offset and token < pad_token:
#         return token - stay_offset
#     if token < stay_offset:
#         return token
#     return -1


# def get_geo_cost(t1, t2, stay_offset=19, pad_token=38):
#     n1 = get_node_id(t1, stay_offset, pad_token)
#     n2 = get_node_id(t2, stay_offset, pad_token)
#     if n1 == -1 or n2 == -1: return 1.0
#     if n1 == n2: return 0.0
#     try:
#         dist = NODE_DISTANCES[n1][n2]
#     except KeyError:
#         return 1.0
#     if dist == 1: return 0.3
#     elif dist == 2: return 0.6
#     else: return 1.0

# def get_geo_cost_relaxed(t1, t2, stay_offset=19, pad_token=38):
#     n1 = get_node_id(t1, stay_offset, pad_token)
#     n2 = get_node_id(t2, stay_offset, pad_token)
#     if n1 == -1 or n2 == -1: return 1.0
#     if n1 == n2: return 0.0
#     try:
#         dist = NODE_DISTANCES[n1][n2]
#         if dist <= 1: return 0.0
#         else: return 1.0
#     except KeyError:
#         return 1.0

# def get_stay_events(seq, stay_offset=19, pad_token=38):
#     events = []
#     n = len(seq)
#     i = 0
#     while i < n:
#         token = seq[i]
#         if stay_offset <= token < pad_token:
#             start = i
#             node_id = token - stay_offset
#             while i < n and seq[i] == token:
#                 i += 1
#             end = i
#             duration = end - start
#             events.append({'start': start, 'end': end, 'node': node_id, 'dur': duration})
#         else:
#             i += 1
#     return events


# def calc_stay_metrics_pair(gt_seq, pred_seq, node_dists):
#     gt_events = get_stay_events(gt_seq)
#     pred_events = get_stay_events(pred_seq)
#     results = []
#     for gt_e in gt_events:
#         best_match = None
#         max_overlap = 0
#         for pred_e in pred_events:
#             overlap_start = max(gt_e['start'], pred_e['start'])
#             overlap_end = min(gt_e['end'], pred_e['end'])
#             overlap = max(0, overlap_end - overlap_start)
#             if overlap > 0:
#                 if overlap > max_overlap:
#                     max_overlap = overlap
#                     best_match = pred_e
#         metric = {
#             'detected': False, 'len_diff': None, 'loc_dist': None,
#             'gt_dur': gt_e['dur'], 'gt_node': gt_e['node']
#         }
#         if best_match:
#             metric['detected'] = True
#             metric['len_diff'] = best_match['dur'] - gt_e['dur']
#             u, v = gt_e['node'], best_match['node']
#             if u == v: dist = 0
#             elif u in node_dists and v in node_dists[u]: dist = node_dists[u][v]
#             else: dist = 999
#             metric['loc_dist'] = dist
#         else:
#             metric['len_diff'] = -gt_e['dur']
#             metric['loc_dist'] = None
#         results.append(metric)
#     return results


# def summarize_stay(metrics_list, alpha=1.5, dist_thresh=3):
#     if not metrics_list: return None, None, None, None, None
#     detected = [m for m in metrics_list if m['detected']]
#     if not detected: return 0.0, None, None, None, None
    
#     det_rate = len(detected) / len(metrics_list)
#     diffs = [m['len_diff'] for m in detected]
#     dists = [m['loc_dist'] for m in detected]
    
#     costs = []
#     for m in detected:
#         d = m['loc_dist']
#         abs_l = abs(m['len_diff'])
#         if d > dist_thresh: time_cost = m['gt_dur']
#         else: time_cost = abs_l
#         costs.append(time_cost + (alpha * d))
    
#     return det_rate, np.mean(diffs), np.mean(np.abs(diffs)), np.mean(dists), np.mean(costs)


# def calc_levenshtein(seq1, seq2, geo=False, relaxed=False):
#     if not geo: return Levenshtein.distance(seq1, seq2)
#     n, m = len(seq1), len(seq2)
#     dp = [[0] * (m + 1) for _ in range(n + 1)]
#     for i in range(n + 1): dp[i][0] = i
#     for j in range(m + 1): dp[0][j] = j
#     for i in range(1, n + 1):
#         for j in range(1, m + 1):
#             if seq1[i-1] == seq2[j-1]: cost = 0
#             else:
#                 cost = get_geo_cost_relaxed(seq1[i-1], seq2[j-1]) if relaxed else get_geo_cost(seq1[i-1], seq2[j-1])
#             dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
#     return dp[n][m]

# def calc_dtw(seq1, seq2, geo=False, relaxed=False):
#     n, m = len(seq1), len(seq2)
#     dtw_matrix = [[float('inf')] * (m + 1) for _ in range(n + 1)]
#     dtw_matrix[0][0] = 0
#     for i in range(1, n + 1):
#         for j in range(1, m + 1):
#             if geo:
#                 cost = get_geo_cost_relaxed(seq1[i-1], seq2[j-1]) if relaxed else get_geo_cost(seq1[i-1], seq2[j-1])
#             else:
#                 cost = 0 if seq1[i-1] == seq2[j-1] else 1
#             dtw_matrix[i][j] = cost + min(dtw_matrix[i-1][j], dtw_matrix[i][j-1], dtw_matrix[i-1][j-1])
#     return dtw_matrix[n][m]


# class ContextDeterminer:
#     def __init__(self, config):
#         self.config = config
    
#     def get_holiday(self, timestamp_int):
#         date_int = timestamp_int // 10000
#         return 1 if date_int in self.config["holidays"] else 0
    
#     def get_timezone(self, timestamp_int):
#         hour = (timestamp_int // 100) % 100
#         if hour >= self.config["night_start"] or hour < self.config["night_end"]:
#             return 1
#         return 0
    
#     def get_event_nodes(self, timestamp_int):
#         date_int = timestamp_int // 10000
#         hour = (timestamp_int // 100) % 100
#         event_nodes = []
#         for (ev_date, ev_start, ev_end, ev_nodes) in self.config["events"]:
#             if date_int == ev_date and ev_start <= hour < ev_end:
#                 event_nodes.extend(ev_nodes)
#         return event_nodes
    
#     def get_event_flag(self, token_id, timestamp_int, base_N=19):
#         event_nodes = self.get_event_nodes(timestamp_int)
#         if not event_nodes: return 0
#         if 0 <= token_id < base_N: node_id = token_id
#         elif base_N <= token_id < base_N * 2: node_id = token_id - base_N
#         else: return 0
#         return 1 if node_id in event_nodes else 0


# # =========================================================
# # 3. Evaluator Class (Koopman Only with CF Support)
# # =========================================================

# class KoopmanEvaluator:
#     def __init__(self, model, tokenizer, ctx_det, device, base_N=19):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.ctx_det = ctx_det
#         self.device = device
#         self.base_N = base_N
#         self.model.eval()
    
#     def generate_trajectory(self, prompt_seq, start_time, agent_id, gen_len):
#         with torch.no_grad():
#             current_seq = list(prompt_seq)
#             for step in range(gen_len):
#                 tokens = torch.tensor([current_seq], dtype=torch.long).to(self.device)
#                 stay_counts = self.tokenizer.calculate_stay_counts(tokens).to(self.device)
                
#                 seq_len = len(current_seq)
#                 holidays = torch.full((1, seq_len), self.ctx_det.get_holiday(start_time), dtype=torch.long).to(self.device)
#                 time_zones = torch.full((1, seq_len), self.ctx_det.get_timezone(start_time), dtype=torch.long).to(self.device)
                
#                 events = torch.zeros((1, seq_len), dtype=torch.long).to(self.device)
#                 for i, token in enumerate(current_seq):
#                     events[0, i] = self.ctx_det.get_event_flag(token, start_time, self.base_N)
                
#                 agent_ids = torch.tensor([agent_id], dtype=torch.long).to(self.device)
                
#                 outputs = self.model.forward_rollout(
#                     prefix_tokens=tokens, prefix_stay_counts=stay_counts, prefix_agent_ids=agent_ids,
#                     prefix_holidays=holidays, prefix_time_zones=time_zones, prefix_events=events, K=1
#                 )
#                 pred_logits = outputs['pred_logits'][0, 0, :]
#                 next_token = torch.argmax(pred_logits).item()
#                 current_seq.append(next_token)
#             return current_seq[len(prompt_seq):]

#     def calc_next_token_metrics(self, prompt_seq, gt_next, start_time, agent_id, 
#                                 force_time_zone=None, force_no_events=False):
#         """
#         次トークン予測の精度・確率を計算
#         ★反実仮想対応: force_time_zone, force_no_events 引数を追加
#         """
#         with torch.no_grad():
#             tokens = torch.tensor([prompt_seq], dtype=torch.long).to(self.device)
#             stay_counts = self.tokenizer.calculate_stay_counts(tokens).to(self.device)
            
#             seq_len = len(prompt_seq)
            
#             # Holiday (Normal)
#             holidays = torch.full((1, seq_len), self.ctx_det.get_holiday(start_time), dtype=torch.long).to(self.device)
            
#             # TimeZone (Check Force)
#             if force_time_zone is not None:
#                 # 反実仮想：指定されたTimeZoneID (0=Day) で埋める
#                 time_zones = torch.full((1, seq_len), force_time_zone, dtype=torch.long).to(self.device)
#             else:
#                 # 通常
#                 time_zones = torch.full((1, seq_len), self.ctx_det.get_timezone(start_time), dtype=torch.long).to(self.device)
            
#             # Events (Check Force)
#             events = torch.zeros((1, seq_len), dtype=torch.long).to(self.device)
            
#             if force_no_events:
#                 # 反実仮想：全て0にする（初期化で0なので何もしない、または明示的に0）
#                 pass 
#             else:
#                 # 通常
#                 for i, token in enumerate(prompt_seq):
#                     events[0, i] = self.ctx_det.get_event_flag(token, start_time, self.base_N)
            
#             agent_ids = torch.tensor([agent_id], dtype=torch.long).to(self.device)
            
#             outputs = self.model.forward_rollout(
#                 prefix_tokens=tokens, prefix_stay_counts=stay_counts, prefix_agent_ids=agent_ids,
#                 prefix_holidays=holidays, prefix_time_zones=time_zones, prefix_events=events, K=1
#             )
            
#             pred_logits = outputs['pred_logits'][0, 0, :]
#             pred_token = torch.argmax(pred_logits).item()
            
#             probs = F.softmax(pred_logits, dim=0)
#             gt_prob = probs[gt_next].item()
#             accuracy = (pred_token == gt_next)
            
#             return accuracy, gt_prob


# # =========================================================
# # 4. Main Evaluation Function
# # =========================================================

# def main():
#     os.makedirs(CONFIG["output_dir"], exist_ok=True)
#     log_file_path = os.path.join(CONFIG["output_dir"], "terminal_log_cf.txt")
#     sys.stdout = Logger(log_file_path)
    
#     print("="*60)
#     print("Koopman Counterfactual Evaluation Script")
#     print(f"Log saved to: {log_file_path}")
#     print("="*60)
    
#     device = torch.device(f"cuda:{CONFIG['gpu_id']}" if torch.cuda.is_available() else "cpu")
#     print(f"Device: {device}")
    
#     # データロード
#     print("\n=== Loading Data ===")
#     trip_arrz = np.load(CONFIG["data_npz_path"])
#     adj_matrix = torch.load(CONFIG["adj_matrix_path"], weights_only=True)
    
#     if adj_matrix.shape[0] == 38:
#         base_N = 19
#         base_adj = adj_matrix[:base_N, :base_N]
#     else:
#         base_adj = adj_matrix
#         base_N = int(base_adj.shape[0])
    
#     expanded_adj = expand_adjacency_matrix(adj_matrix)
#     dummy_node_features = torch.zeros((len(adj_matrix), 1))
#     expanded_features = torch.cat([dummy_node_features, dummy_node_features], dim=0)
#     network = Network(expanded_adj, expanded_features)
    
#     trip_arr = trip_arrz['route_arr']
#     time_arr = trip_arrz['time_arr']
#     agent_ids_arr = trip_arrz['agent_ids'] if 'agent_ids' in trip_arrz else np.zeros(len(trip_arr), dtype=int)
    
#     # Split
#     if not os.path.exists(common_split_path):
#         raise FileNotFoundError(f"Common split file not found: {common_split_path}")
    
#     split_data = np.load(common_split_path)
#     if CONFIG["eval_data"] == "TRAIN": eval_seq_indices = split_data['train_sequences']
#     elif CONFIG["eval_data"] == "VAL": eval_seq_indices = split_data['val_sequences']
#     elif CONFIG["eval_data"] == "TEST": eval_seq_indices = split_data['test_sequences']
#     else: raise ValueError(f"Invalid eval_data: {CONFIG['eval_data']}")
    
#     print(f"\nEvaluating on: {CONFIG['eval_data']} ({len(eval_seq_indices)} sequences)")
    
#     # モデルロード
#     print("\n=== Loading Koopman Model ===")
#     checkpoint = torch.load(CONFIG["model_koopman_path"], map_location=device, weights_only=False)
#     config = checkpoint["config"]
#     model = KoopmanRoutesFormer(
#         vocab_size=config["vocab_size"], token_emb_dim=config["token_emb_dim"], d_model=config["d_model"],
#         nhead=config["nhead"], num_layers=config["num_layers"], d_ff=config["d_ff"], z_dim=config["z_dim"],
#         pad_token_id=config["pad_token_id"], base_N=config["base_N"], use_aux_loss=config.get("use_aux_loss", False)
#     ).to(device)
#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.eval()
    
#     tokenizer = Tokenization(network)
#     ctx_det = ContextDeterminer(CONFIG)
#     evaluator = KoopmanEvaluator(model, tokenizer, ctx_det, device, base_N)
    
#     print("\n=== Starting Evaluation ===")
    
#     for prefix_len in CONFIG["prefix_lengths"]:
#         print(f"\n--- Evaluating with Prefix Length = {prefix_len} ---")
#         metrics_list = []
        
#         for idx in tqdm(eval_seq_indices, desc=f"Prefix={prefix_len}"):
#             seq = trip_arr[idx]
#             start_time = time_arr[idx]
#             agent_id = agent_ids_arr[idx]
            
#             pad_indices = np.where(seq == CONFIG["pad_token"])[0]
#             seq_len = pad_indices[0] if len(pad_indices) > 0 else len(seq)
#             seq = seq[:seq_len]
            
#             if seq_len <= prefix_len: continue
            
#             prompt_seq = seq[:prefix_len].tolist()
#             gt_future = seq[prefix_len:].tolist()
#             if len(gt_future) == 0: continue
            
#             gt_next = gt_future[0]
            
#             # --- 1. Base Evaluation (Normal Condition) ---
#             acc, prob = evaluator.calc_next_token_metrics(prompt_seq, gt_next, start_time, agent_id)
#             pred_seq = evaluator.generate_trajectory(prompt_seq, start_time, agent_id, len(gt_future))
            
#             # Context Info
#             is_night = int(ctx_det.get_timezone(start_time))
#             is_holiday = int(ctx_det.get_holiday(start_time))
#             event_nodes = set(ctx_det.get_event_nodes(start_time))
            
#             has_plaza = len(event_nodes) > 0
#             plaza_2  = 1 if 2  in event_nodes else 0
#             plaza_11 = 1 if 11 in event_nodes else 0
#             plaza_14 = 1 if 14 in event_nodes else 0
            
#             # --- 2. Counterfactual: Night ---
#             # Is_night=1 のデータについて、無理やり「昼(0)」として推論し、Probの差を見る
#             prob_cf_night_off = np.nan
#             diff_night = np.nan
            
#             if is_night == 1:
#                 _, prob_cf_night_off = evaluator.calc_next_token_metrics(
#                     prompt_seq, gt_next, start_time, agent_id,
#                     force_time_zone=0  # Force Day
#                 )
#                 # 正解埋め込み(Night) - 反実埋め込み(Day)
#                 # 値が正なら「Nightと正しく入れたおかげで確率が上がった」
#                 diff_night = prob - prob_cf_night_off
            
#             # --- 3. Counterfactual: Plaza ---
#             # Plazaイベントがあるデータについて、無理やり「イベントなし」として推論し、Probの差を見る
#             prob_cf_plaza_off = np.nan
#             diff_plaza = np.nan
            
#             if has_plaza:
#                 _, prob_cf_plaza_off = evaluator.calc_next_token_metrics(
#                     prompt_seq, gt_next, start_time, agent_id,
#                     force_no_events=True # Force No Events
#                 )
#                 # 正解埋め込み(Eventあり) - 反実埋め込み(Eventなし)
#                 # 値が正なら「Eventありと正しく入れたおかげで確率が上がった」
#                 diff_plaza = prob - prob_cf_plaza_off

#             # Metrics Calculation
#             stay_metrics = calc_stay_metrics_pair(gt_future, pred_seq, NODE_DISTANCES)
#             s_rate, s_ldiff, s_labs, s_loc, s_cost = summarize_stay(stay_metrics)
#             dwell_metrics = summarize_dwell_metrics(pred_seq, gt_future, base_N=base_N)
            
#             metrics_list.append({
#                 'id': idx,
#                 'prefix_len': prefix_len,
#                 'gen_len': len(gt_future),
#                 'time_arr': int(start_time),
                
#                 # Conditions
#                 'is_night': is_night,
#                 'is_holiday': is_holiday,
#                 'plaza_node2': plaza_2,
#                 'plaza_node11': plaza_11,
#                 'plaza_node14': plaza_14,
                
#                 # --- Base Metrics ---
#                 'k_acc': 1 if acc else 0,
#                 'k_prob': prob,
                
#                 # --- Counterfactual Results ---
#                 'k_prob_cf_night_off': prob_cf_night_off, # 反実仮想(NightOFF)時の確率
#                 'diff_night': diff_night,                 # 改善幅 (Base - NightOFF)
                
#                 'k_prob_cf_plaza_off': prob_cf_plaza_off, # 反実仮想(PlazaOFF)時の確率
#                 'diff_plaza': diff_plaza,                 # 改善幅 (Base - PlazaOFF)
                
#                 # --- Trajectory Metrics ---
#                 'k_ed': calc_levenshtein(gt_future, pred_seq),
#                 'k_ged': calc_levenshtein(gt_future, pred_seq, geo=True),
#                 'k_ged2': calc_levenshtein(gt_future, pred_seq, geo=True, relaxed=True),
#                 'k_dtw': calc_dtw(gt_future, pred_seq),
#                 'k_gdtw': calc_dtw(gt_future, pred_seq, geo=True),
#                 'k_gdtw2': calc_dtw(gt_future, pred_seq, geo=True, relaxed=True),
                
#                 'k_stay_rate': s_rate if s_rate is not None else np.nan,
#                 'k_stay_len_diff': s_ldiff if s_ldiff is not None else np.nan,
#                 'k_stay_len_abs': s_labs if s_labs is not None else np.nan,
#                 'k_stay_dist': s_loc if s_loc is not None else np.nan,
#                 'k_stay_cost': s_cost if s_cost is not None else np.nan,
                
#                 'k_dwell_u': dwell_metrics['dwell_u'],
#                 'k_dwell_u_all': dwell_metrics['dwell_u_all'],
#                 'k_dwell_u_realtime': dwell_metrics['dwell_u_realtime'],
#                 'k_dwell_u_all_realtime': dwell_metrics['dwell_u_all_realtime'],
#                 'k_node_accuracy': dwell_metrics['node_accuracy'],
                
#                 'prompt': prompt_seq,
#                 'gt': gt_future,
#                 'pred': pred_seq,
#             })
        
#         df = pd.DataFrame(metrics_list)
#         csv_path = os.path.join(CONFIG["output_dir"], f"metrics_cf_prefix{prefix_len}.csv")
#         df.to_csv(csv_path, index=False)
#         print(f"Saved metrics to: {csv_path}")
        
#         # 簡易集計表示
#         print(f"\nPrefix {prefix_len} Summary:")
#         print(f"  Overall Prob: {df['k_prob'].mean():.4f}")
        
#         # Night Effect
#         df_night = df[df['is_night'] == 1]
#         if len(df_night) > 0:
#             print(f"  [Night Data N={len(df_night)}]")
#             print(f"    Avg Prob (Night=1): {df_night['k_prob'].mean():.4f}")
#             print(f"    Avg Prob (Night=0): {df_night['k_prob_cf_night_off'].mean():.4f}")
#             print(f"    Night Impact (Diff): {df_night['diff_night'].mean():.4f}")
        
#         # Plaza Effect
#         df_plaza = df[(df['plaza_node2']==1) | (df['plaza_node11']==1) | (df['plaza_node14']==1)]
#         if len(df_plaza) > 0:
#             print(f"  [Plaza Data N={len(df_plaza)}]")
#             print(f"    Avg Prob (Plaza=ON):  {df_plaza['k_prob'].mean():.4f}")
#             print(f"    Avg Prob (Plaza=OFF): {df_plaza['k_prob_cf_plaza_off'].mean():.4f}")
#             print(f"    Plaza Impact (Diff):  {df_plaza['diff_plaza'].mean():.4f}")

#     print("\nEvaluation Complete!")

# if __name__ == "__main__":
#     main()