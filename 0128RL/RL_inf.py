"""
Prism-RL Evaluation Script

Prism-RLモデルの経路生成能力を評価
DKP_RF_inf.py と同じ評価指標で比較

評価指標:
- 次トークン予測: N/A (Prism-RLは系列全体を生成)
- Edit Distance (ED, Geo-ED, Geo-ED2)
- DTW (DTW, Geo-DTW, Geo-DTW2)
- Stay metrics (Detection Rate, Length Diff, Location Distance)
- DwellU (NEW): 滞在場所・時間の一致度
"""

import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from prism_rl_generator import PrismRLGenerator, load_prism_rl_params
from utils_prism import (
    calculate_dwell_overlap,
    node_sequence_to_tokens,
    summarize_dwell_metrics
)

# 従来の評価関数をインポート（DKP_RF_inf.pyから流用）
import networkx as nx


# =========================================================
# Configuration
# =========================================================

CONFIG = {
    # データパス
    "data_npz_path": "/home/mizutani/projects/RF/data/input_real_m5.npz",
    "split_indices_path": "/home/mizutani/projects/RF/data/common_split_indices_m5.npz",
    "prism_params_csv":"/home/mizutani/projects/RF/0128RL/data/okinawa_output/results/PrismRL_est_test_20260128T2339.csv",
    
    # Prism-RLモデル
    "link_csv": "link.csv",
    "node_csv": "node.csv",
    
    # Prism-RL設定
    "tau": 3.0,      # Detour rate
    "J": 3,          # 最小 choice stage
    
    # 評価設定
    "eval_data": "TEST",  # TRAIN, VAL, TEST
    "prefix_lengths": [5],  # 複数のPrefix長で評価可能
    "short_long_threshold": 8,
    
    # 出力
    "output_dir": "/home/mizutani/projects/RF/0128RL/prism_rl_evaluation",
    "plot_max_samples": 1000,
    
    # その他
    "pad_token": 38,
    "base_N": 19,
}

# 2-hop Adjacency Map (従来と同じ)
ADJACENCY_MAP = {
    0: [1, 2, 4, 11], 1: [0, 2, 4, 5, 9], 2: [0, 1, 5, 6, 7],
    4: [0, 1, 5, 8, 9, 10, 11], 5: [1, 2, 4, 6, 10], 6: [2, 5, 7, 10, 14],
    7: [2, 6, 13, 14, 15], 8: [4, 9, 11], 9: [1, 4, 8, 10, 12],
    10: [4, 5, 6, 9, 12, 13], 11: [0, 4, 8], 12: [9, 10, 13],
    13: [7, 10, 12, 14, 15], 14: [6, 7, 13, 15, 16], 15: [7, 13, 14],
    16: [14, 17, 18], 17: [16, 18], 18: [16, 17]
}


# =========================================================
# Logger
# =========================================================

class Logger(object):
    """ターミナル出力とファイル出力を同時に行う"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# =========================================================
# Helper Functions (従来のDKP_RF_inf.pyから流用)
# =========================================================

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
    """Edit Distance"""
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
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )
    
    return dp[n][m]


def calc_dtw(seq1, seq2, geo=False, relaxed=False, base_N=19, node_distances=None, adjacency_map=None):
    """DTW"""
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
    """滞在指標のペア計算"""
    gt_stays = extract_stays(gt_seq, stay_offset)
    pred_stays = extract_stays(pred_seq, stay_offset)
    
    if len(gt_stays) == 0:
        return None
    
    matched = []
    for gs in gt_stays:
        best_match = None
        best_dist = float('inf')
        
        for ps in pred_stays:
            if ps['node'] == gs['node']:
                dist = abs(ps['start'] - gs['start'])
                if dist < best_dist:
                    best_dist = dist
                    best_match = ps
        
        if best_match is not None:
            matched.append({
                'gt': gs,
                'pred': best_match,
                'len_diff': best_match['length'] - gs['length'],
                'loc_dist': 0 if best_match['node'] == gs['node'] else node_distances.get(gs['node'], {}).get(best_match['node'], 1)
            })
    
    return matched


def summarize_stay(matched):
    """滞在指標のサマリー"""
    if matched is None or len(matched) == 0:
        return None, None, None, None, None
    
    rate = len(matched)
    len_diff_avg = np.mean([m['len_diff'] for m in matched])
    len_diff_abs = np.mean([abs(m['len_diff']) for m in matched])
    loc_dist = np.mean([m['loc_dist'] for m in matched])
    cost = (1 - rate) + len_diff_abs + loc_dist
    
    return rate, len_diff_avg, len_diff_abs, loc_dist, cost


# =========================================================
# Main Evaluation
# =========================================================

def main():
    print("="*60)
    print("Prism-RL Evaluation Script")
    print("="*60)
    
    # 出力ディレクトリ作成
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # ログファイル設定
    log_path = os.path.join(CONFIG["output_dir"], "terminal_log.txt")
    sys.stdout = Logger(log_path)
    print(f"Log saved to: {log_path}")
    print("="*60)
    
    # =====================================================
    # データ読み込み
    # =====================================================
    print("\n=== Loading Data ===")
    
    data = np.load(CONFIG["data_npz_path"])
    routes = data['route_arr']
    
    # 分割インデックス
    split_data = np.load(CONFIG["split_indices_path"])
    train_indices = split_data['train_sequences']
    val_indices = split_data['val_sequences']
    test_indices = split_data['test_sequences']
    
    print(f"Train sequences: {len(train_indices)}")
    print(f"Val sequences: {len(val_indices)}")
    print(f"Test sequences: {len(test_indices)}")
    
    # 評価データ選択
    eval_data_name = CONFIG["eval_data"].upper()
    if eval_data_name == "TRAIN":
        eval_indices = train_indices
    elif eval_data_name == "VAL":
        eval_indices = val_indices
    elif eval_data_name == "TEST":
        eval_indices = test_indices
    else:
        raise ValueError(f"Invalid eval_data: {CONFIG['eval_data']}")
    
    print(f"\nEvaluating on: {eval_data_name} ({len(eval_indices)} sequences)")
    
    # ノード距離行列の計算
    print("\n=== Computing Node Distances ===")
    link_df = pd.read_csv(CONFIG["link_csv"])
    
    # NetworkX グラフ構築
    G = nx.Graph()
    for _, row in link_df.iterrows():
        O, D = row['O'], row['D']
        if O != D:  # 自己ループは除外
            G.add_edge(O, D)
    
    # 最短距離
    all_nodes = sorted(G.nodes())
    NODE_DISTANCES = {}
    for node in all_nodes:
        lengths = dict(nx.single_source_shortest_path_length(G, node))
        NODE_DISTANCES[node] = lengths
    
    base_N = CONFIG["base_N"]
    
    # =====================================================
    # Prism-RLモデル読み込み
    # =====================================================
    print("\n=== Loading Prism-RL Model ===")
    
    node_df = pd.read_csv(CONFIG["node_csv"])
    params = load_prism_rl_params(CONFIG["prism_params_csv"])
    
    print(f"Parameters: {params}")
    
    generator = PrismRLGenerator(
        link_df=link_df,
        node_df=node_df,
        params=params,
        tau=CONFIG["tau"],
        J=CONFIG["J"]
    )
    
    # =====================================================
    # 評価ループ
    # =====================================================
    print("\n=== Starting Evaluation ===")
    print(f"Prefix lengths: {CONFIG['prefix_lengths']}")
    print(f"Short/Long threshold: {CONFIG['short_long_threshold']}")
    
    all_results = []
    
    for prefix_len in CONFIG["prefix_lengths"]:
        print(f"\n--- Evaluating with Prefix Length = {prefix_len} ---")
        
        metrics_list = []
        
        for idx in tqdm(eval_indices, desc=f"Prefix={prefix_len}"):
            seq = routes[idx]
            
            # パディングを除去
            pad_indices = np.where(seq == CONFIG["pad_token"])[0]
            seq_len = pad_indices[0] if len(pad_indices) > 0 else len(seq)
            seq = seq[:seq_len]
            
            # Prefix長が十分か確認
            if seq_len <= prefix_len:
                continue
            
            # Prefix と Future に分割
            prefix_tokens = seq[:prefix_len]
            gt_future = seq[prefix_len:]
            
            if len(gt_future) == 0:
                continue
            
            # PrefixをノードIDに変換
            prefix_nodes = []
            for token in prefix_tokens:
                if token < base_N:
                    node = token
                elif token < base_N * 2:
                    node = token - base_N
                else:
                    continue
                prefix_nodes.append(node)
            
            # 正解FutureをノードIDに変換
            gt_future_nodes = []
            for token in gt_future:
                if token < base_N:
                    node = token
                elif token < base_N * 2:
                    node = token - base_N
                else:
                    continue
                gt_future_nodes.append(node)
            
            # O→D を取得
            if len(prefix_nodes) == 0 or len(gt_future_nodes) == 0:
                continue
            
            origin = prefix_nodes[-1]  # Prefixの最後
            destination = gt_future_nodes[-1]  # 正解の最後
            
            # Prism-RLで経路生成
            try:
                pred_path = generator.generate_path(origin, destination)
                
                # Prefix除去（originを除く）
                if len(pred_path) > 1:
                    pred_future_nodes = pred_path[1:]  # originを除く
                else:
                    pred_future_nodes = []
                
            except Exception as e:
                print(f"\nWarning: Failed to generate path for sample {idx}: {e}")
                continue
            
            if len(pred_future_nodes) == 0:
                continue
            
            # 予測系列をトークンに変換
            pred_future_tokens = node_sequence_to_tokens(pred_future_nodes, base_N)
            
            gen_len = len(gt_future)
            
            # Edit Distance
            ed = calc_levenshtein(gt_future.tolist(), pred_future_tokens, base_N=base_N)
            ged = calc_levenshtein(gt_future.tolist(), pred_future_tokens, geo=True, base_N=base_N, node_distances=NODE_DISTANCES)
            ged2 = calc_levenshtein(gt_future.tolist(), pred_future_tokens, geo=True, relaxed=True, base_N=base_N, adjacency_map=ADJACENCY_MAP)
            
            # DTW
            dtw = calc_dtw(gt_future.tolist(), pred_future_tokens, base_N=base_N)
            gdtw = calc_dtw(gt_future.tolist(), pred_future_tokens, geo=True, base_N=base_N, node_distances=NODE_DISTANCES)
            gdtw2 = calc_dtw(gt_future.tolist(), pred_future_tokens, geo=True, relaxed=True, base_N=base_N, adjacency_map=ADJACENCY_MAP)
            
            # 滞在指標
            stay_metrics = calc_stay_metrics_pair(gt_future.tolist(), pred_future_tokens, NODE_DISTANCES, CONFIG["base_N"])
            rate, ldiff, labs, loc, cost = summarize_stay(stay_metrics)
            
            # DwellU（新指標）
            dwell_metrics = summarize_dwell_metrics(pred_future_nodes, gt_future.tolist(), base_N=base_N)
            
            # メトリクス保存
            metrics_list.append({
                'id': idx,
                'prefix_len': prefix_len,
                'gen_len': gen_len,
                'origin': origin,
                'destination': destination,
                
                # Edit Distance
                'ed': ed,
                'ged': ged,
                'ged2': ged2,
                
                # DTW
                'dtw': dtw,
                'gdtw': gdtw,
                'gdtw2': gdtw2,
                
                # 滞在指標
                'stay_rate': rate if rate is not None else np.nan,
                'stay_len_diff': ldiff if ldiff is not None else np.nan,
                'stay_len_abs': labs if labs is not None else np.nan,
                'stay_dist': loc if loc is not None else np.nan,
                'stay_cost': cost if cost is not None else np.nan,
                
                # DwellU（新指標）
                'dwell_u': dwell_metrics['dwell_u'],
                'dwell_u_all': dwell_metrics['dwell_u_all'],
                'node_accuracy': dwell_metrics['node_accuracy'],
                
                # 系列保存
                'gt': gt_future.tolist(),
                'pred': pred_future_tokens,
            })
        
        # DataFrameに変換
        df = pd.DataFrame(metrics_list)
        
        # CSV保存
        csv_path = os.path.join(CONFIG["output_dir"], f"metrics_prism_rl_prefix{prefix_len}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved metrics to: {csv_path}")
        
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
        
        # Edit Distance
        ed = (d['ed'] / d['gen_len']).mean()
        ged = (d['ged'] / d['gen_len']).mean()
        ged2 = (d['ged2'] / d['gen_len']).mean()
        
        # DTW
        dtw_val = (d['dtw'] / d['gen_len']).mean()
        gdtw = (d['gdtw'] / d['gen_len']).mean()
        gdtw2 = (d['gdtw2'] / d['gen_len']).mean()
        
        print(f"  [Gen] ED (norm):       {ed:.4f}")
        print(f"  [Gen] Geo-ED (norm):   {ged:.4f}")
        print(f"  [Gen] Geo-ED2 (norm):  {ged2:.4f}  (Relaxed: 0 if <=2hop)")
        print(f"  [Gen] DTW (norm):      {dtw_val:.4f}")
        print(f"  [Gen] Geo-DTW (norm):  {gdtw:.4f}")
        print(f"  [Gen] Geo-DTW2 (norm): {gdtw2:.4f}  (Relaxed: 0 if <=2hop)")
        
        # 滞在指標
        print("  [Stay Metrics] (Detected Stays Only)")
        print(f"    Detection Rate:      {d['stay_rate'].mean():.4f}")
        print(f"    Length Diff (avg):   {d['stay_len_diff'].mean():.4f}")
        print(f"    Length Diff (Abs):   {d['stay_len_abs'].mean():.4f}")
        print(f"    Loc Dist (hops):     {d['stay_dist'].mean():.4f}")
        print(f"    Integrated Cost:     {d['stay_cost'].mean():.4f}")
        
        # DwellU (新指標)
        print("  [DwellU Metrics] (NEW)")
        print(f"    DwellU (stays only): {d['dwell_u'].mean():.4f}")
        print(f"    DwellU (all steps):  {d['dwell_u_all'].mean():.4f}")
        print(f"    Node Accuracy:       {d['node_accuracy'].mean():.4f}")
    
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
        
        plot_pdf_path = os.path.join(CONFIG["output_dir"], f"trajectories_prism_rl_prefix{prefix_len}.pdf")
        print(f"Generating PDF for prefix={prefix_len} (first {CONFIG['plot_max_samples']} samples)...")
        
        with PdfPages(plot_pdf_path) as pdf:
            n_plot = min(len(df), CONFIG["plot_max_samples"])
            
            for i in range(n_plot):
                row = df.iloc[i]
                gt = row['gt']
                pred = row['pred']
                
                # トークンをノードIDに変換（可視化用）
                gt_nodes = []
                for token in gt:
                    if token < base_N:
                        gt_nodes.append(token)
                    elif token < base_N * 2:
                        gt_nodes.append(token - base_N)
                
                pred_nodes = []
                for token in pred:
                    if token < base_N:
                        pred_nodes.append(token)
                    elif token < base_N * 2:
                        pred_nodes.append(token - base_N)
                
                fig, ax = plt.subplots(figsize=(12, 4))
                
                ax.plot(range(len(gt_nodes)), gt_nodes, 's-', label='Ground Truth', color='green', linewidth=2)
                ax.plot(range(len(pred_nodes)), pred_nodes, '^-', label='Prism-RL Pred', color='blue', linewidth=1.5, alpha=0.7)
                
                ax.set_xlabel('Step')
                ax.set_ylabel('Node ID')
                ax.set_title(f'Sample {i+1} | Prefix={prefix_len} | GT Len={len(gt)}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
        
        print(f"Saved to: {plot_pdf_path}")
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print(f"Results saved to: {CONFIG['output_dir']}")
    print("="*60)


if __name__ == "__main__":
    main()