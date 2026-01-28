"""
Evaluation Utilities for Prism-RL

評価指標の計算関数
- DwellU (Dwell Overlap Utility): 滞在場所・時間の一致度
- DwellU RealTime: 圧縮を復元した実時間版
- その他従来指標（ED, DTW, Stay metrics）
"""

import numpy as np
from typing import List, Dict, Tuple


def get_decompressed_steps(compressed_count: int) -> float:
    """
    圧縮トークン数を実時間（平均）に戻す
    
    データ整形時の圧縮:
      1-2ステップ   → 圧縮count 1
      3-5ステップ   → 圧縮count 2
      6-10ステップ  → 圧縮count 3
      11-20ステップ → 圧縮count 4
      21-60ステップ → 圧縮count 5
      61-119ステップ → 圧縮count 6
      120+ステップ  → 圧縮count 7
    
    Args:
        compressed_count: 連続する同じ滞在トークンの個数
    
    Returns:
        estimated_steps: 推定実時間（各ビンの平均値 × count）
    """
    # 各ビンの平均値（実時間/トークン）
    bin_averages = {
        1: (1 + 2) / 2,      # 1.5
        2: (3 + 5) / 2,      # 4.0
        3: (6 + 10) / 2,     # 8.0
        4: (11 + 20) / 2,    # 15.5
        5: (21 + 60) / 2,    # 40.5
        6: (61 + 119) / 2,   # 90.0
        7: 120,              # 120+ (下限を使用)
    }
    
    if compressed_count == 0:
        return 0.0
    
    # 平均ステップ数/トークン × トークン数
    avg_steps_per_token = bin_averages.get(compressed_count, 1.5)
    return avg_steps_per_token * compressed_count


def calculate_dwell_overlap(pred_seq: List[int], 
                            gt_seq: List[int],
                            exclude_moves: bool = True,
                            base_N: int = 19,
                            use_realtime: bool = False) -> float:
    """
    DwellU (Dwell Overlap Utility) を計算
    
    各ノードでの滞在ステップ数の重複度を測定
    
    DwellU = Σ min(pred_dwell[n], gt_dwell[n]) / Σ max(pred_dwell[n], gt_dwell[n])
    
    1.0: すべての場所で滞在量が完全一致
    0.0: 滞在配分が全く被っていない
    
    Args:
        pred_seq: 予測系列（ノードのリスト）
        gt_seq: 正解系列（トークンのリスト、19以上は滞在）
        exclude_moves: Trueなら移動トークン(0-18)を除外して滞在のみカウント
        base_N: Move/Stayの境界
        use_realtime: Trueなら圧縮を復元して実時間で計算
    
    Returns:
        dwell_u: DwellU スコア [0.0, 1.0]
    """
    # 正解系列をノードIDに変換
    gt_nodes = []
    for token in gt_seq:
        if token < base_N:
            node = token  # Move token
        elif token < base_N * 2:
            node = token - base_N  # Stay token
        else:
            continue  # PAD/ENDは無視
        gt_nodes.append(node)
    
    # 各ノードでの滞在ステップ数をカウント
    if use_realtime:
        # 実時間版: 圧縮を復元
        pred_dwell = count_realtime_stays(pred_seq, exclude_moves)
        gt_dwell = count_realtime_stays(gt_nodes, exclude_moves)
    else:
        # 圧縮版: トークン数そのまま
        if exclude_moves:
            pred_dwell = count_consecutive_stays(pred_seq)
            gt_dwell = count_consecutive_stays(gt_nodes)
        else:
            pred_dwell = {}
            for node in pred_seq:
                pred_dwell[node] = pred_dwell.get(node, 0) + 1
            gt_dwell = {}
            for node in gt_nodes:
                gt_dwell[node] = gt_dwell.get(node, 0) + 1
    
    # 全ノードの和集合
    all_nodes = set(pred_dwell.keys()) | set(gt_dwell.keys())
    
    if len(all_nodes) == 0:
        return 0.0
    
    # 分子: Σ min(pred, gt)
    numerator = 0
    for node in all_nodes:
        pred_count = pred_dwell.get(node, 0)
        gt_count = gt_dwell.get(node, 0)
        numerator += min(pred_count, gt_count)
    
    # 分母: Σ max(pred, gt)
    denominator = 0
    for node in all_nodes:
        pred_count = pred_dwell.get(node, 0)
        gt_count = gt_dwell.get(node, 0)
        denominator += max(pred_count, gt_count)
    
    if denominator == 0:
        return 0.0
    
    dwell_u = numerator / denominator
    return dwell_u


def count_consecutive_stays(seq: List[int]) -> Dict[int, int]:
    """
    連続滞在のみをカウント（移動は除外）
    
    例: [1, 2, 2, 2, 3, 4, 4] -> {2: 3, 4: 2}
    
    Args:
        seq: ノード系列
    
    Returns:
        dwell_counts: {node: consecutive_count}
    """
    dwell_counts = {}
    
    i = 0
    while i < len(seq):
        node = seq[i]
        count = 1
        
        # 連続する同じノードをカウント
        j = i + 1
        while j < len(seq) and seq[j] == node:
            count += 1
            j += 1
        
        # 2ステップ以上連続する場合のみカウント（滞在と判定）
        if count >= 2:
            dwell_counts[node] = dwell_counts.get(node, 0) + count
        
        i = j
    
    return dwell_counts


def count_realtime_stays(seq: List[int], exclude_moves: bool = True) -> Dict[int, float]:
    """
    実時間復元版の滞在カウント
    
    連続滞在トークンを実時間に復元してカウント
    
    Args:
        seq: ノード系列
        exclude_moves: Trueなら連続滞在（2+ステップ）のみ
    
    Returns:
        dwell_counts: {node: estimated_realtime_steps}
    """
    dwell_counts = {}
    
    i = 0
    while i < len(seq):
        node = seq[i]
        count = 1
        
        # 連続する同じノードをカウント
        j = i + 1
        while j < len(seq) and seq[j] == node:
            count += 1
            j += 1
        
        # 実時間に復元
        if exclude_moves and count >= 2:
            # 滞在のみ: 2ステップ以上連続
            realtime_steps = get_decompressed_steps(count)
            dwell_counts[node] = dwell_counts.get(node, 0.0) + realtime_steps
        elif not exclude_moves:
            # 移動含む全体
            if count >= 2:
                # 連続滞在 → 復元
                realtime_steps = get_decompressed_steps(count)
            else:
                # 単発移動 → 1ステップ
                realtime_steps = 1.0
            dwell_counts[node] = dwell_counts.get(node, 0.0) + realtime_steps
        
        i = j
    
    return dwell_counts


def extract_stay_segments(seq: List[int], 
                         stay_offset: int = 19) -> List[Dict]:
    """
    滞在セグメントを抽出
    
    Args:
        seq: トークン系列
        stay_offset: 滞在トークンのオフセット
    
    Returns:
        segments: [{'node': int, 'start': int, 'length': int}, ...]
    """
    segments = []
    i = 0
    
    while i < len(seq):
        token = seq[i]
        
        # 滞在トークンかチェック
        if token >= stay_offset and token < stay_offset * 2:
            node = token - stay_offset
            start = i
            length = 1
            
            # 連続する同じトークンをカウント
            j = i + 1
            while j < len(seq) and seq[j] == token:
                length += 1
                j += 1
            
            segments.append({
                'node': node,
                'start': start,
                'length': length
            })
            
            i = j
        else:
            i += 1
    
    return segments


def node_sequence_to_tokens(node_seq: List[int], 
                           base_N: int = 19) -> List[int]:
    """
    ノード系列をトークン系列に変換
    
    連続する同じノードは Stay token に変換
    
    Args:
        node_seq: ノードのリスト [1, 2, 2, 2, 3]
        base_N: Move/Stay の境界
    
    Returns:
        tokens: トークンのリスト [1(move), 21(stay), 21(stay), 21(stay), 3(move)]
    """
    if len(node_seq) == 0:
        return []
    
    tokens = []
    
    for i, node in enumerate(node_seq):
        if i == 0:
            # 最初は必ず Move
            tokens.append(node)
        else:
            if node == node_seq[i-1]:
                # 同じノード → Stay token
                tokens.append(node + base_N)
            else:
                # 異なるノード → Move token
                tokens.append(node)
    
    return tokens


def calc_node_level_accuracy(pred_seq: List[int],
                             gt_seq: List[int],
                             base_N: int = 19) -> float:
    """
    ノードレベルの精度（場所が合っているか）
    
    各ステップで正しいノードにいるかをカウント
    
    Args:
        pred_seq: 予測系列（ノードのリスト）
        gt_seq: 正解系列（トークンのリスト）
        base_N: Move/Stayの境界
    
    Returns:
        accuracy: [0.0, 1.0]
    """
    # 正解系列をノードIDに変換
    gt_nodes = []
    for token in gt_seq:
        if token < base_N:
            node = token
        elif token < base_N * 2:
            node = token - base_N
        else:
            continue
        gt_nodes.append(node)
    
    # 長さを揃える
    min_len = min(len(pred_seq), len(gt_nodes))
    
    if min_len == 0:
        return 0.0
    
    # 一致カウント
    correct = 0
    for i in range(min_len):
        if pred_seq[i] == gt_nodes[i]:
            correct += 1
    
    # 長さの差もペナルティ
    total = max(len(pred_seq), len(gt_nodes))
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def summarize_dwell_metrics(pred_seq: List[int],
                            gt_seq: List[int],
                            base_N: int = 19) -> Dict[str, float]:
    """
    滞在関連の指標をまとめて計算
    
    Args:
        pred_seq: 予測系列（ノードのリスト）
        gt_seq: 正解系列（トークンのリスト）
        base_N: Move/Stayの境界
    
    Returns:
        metrics: {
            'dwell_u': DwellU (圧縮版、滞在のみ),
            'dwell_u_all': DwellU (圧縮版、移動含む),
            'dwell_u_realtime': DwellU (実時間版、滞在のみ),
            'dwell_u_all_realtime': DwellU (実時間版、移動含む),
            'node_accuracy': ノードレベル精度,
        }
    """
    metrics = {}
    
    # DwellU (圧縮版、滞在のみ)
    metrics['dwell_u'] = calculate_dwell_overlap(
        pred_seq, gt_seq, exclude_moves=True, base_N=base_N, use_realtime=False
    )
    
    # DwellU (圧縮版、移動含む全体)
    metrics['dwell_u_all'] = calculate_dwell_overlap(
        pred_seq, gt_seq, exclude_moves=False, base_N=base_N, use_realtime=False
    )
    
    # DwellU (実時間版、滞在のみ)
    metrics['dwell_u_realtime'] = calculate_dwell_overlap(
        pred_seq, gt_seq, exclude_moves=True, base_N=base_N, use_realtime=True
    )
    
    # DwellU (実時間版、移動含む全体)
    metrics['dwell_u_all_realtime'] = calculate_dwell_overlap(
        pred_seq, gt_seq, exclude_moves=False, base_N=base_N, use_realtime=True
    )
    
    # ノードレベル精度
    metrics['node_accuracy'] = calc_node_level_accuracy(
        pred_seq, gt_seq, base_N=base_N
    )
    
    return metrics


if __name__ == "__main__":
    # テスト
    print("="*60)
    print("Evaluation Utilities Test")
    print("="*60)
    
    # テストデータ
    # 正解: [1(move), 21(stay), 21(stay), 3(move), 4(move), 23(stay)]
    # → ノード2に2ステップ滞在、ノード4に1ステップ滞在
    gt_seq = [1, 21, 21, 3, 4, 23]
    
    # 予測: [1, 2, 2, 3, 5, 5]
    # → ノード2に2ステップ滞在、ノード5に2ステップ滞在
    pred_seq = [1, 2, 2, 3, 5, 5]
    
    print("\nGround Truth (tokens):", gt_seq)
    print("Prediction (nodes):   ", pred_seq)
    
    # DwellU計算（圧縮版）
    dwell_u = calculate_dwell_overlap(pred_seq, gt_seq, exclude_moves=True, use_realtime=False)
    print(f"\nDwellU (stays only, compressed):        {dwell_u:.4f}")
    
    dwell_u_all = calculate_dwell_overlap(pred_seq, gt_seq, exclude_moves=False, use_realtime=False)
    print(f"DwellU (all steps, compressed):         {dwell_u_all:.4f}")
    
    # DwellU計算（実時間版）
    dwell_u_rt = calculate_dwell_overlap(pred_seq, gt_seq, exclude_moves=True, use_realtime=True)
    print(f"DwellU (stays only, realtime):          {dwell_u_rt:.4f}")
    
    dwell_u_all_rt = calculate_dwell_overlap(pred_seq, gt_seq, exclude_moves=False, use_realtime=True)
    print(f"DwellU (all steps, realtime):           {dwell_u_all_rt:.4f}")
    
    # ノード精度
    node_acc = calc_node_level_accuracy(pred_seq, gt_seq)
    print(f"Node Accuracy:                           {node_acc:.4f}")
    
    # まとめて計算
    metrics = summarize_dwell_metrics(pred_seq, gt_seq)
    print("\nAll metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # 実時間復元のテスト
    print("\n" + "="*60)
    print("RealTime Decompression Test")
    print("="*60)
    for count in range(1, 8):
        realtime = get_decompressed_steps(count)
        print(f"Compressed count {count} → Estimated realtime: {realtime:.1f} steps")