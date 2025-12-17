import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.metrics.distance import edit_distance
from itertools import groupby

# ==========================================
# ファイルパス設定
# ==========================================
teacher_path = "/home/mizutani/projects/RF/runs/20251216_230505/teacher_20251216_230505.csv"         # 正解データ（Ground Truth）
result_path = "/home/mizutani/projects/RF/runs/20251216_230505/result_ordinary_20251216_230505.csv"  # 推論結果（Prediction）

# 特殊トークンの定義（あなたの環境に合わせて調整してください）
PAD_TOKEN = 19
START_TOKEN = 21  # <b> (Network.N + 2)
END_TOKEN = 20    # <e> (Network.N + 1)
MASK_TOKEN = 22   # <m>

def clean_sequence(seq):
    """
    評価のためにパディングや特殊トークンを除去し、純粋なルート配列にする
    """
    # 数値型に変換し、リスト化
    seq = [int(x) for x in seq]
    
    # 除外したいトークンのセット
    ignore_tokens = {PAD_TOKEN, START_TOKEN, MASK_TOKEN, END_TOKEN}
    
    # トークン除去
    cleaned = [x for x in seq if x not in ignore_tokens]
    return cleaned

def compress_consecutive(seq):
    """
    連続する重複を削除して、純粋な移動経路にする
    例: [1, 1, 1, 2, 2, 3] -> [1, 2, 3]
    """
    return [k for k, g in groupby(seq)]

def calculate_spatial_metrics(teacher_df, result_df):
    bleu_scores = []
    edit_distances = []
    normalized_edit_distances = []
    
    # 行ごとに比較
    num_samples = len(teacher_df)
    for i in range(num_samples):
        # まず通常のクリーニング
        gt_seq_raw = clean_sequence(teacher_df.iloc[i].values)
        pred_seq_raw = clean_sequence(result_df.iloc[i].values)
        
        # ★ここで連続重複を削除（空間的なルートのみ抽出）★
        gt_seq = compress_consecutive(gt_seq_raw)
        pred_seq = compress_consecutive(pred_seq_raw)
        
        # --- 以下、同じ計算 ---
        # BLEU
        if len(gt_seq) > 1 and len(pred_seq) > 1: # 短すぎるとBLEUは計算不能になりやすい
             score = sentence_bleu([gt_seq], pred_seq, smoothing_function=SmoothingFunction().method1)
             bleu_scores.append(score)
        
        # Edit Distance
        dist = edit_distance(gt_seq, pred_seq)
        edit_distances.append(dist)
        
        # Normalized Edit Distance
        max_len = max(len(gt_seq), len(pred_seq))
        if max_len > 0:
            norm_dist = dist / max_len
        else:
            norm_dist = 0.0
        normalized_edit_distances.append(norm_dist)

    return bleu_scores, edit_distances, normalized_edit_distances

# 実行
teacher_df = pd.read_csv(teacher_path)
result_df = pd.read_csv(result_path)

bleu_sp, ed_sp, norm_ed_sp = calculate_spatial_metrics(teacher_df, result_df)

print("\n=== 【空間評価】Spatial Evaluation Metrics (重複排除後) ===")
print(f"BLEU Score (Average): {np.mean(bleu_sp):.4f}")
print(f"Edit Distance (Average): {np.mean(ed_sp):.4f}")
print(f"Normalized Edit Dist (Average): {np.mean(norm_ed_sp):.4f}")