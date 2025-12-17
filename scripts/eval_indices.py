import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.metrics.distance import edit_distance

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

def calculate_metrics(teacher_df, result_df):
    bleu_scores = []
    edit_distances = []
    normalized_edit_distances = []
    
    # 行ごとに比較
    num_samples = len(teacher_df)
    for i in range(num_samples):
        # 1行ずつ取得
        gt_seq = clean_sequence(teacher_df.iloc[i].values)
        pred_seq = clean_sequence(result_df.iloc[i].values)
        
        # --- BLEU Score ---
        # 参照(Ground Truth)はリストのリストで渡す必要がある
        # smoothing_functionは短い系列でスコアが0になるのを防ぐ
        score = sentence_bleu([gt_seq], pred_seq, smoothing_function=SmoothingFunction().method1)
        bleu_scores.append(score)
        
        # --- Edit Distance (Levenshtein Distance) ---
        # 挿入・削除・置換が何回必要か
        dist = edit_distance(gt_seq, pred_seq)
        edit_distances.append(dist)
        
        # 正規化編集距離 (0~1) : 距離 / 長い方の系列長
        # 0に近いほど良い（完全に一致）
        max_len = max(len(gt_seq), len(pred_seq))
        if max_len > 0:
            norm_dist = dist / max_len
        else:
            norm_dist = 0.0 # 両方空なら一致とみなす
        normalized_edit_distances.append(norm_dist)

    return bleu_scores, edit_distances, normalized_edit_distances

# 実行
teacher_df = pd.read_csv(teacher_path)
result_df = pd.read_csv(result_path)

# 行数が合っているか確認
min_len = min(len(teacher_df), len(result_df))
teacher_df = teacher_df.iloc[:min_len]
result_df = result_df.iloc[:min_len]

bleu, ed, norm_ed = calculate_metrics(teacher_df, result_df)

print(f"=== Evaluation Metrics (Sample size: {len(bleu)}) ===")
print(f"BLEU Score (Average): {np.mean(bleu):.4f}")
print(f"Edit Distance (Average): {np.mean(ed):.4f}")
print(f"Normalized Edit Dist (Average): {np.mean(norm_ed):.4f}")