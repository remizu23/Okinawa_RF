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
from matplotlib.backends.backend_pdf import PdfPages # ★追加: PDF出力用
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

# 入力データパス
REAL_DATA_PATH = "/home/mizutani/projects/RF/data/input_real_test.npz"

# モデルパス
MODEL_KOOPMAN_PATH = "/home/mizutani/projects/RF/runs/20260111_173552/model_weights_20260111_173552.pth"
MODEL_NORMAL_PATH  = "/home/mizutani/projects/RF/runs/20260111_020209/model_weights_20260111_020209.pth"


# =========================================================
# 0. 保存先設定
# =========================================================
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"/home/mizutani/projects/RF/runs/eval_real_pdf_{run_id}"
os.makedirs(out_dir, exist_ok=True)

print(f"=== Evaluation Started: {run_id} ===")
def save_log(msg):
    print(msg)
    with open(os.path.join(out_dir, "evaluation_log.txt"), "a") as f:
        f.write(msg + "\n")

# =========================================================
# 1. データ読み込み関数
# =========================================================
def load_real_test_data(npz_path):
    save_log(f"Loading real test data from {npz_path}...")
    if not os.path.exists(npz_path):
        save_log(f"Error: File {npz_path} not found.")
        return []

    try:
        data = np.load(npz_path)
        route_arr = data['route_arr'] 
    except Exception as e:
        save_log(f"Error loading npz: {e}")
        return []

    test_data = []
    for i, seq in enumerate(route_arr):
        valid_traj = [int(x) for x in seq if x != PAD_TOKEN]
        test_data.append({
            'agent_id': i,
            'type': 'real',
            'trajectory': valid_traj
        })
    
    save_log(f"Loaded {len(test_data)} trajectories.")
    return test_data

# =========================================================
# 2. モデルロード関数
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

    model = KoopmanRoutesFormer(
        vocab_size=config.get('vocab_size', VOCAB_SIZE),
        token_emb_dim=config['token_emb_dim'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        z_dim=config['z_dim'],
        pad_token_id=config.get('pad_token_id', PAD_TOKEN),
        num_agents=config.get('num_agents', 1),
        agent_emb_dim=config.get('agent_emb_dim', 16),
        max_stay_count=config.get('max_stay_count', 500),
        stay_emb_dim=config.get('stay_emb_dim', 16)
    )
    
    model.has_extra_inputs = ('agent_emb_dim' in config) or (hasattr(model, 'agent_embedding'))

    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    except RuntimeError as e:
        save_log(f"Critical Error in load_state_dict: {e}")
        raise e    
    model.to(device)
    model.eval()
    return model, config

# =========================================================
# 3. 推論 & 評価
# =========================================================
def calculate_stay_counts_seq(seq):
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
    model.eval()
    current_seq = initial_seq.copy()
    needs_extra = getattr(model, 'has_extra_inputs', False)
    
    with torch.no_grad():
        for _ in range(predict_len):
            input_tensor = torch.tensor([current_seq], dtype=torch.long).to(device)
            
            if needs_extra:
                current_counts = calculate_stay_counts_seq(current_seq)
                stay_tensor = torch.tensor([current_counts], dtype=torch.long).to(device)
                safe_agent_id = agent_id % model.agent_embedding.num_embeddings 
                agent_tensor = torch.tensor([safe_agent_id], dtype=torch.long).to(device)
                output = model(input_tensor, stay_tensor, agent_tensor)
            else:
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
    
    MIN_TRAJ_LEN = prompt_len + 5

    for i, data in enumerate(test_data):
        full_traj = data['trajectory']
        agent_id = data['agent_id']

        total_len = len(full_traj)
        if total_len <= MIN_TRAJ_LEN: continue
            
        prompt_seq = full_traj[:prompt_len]
        gt_future = full_traj[prompt_len:]
        pred_len = len(gt_future)
        
        # 推論
        pred_k_future = predict_trajectory(model_koopman, prompt_seq, pred_len, agent_id, device)
        pred_n_future = predict_trajectory(model_normal, prompt_seq, pred_len, agent_id, device)
        
        # 指標計算
        dist_k_lev = Levenshtein.distance(to_str(gt_future), to_str(pred_k_future))
        dist_n_lev = Levenshtein.distance(to_str(gt_future), to_str(pred_n_future))
        
        dist_k_dtw = calc_dtw(gt_future, pred_k_future)
        dist_n_dtw = calc_dtw(gt_future, pred_n_future)
        
        results.append({
            'id': agent_id,
            'type': data['type'],
            'total_len': total_len,
            'score_k_lev': dist_k_lev / pred_len, # Normalized
            'score_n_lev': dist_n_lev / pred_len,
            'score_k_dtw': dist_k_dtw / pred_len,
            'score_n_dtw': dist_n_dtw / pred_len,
            'prompt': prompt_seq,
            'gt': gt_future,
            'pred_k': pred_k_future,
            'pred_n': pred_n_future
        })
        
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(test_data)}...", end='\r')
            
    return pd.DataFrame(results)

# =========================================================
# 4. 可視化ロジック (★修正: PDF一括出力)
# =========================================================
def decode_for_plot(seq):
    arr = np.array(seq, dtype=float)
    is_stay = (arr >= STAY_OFFSET) & (arr < PAD_TOKEN)
    arr[is_stay] -= STAY_OFFSET
    arr[arr == PAD_TOKEN] = np.nan
    return arr

def save_all_plots_to_pdf(df_res, out_path):
    """
    データフレーム内の全結果をPDFにプロットする関数
    - 1ページあたり20枚 (5行x4列)
    - 各図にED, DTWスコアを明記
    """
    if df_res.empty:
        print("No results to plot.")
        return

    print(f"Plotting {len(df_res)} trajectories to PDF...")
    
    # PDF設定
    pdf = PdfPages(out_path)
    
    # ページ設定 (A4縦想定で適当なサイズに)
    ROWS = 5
    COLS = 4
    PLOTS_PER_PAGE = ROWS * COLS
    FIG_SIZE = (20, 25) # 大きめに確保
    
    # 全データをイテレート
    num_samples = len(df_res)
    num_pages = (num_samples + PLOTS_PER_PAGE - 1) // PLOTS_PER_PAGE
    
    for page in range(num_pages):
        fig, axes = plt.subplots(ROWS, COLS, figsize=FIG_SIZE)
        # axesを1次元配列にして扱いやすくする (1行の場合は例外処理が必要だが5行あるのでOK)
        axes = axes.flatten()
        
        start_idx = page * PLOTS_PER_PAGE
        end_idx = min((page + 1) * PLOTS_PER_PAGE, num_samples)
        
        # このページのデータを描画
        for i in range(PLOTS_PER_PAGE):
            curr_idx = start_idx + i
            ax = axes[i]
            
            if curr_idx < num_samples:
                row = df_res.iloc[curr_idx]
                _plot_on_axis(ax, row)
            else:
                # データがないマスは非表示
                ax.axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        print(f"Saved page {page+1}/{num_pages}", end='\r')
        
    pdf.close()
    print(f"\nPDF saved successfully: {out_path}")

def _plot_on_axis(ax, row):
    """
    個別のAxesに描画するヘルパー関数
    """
    prompt_len = len(row['prompt'])
    
    # データの結合とデコード
    full_gt = decode_for_plot(row['prompt'] + row['gt'])
    full_k  = decode_for_plot(row['prompt'] + row['pred_k'])
    full_n  = decode_for_plot(row['prompt'] + row['pred_n'])
    
    # Ground Truth
    ax.plot(full_gt, label='GT', color='black', linewidth=2.5, alpha=0.3)
    ax.axvline(x=prompt_len-0.5, color='gray', linestyle='--', linewidth=0.8)
    
    # Predictions
    x_range = range(prompt_len, len(full_gt))
    # 長さ調整
    len_k = min(len(x_range), len(full_k[prompt_len:]))
    len_n = min(len(x_range), len(full_n[prompt_len:]))
    
    # Koopman
    ax.plot(list(x_range)[:len_k], full_k[prompt_len:][:len_k], 
            label='Koopman', color='red', linewidth=1.2)
    
    # Normal
    ax.plot(list(x_range)[:len_n], full_n[prompt_len:][:len_n], 
            label='Normal', color='blue', linestyle=':', linewidth=1.2)
    
    # タイトル作成 (スコア表示)
    # K: Koopman, N: Normal, E: ED(Levenshtein), D: DTW
    title_str = (f"ID:{row['id']} Len:{row['total_len']}\n"
                 f"K [ED:{row['score_k_lev']:.2f}, DTW:{row['score_k_dtw']:.2f}]\n"
                 f"N [ED:{row['score_n_lev']:.2f}, DTW:{row['score_n_dtw']:.2f}]")
    
    ax.set_title(title_str, fontsize=9)
    ax.set_yticks(range(0, 19, 2)) # 目盛りを少し間引く
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.grid(True, alpha=0.3)
    
    # 凡例は小さく表示 (最初のプロットだけ、あるいは全部につけるか。全部につける)
    ax.legend(fontsize=7, loc='upper left')


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
        
    # 2. 実データ読み込み
    gt_data = load_real_test_data(REAL_DATA_PATH)
    
    # 3. 評価
    if gt_data:
        # prompt_len=15で評価
        df_res = evaluate_models(model_koopman, model_normal, gt_data, prompt_len=15, device=device)
        
        # グループ分けロジック
        def classify_length(l):
            if 30 <= l <= 60:
                return "Medium (30-60)"
            elif l > 60:
                return "Long (>60)"
            else:
                return "Short (<30)"

        df_res['len_group'] = df_res['total_len'].apply(classify_length)
        
        # CSV保存
        csv_path = os.path.join(out_dir, "evaluation_results.csv")
        df_res.to_csv(csv_path, index=False)
        
        # 全体平均ログ
        save_log("\n=== Overall Scores (Lower is Better) ===")
        save_log("Levenshtein (Norm):")
        save_log(df_res[['score_k_lev', 'score_n_lev']].mean().to_string())
        save_log("DTW (Norm):")
        save_log(df_res[['score_k_dtw', 'score_n_dtw']].mean().to_string())

        # ★PDF一括出力 (変更箇所)
        pdf_path = os.path.join(out_dir, "all_trajectories.pdf")
        save_all_plots_to_pdf(df_res, pdf_path)
        
        save_log("\nDone.")
    else:
        save_log("No valid test data found.")