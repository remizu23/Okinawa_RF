import torch
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import glob
from datetime import datetime
import torch.nn.functional as F
import networkx as nx

# ユーザー定義モジュール
from network import Network, expand_adjacency_matrix
from tokenization import Tokenization
from KP_RF import KoopmanRoutesFormer

# =========================================================
#  Config & Settings
# =========================================================
# ★可視化したいIDがあれば指定 (空リスト [] なら、尤度が高いTop10を自動選択)
TARGET_IDS_MANUAL = [] 
# TARGET_IDS_MANUAL = [10, 25, 33] # 指定例

# 自動選択時の件数
AUTO_PLOT_TOP_K = 20

# 出力先
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"/home/mizutani/projects/RF/runs/20260123_210513/eval"
os.makedirs(OUT_DIR, exist_ok=True)

# データパス (テスト用データ)
NPZ_PATH = '/home/mizutani/projects/RF/data/input_real_test_m4_emb.npz'
ADJ_PATH = '/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt'

# ★★★ モデルパス設定 ★★★
# 提案手法 (Proposed)
MODEL_PATH_PROPOSED = '/home/mizutani/projects/RF/runs/20260123_210513/model_weights_20260123_210513.pth'

# 比較対象 (Ablation) 
MODEL_PATH_ABLATION = '/home/mizutani/projects/RF/runs/20260124_003007/model_weights_20260124_003007.pth' 

# デバイス設定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
#  Helper Functions & Classes
# =========================================================

class SchurAnalyzer:
    """可視化用にSchur分解を行うクラス"""
    def __init__(self, A_matrix):
        self.A = A_matrix
        self.T, self.Z = scipy.linalg.schur(self.A, output='real')
        self.blocks = []
        i = 0
        n = self.A.shape[0]
        while i < n:
            is_2x2 = False
            if i < n - 1 and abs(self.T[i+1, i]) > 1e-6:
                is_2x2 = True
            
            if is_2x2:
                block = self.T[i:i+2, i:i+2]
                eigvals = scipy.linalg.eigvals(block)
                decay = np.mean(np.abs(eigvals))
                freq = np.mean(np.abs(np.angle(eigvals)))
                self.blocks.append({"idx": i, "size": 2, "type": "Osc", "decay": decay, "freq": freq, "desc": f"Osc (|λ|={decay:.2f})"})
                i += 2
            else:
                val = self.T[i, i]
                self.blocks.append({"idx": i, "size": 1, "type": "Decay", "decay": abs(val), "freq": 0.0, "desc": f"Real (|λ|={abs(val):.2f})"})
                i += 1
        self.blocks.sort(key=lambda x: x["decay"], reverse=True)
        
    def transform(self, x):
        return x @ self.Z
        
    def get_block_norms(self, x_schur):
        if x_schur.ndim == 1: x_schur = x_schur[np.newaxis, :]
        norms = []
        for b in self.blocks:
            idx = b["idx"]
            size = b["size"]
            vec_part = x_schur[:, idx : idx+size]
            norm = np.linalg.norm(vec_part, axis=1)
            norms.append(norm)
        return np.stack(norms, axis=1)

def compute_shortest_path_distance_matrix(adj):
    n = adj.shape[0]
    return torch.zeros((n, n), dtype=torch.long)

def load_model_safe(path, network, base_N, dist_mat):
    print(f"Loading Model from {path}...")
    ckpt = torch.load(path, map_location=DEVICE)
    c = ckpt['config']
    state_dict = ckpt['model_state_dict']

    if 'agent_embedding.weight' in state_dict:
        det_num_agents = state_dict['agent_embedding.weight'].shape[0]
    else:
        det_num_agents = c.get('num_agents', 1)

    if 'stay_embedding.weight' in state_dict:
        det_max_stay = state_dict['stay_embedding.weight'].shape[0] - 1
    else:
        det_max_stay = c.get('max_stay_count', 500)

    h_dim = c.get("holiday_emb_dim", 4)
    tz_dim = c.get("time_zone_emb_dim", 4)
    e_dim = c.get("event_emb_dim", 4)

    model = KoopmanRoutesFormer(
        vocab_size=c['vocab_size'],
        token_emb_dim=c.get('token_emb_dim', c.get('d_ie', 64)),
        d_model=c.get('d_model', c.get('d_ie', 64)),
        nhead=c.get('nhead', 4),
        num_layers=c.get('num_layers', 3),
        d_ff=c.get('d_ff', 128),
        z_dim=c['z_dim'],
        pad_token_id=network.N,
        dist_mat_base=dist_mat,
        base_N=base_N,
        holiday_emb_dim=h_dim, time_zone_emb_dim=tz_dim, event_emb_dim=e_dim,
        num_agents=det_num_agents, agent_emb_dim=c.get('agent_emb_dim', 16),
        max_stay_count=det_max_stay, stay_emb_dim=c.get('stay_emb_dim', 16)
    )
    
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    return model, c # Configも返す

def calculate_nll_and_states(model, network, inp_tokens, stay_counts, ag_b, h_b, tz_b, e_b, tgt_tokens):
    """
    NLL計算に加え、可視化用の状態(z, u, probs)も返す
    """
    B_size, T = inp_tokens.shape
    
    def align_ctx(ctx, target_len):
        out = torch.zeros((B_size, target_len), dtype=torch.long, device=DEVICE)
        copy_len = min(ctx.shape[1], target_len - 1)
        if copy_len > 0:
            out[:, 1 : 1+copy_len] = ctx[:, :copy_len]
        return out

    h_in  = align_ctx(h_b, T)
    tz_in = align_ctx(tz_b, T)
    e_in  = align_ctx(e_b, T)

    with torch.no_grad():
        logits, z_hat, _, u_all = model(
            tokens=inp_tokens, stay_counts=stay_counts, agent_ids=ag_b,
            holidays=h_in, time_zones=tz_in, events=e_in
        )
        # NLL
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=network.N)
        loss_per_step = loss_fn(logits.view(-1, logits.size(-1)), tgt_tokens.view(-1))
        
        valid_mask = (tgt_tokens.view(-1) != network.N)
        if valid_mask.sum() > 0:
            avg_nll = loss_per_step[valid_mask].mean().item()
        else:
            avg_nll = 99.9

    return avg_nll, logits, z_hat, u_all

# =========================================================
#  Visualization Logic
# =========================================================

def plot_sample(model, network, sample_data, schur, sample_idx, nll_val, out_dir):
    """Proposed Modelの結果を可視化する"""
    
    # データの準備 (再計算になってしまうが、コード構造上シンプルにするため再度Forward)
    route = sample_data['route']
    valid_len = np.sum(route != 38)
    
    rt_b = torch.tensor(route[:valid_len], dtype=torch.long).unsqueeze(0).to(DEVICE)
    ag_b = torch.tensor(sample_data['agent'], dtype=torch.long).unsqueeze(0).to(DEVICE)
    h_b  = torch.tensor(sample_data['holiday'][:valid_len], dtype=torch.long).unsqueeze(0).to(DEVICE)
    tz_b = torch.tensor(sample_data['timezone'][:valid_len], dtype=torch.long).unsqueeze(0).to(DEVICE)
    e_b  = torch.tensor(sample_data['event'][:valid_len], dtype=torch.long).unsqueeze(0).to(DEVICE)
    
    tokenizer = Tokenization(network)
    inp_tokens = tokenizer.tokenization(rt_b, mode="simple").long().to(DEVICE)
    stay_counts = tokenizer.calculate_stay_counts(inp_tokens)
    tgt_tokens = tokenizer.tokenization(rt_b, mode="next").long().to(DEVICE)

    # Forward
    _, logits, z_hat, u_all = calculate_nll_and_states(
        model, network, inp_tokens, stay_counts, ag_b, h_b, tz_b, e_b, tgt_tokens
    )

    # Numpy変換
    z_seq = z_hat[0].cpu().numpy()
    u_seq = u_all[0].cpu().numpy()
    probs = F.softmax(logits[0], dim=-1).cpu().numpy()
    tokens_np = inp_tokens[0].cpu().numpy()
    tgt_np = tgt_tokens[0].cpu().numpy()

    # Schur Analysis
    z_schur = schur.transform(z_seq)
    z_norms = schur.get_block_norms(z_schur)
    
    B_np = model.B.detach().cpu().numpy()
    Bu = u_seq @ B_np.T 
    Bu_schur = schur.transform(Bu)
    Bu_norms = schur.get_block_norms(Bu_schur)

    # Plot
    def get_token_label(t):
        if t < network.N: return str(t)
        if t == network.N: return "<p>"
        if t == tokenizer.SPECIAL_TOKENS["<b>"]: return "<b>"
        if t == tokenizer.SPECIAL_TOKENS["<e>"]: return "<e>"
        return str(t)

    token_labels = [get_token_label(t) for t in tokens_np]
    yticklabels = [b["desc"] for b in schur.blocks]
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 18), sharex=True)
    
    # 1. Z Energy
    sns.heatmap(z_norms.T, ax=axes[0], cmap="viridis", cbar_kws={'label': 'Energy'})
    axes[0].set_title(f"Sample {sample_idx} (NLL={nll_val:.4f}): Latent State $z$ Evolution")
    axes[0].set_ylabel("Eigen Mode")
    axes[0].set_yticks(np.arange(len(yticklabels))+0.5)
    axes[0].set_yticklabels(yticklabels, rotation=0, fontsize=9)

    # 2. Bu Force
    sns.heatmap(Bu_norms.T, ax=axes[1], cmap="magma", cbar_kws={'label': 'Force'})
    axes[1].set_title(f"Control Input $Bu$ Evolution")
    axes[1].set_ylabel("Eigen Mode")
    axes[1].set_yticks(np.arange(len(yticklabels))+0.5)
    axes[1].set_yticklabels(yticklabels, rotation=0, fontsize=9)

    # 3. Probability
    vocab_to_plot = network.N + 5
    probs_plot = probs[:, :vocab_to_plot].T
    sns.heatmap(probs_plot, ax=axes[2], cmap="Blues", cbar_kws={'label': 'Prob'})
    axes[2].set_title(f"Prediction Probability (Lower NLL is better)")
    axes[2].set_ylabel("Token ID")
    
    seq_len_plot = probs_plot.shape[1]
    x_coords = np.arange(seq_len_plot) + 0.5
    y_coords = tgt_np[:seq_len_plot] + 0.5
    valid_mask = y_coords < vocab_to_plot
    axes[2].scatter(x_coords[valid_mask], y_coords[valid_mask], 
                    color='red', marker='x', s=50, label='Ground Truth')
    axes[2].legend(loc='upper right')
    axes[2].set_xticks(np.arange(seq_len_plot) + 0.5)
    axes[2].set_xticklabels(token_labels, rotation=0, fontsize=10)
    axes[2].set_xlabel("Input Token Sequence")

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"eval_sample_{sample_idx}.png")
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"Saved plot: {save_path}")

# =========================================================
#  Main Process
# =========================================================

def main():
    print("Loading Resources...")
    # 1. Network
    adj_matrix = torch.load(ADJ_PATH, weights_only=True)
    if adj_matrix.shape[0] == 38:
        base_N = 19
        base_adj = adj_matrix[:base_N, :base_N]
    else:
        base_adj = adj_matrix
        base_N = int(base_adj.shape[0])

    expanded_adj = expand_adjacency_matrix(adj_matrix)
    dummy_feat = torch.zeros((len(adj_matrix), 1))
    network = Network(expanded_adj, torch.cat([dummy_feat, dummy_feat], dim=0))
    dist_mat = compute_shortest_path_distance_matrix(base_adj)

    # 2. Models
    print("--- Loading Models ---")
    model_prop, c_prop = load_model_safe(MODEL_PATH_PROPOSED, network, base_N, dist_mat)
    model_abl, _ = load_model_safe(MODEL_PATH_ABLATION, network, base_N, dist_mat)

    # 3. Data
    print(f"Loading Data from {NPZ_PATH}...")
    data = np.load(NPZ_PATH)
    routes = data['route_arr']
    agent_ids = data['agent_ids'] if 'agent_ids' in data else np.zeros(len(routes))
    holidays = data['holiday_arr']
    timezones = data['time_zone_arr']
    events = data['event_arr']

    # 4. Full Evaluation Loop
    results = []
    total_samples = len(routes)
    print(f"\nCalculating NLL for ALL {total_samples} samples...")
    
    tokenizer = Tokenization(network)

    for idx in range(total_samples):
        if idx % 50 == 0: print(f"Processing {idx}/{total_samples}...")

        raw_route = routes[idx]
        valid_len = np.sum(raw_route != 38)
        if valid_len < 2: continue

        # Batch
        rt_b = torch.tensor(raw_route[:valid_len], dtype=torch.long).unsqueeze(0).to(DEVICE)
        ag_b = torch.tensor(agent_ids[idx], dtype=torch.long).unsqueeze(0).to(DEVICE)
        h_b  = torch.tensor(holidays[idx][:valid_len], dtype=torch.long).unsqueeze(0).to(DEVICE)
        tz_b = torch.tensor(timezones[idx][:valid_len], dtype=torch.long).unsqueeze(0).to(DEVICE)
        e_b  = torch.tensor(events[idx][:valid_len], dtype=torch.long).unsqueeze(0).to(DEVICE)

        # Tokenize
        inp_tokens = tokenizer.tokenization(rt_b, mode="simple").long().to(DEVICE)
        tgt_tokens = tokenizer.tokenization(rt_b, mode="next").long().to(DEVICE)
        stay_counts = tokenizer.calculate_stay_counts(inp_tokens)

        # NLL Only
        nll_prop, _, _, _ = calculate_nll_and_states(
            model_prop, network, inp_tokens, stay_counts, ag_b, h_b, tz_b, e_b, tgt_tokens
        )
        nll_abl, _, _, _ = calculate_nll_and_states(
            model_abl, network, inp_tokens, stay_counts, ag_b, h_b, tz_b, e_b, tgt_tokens
        )
        
        # Meta info
        path_list = raw_route[:valid_len].tolist()
        row = {
            "id": idx,
            "nll_proposed": round(nll_prop, 4),
            "nll_ablation": round(nll_abl, 4),
            "nll_diff": round(nll_prop - nll_abl, 4),
            "length": valid_len,
            "start_node": path_list[0],
            "end_node": path_list[-1],
            "cond_holiday": "Yes" if holidays[idx][0]==1 else "No",
            "cond_timezone": "Night" if timezones[idx][0]==1 else "Day",
            "cond_event": "Active" if np.any(events[idx][:valid_len]==1) else "None",
            "full_route": ",".join(map(str, path_list))
        }
        results.append(row)

    # 5. Save Results
    df = pd.DataFrame(results)
    csv_path = os.path.join(OUT_DIR, "full_comparison_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nFull CSV saved: {csv_path}")

    # 6. Select Visualization Targets
    if TARGET_IDS_MANUAL:
        print(f"Using Manual Target IDs: {TARGET_IDS_MANUAL}")
        plot_ids = TARGET_IDS_MANUAL
    else:
        # Sort by Proposed NLL (ascending = better likelihood)
        df_sorted = df.sort_values("nll_proposed", ascending=True)
        top_k = df_sorted.head(AUTO_PLOT_TOP_K)
        plot_ids = top_k['id'].tolist()
        print(f"Auto-selected Top {AUTO_PLOT_TOP_K} samples (Lowest NLL): {plot_ids}")

    # 7. Visualization Loop
    print("\nGenerating Plots for selected samples...")
    
    # Schur Init (Only for Proposed Model Visualization)
    A_np = model_prop.A.detach().cpu().numpy()
    schur = SchurAnalyzer(A_np)
    
    for idx in plot_ids:
        # Get data from Original NPZ arrays using idx
        sample_data = {
            'route': routes[idx],
            'agent': agent_ids[idx],
            'holiday': holidays[idx],
            'timezone': timezones[idx],
            'event': events[idx]
        }
        # Get NLL from dataframe
        row = df[df['id'] == idx]
        if len(row) > 0:
            val = row.iloc[0]['nll_proposed']
        else:
            val = 0.0
            
        plot_sample(model_prop, network, sample_data, schur, idx, val, OUT_DIR)

    # 8. Report
    txt_path = os.path.join(OUT_DIR, "summary_report.txt")
    with open(txt_path, "w") as f:
        f.write("=== Full Comparison Report ===\n")
        f.write(f"Proposed: {MODEL_PATH_PROPOSED}\n")
        f.write(f"Ablation: {MODEL_PATH_ABLATION}\n")
        f.write(f"Total Samples: {len(df)}\n")
        f.write(f"Avg NLL (Prop): {df['nll_proposed'].mean():.4f}\n")
        f.write(f"Avg NLL (Abl): {df['nll_ablation'].mean():.4f}\n")
        wins = (df['nll_diff'] < 0).sum()
        f.write(f"Proposed Wins: {wins}/{len(df)} ({wins/len(df):.1%})\n")
        f.write(f"\nPlotted IDs: {plot_ids}\n")

    print(f"All Done. Output Directory: {OUT_DIR}")

if __name__ == "__main__":
    main()