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
import itertools

# ユーザー定義モジュール
from network import Network, expand_adjacency_matrix
from tokenization import Tokenization
from KP_RF import KoopmanRoutesFormer

# =========================================================
#  Config & Settings (ここを書き換えて操作します)
# =========================================================
# 動作モード: "SCAN" (CSV出力) or "PLOT" (指定IDの可視化)
MODE = "PLOT" 

# PLOTモードの時に可視化したいサンプルIDのリスト
TARGET_IDS = [1013, 83, 2576, 2564, 1910, 435, 3752, 4870, 1774, 128, 3215, 2, 35, 142, 156, 1716, 4614, 224, 4, 220, 995, 540, 2380, 2395, 140, 359, 166, 1581, 33, 126]  

# output
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"/home/mizutani/projects/RF/runs/detailed_analysis_{run_id}"
os.makedirs(OUT_DIR, exist_ok=True)

# データパス
NPZ_PATH = '/home/mizutani/projects/RF/data/input_real_m4.npz'
# 学習済みモデルのパス
model_path = '/home/mizutani/projects/RF/runs/20260121_145835/model_weights_20260121_145835.pth'

# =========================================================
#  Common Definitions
# =========================================================
ADJ_MAP_CODE = """
ADJACENCY_MAP = {
    0:[1,2,4,11],
    1:[0,2,4,5,9],
    2:[0,1,5,6,7],
    4:[0,1,5,8,9,10,11],
    5:[1,2,4,6,10],
    6:[2,5,7,10,14],
    7:[2,6,13,14,15],
    8:[4,9,11],
    9:[1,4,8,10,12],
    10:[4,5,6,9,12,13],
    11:[0,4,8],
    12:[9,10,13],
    13:[7,10,12,14,15],
    14:[6,7,13,15,16],
    15:[7,13,14],
    16:[14,17,18],
    17:[16,18],
    18:[16,17]
}
"""
N_BASE = 19
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
#  Analysis Classes & Functions
# =========================================================

class SchurAnalyzer:
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
                self.blocks.append({
                    "idx": i, "size": 2, "type": "Osc",
                    "decay": decay, "freq": freq,
                    "desc": f"Osc (|λ|={decay:.2f})"
                })
                i += 2
            else:
                val = self.T[i, i]
                self.blocks.append({
                    "idx": i, "size": 1, "type": "Decay",
                    "decay": abs(val), "freq": 0.0,
                    "desc": f"Real (|λ|={abs(val):.2f})"
                })
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

def get_latest_model_path(base_dir):
    pth_files = glob.glob(os.path.join(base_dir, "*", "*.pth"))
    if not pth_files: return None
    return max(pth_files, key=os.path.getctime)

def load_adjacency_from_map():
    adj_map = {}
    exec(ADJ_MAP_CODE, globals(), adj_map)
    mapping = adj_map['ADJACENCY_MAP']
    adj = torch.zeros((N_BASE, N_BASE), dtype=torch.float32)
    for src, neighbors in mapping.items():
        for dst in neighbors:
            adj[src, dst] = 1.0
            adj[dst, src] = 1.0
    return adj

def compute_hop_dist(adj):
    n = adj.shape[0]
    dist = torch.full((n, n), 999, dtype=torch.long)
    for i in range(n):
        dist[i, i] = 0
        queue = [i]
        while queue:
            curr = queue.pop(0)
            d = dist[i, curr]
            neighbors = torch.where(adj[curr])[0]
            for nxt in neighbors:
                if dist[i, nxt] > d + 1:
                    dist[i, nxt] = d + 1
                    queue.append(nxt)
    return dist


def analyze_gate_drivers(model, z_seq, gate_vals, schur, sample_idx, out_dir):
    """
    ゲート g(z) がどのSchurモードによって駆動されているかを解析する
    """
    # 1. 重みとバイアスの取得
    # Linear(z_dim, 1) -> weight shape is [1, z_dim] -> flatten to [z_dim]
    W_gate = model.delta_gate.weight.detach().cpu().numpy().flatten()
    b_gate = model.delta_gate.bias.detach().cpu().numpy().flatten()
    
    # 2. 重みをSchur基底へ射影
    # logit = z . W^T + b
    # z = z_schur . Z^T
    # logit = (z_schur . Z^T) . W^T + b = z_schur . (W . Z)^T + b
    # したがって、Schur空間での重み W_schur = W @ Z
    W_schur = W_gate @ schur.Z
    
    # 3. 寄与度の計算 (Contribution)
    # z_schur: [T, z_dim]
    z_schur = schur.transform(z_seq)
    
    # 要素ごとの積 (Broadcasting: [T, dim] * [dim])
    contributions = z_schur * W_schur
    
    # ブロックごとにまとめる (エネルギーではなく、符号付きの「和」をとる)
    # これにより「このモードはプラスに効いた/マイナスに効いた」が分かる
    T_len = z_seq.shape[0]
    num_blocks = len(schur.blocks)
    block_contribs = np.zeros((T_len, num_blocks))
    
    for i, b in enumerate(schur.blocks):
        idx = b["idx"]
        size = b["size"]
        # 部分ベクトルの寄与の合計
        # 例: 振動モードなら実部と虚部に対応する2次元の寄与を足し合わせる
        part = contributions[:, idx : idx+size]
        block_contribs[:, i] = part.sum(axis=1)

    # 4. 可視化 (Stacked Bar Chart or Heatmap)
    # ここでは「時系列に沿って、どのモードがゲートを押し上げたか」を見るため、
    # 貢献度のヒートマップを描きます
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, height_ratios=[1, 2])
    
    # 上段: 実際のゲート値とLogit
    ax1.plot(gate_vals, color="black", linewidth=2, label="Gate Probability (Sigmoid)")
    ax1.set_ylabel("Gate Probability")
    ax1.set_ylim(-0.1, 1.1)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Sample {sample_idx}: What drives the Gate g(z)?")
    
    # Logit (Sigmoidの中身) も参考表示
    # Logit = sum(contribs) + bias
    logit_seq = block_contribs.sum(axis=1) + b_gate
    ax1_twin = ax1.twinx()
    ax1_twin.plot(logit_seq, color="gray", linestyle="--", alpha=0.5, label="Raw Logit")
    ax1_twin.set_ylabel("Logit (Input to Sigmoid)")
    # Logit=0 が Gate=0.5 の境界線
    ax1_twin.axhline(0, color="red", linestyle=":", linewidth=0.5)
    
    # 下段: モードごとの寄与度 (ヒートマップ)
    # 赤 = ゲートを開けようとする力 (Positive)
    # 青 = ゲートを閉じようとする力 (Negative)
    sns.heatmap(block_contribs.T, ax=ax2, cmap="coolwarm", center=0, cbar_kws={'label': 'Contribution to Logit'})
    
    yticklabels = [b["desc"] for b in schur.blocks]
    ax2.set_yticks(np.arange(len(yticklabels))+0.5)
    ax2.set_yticklabels(yticklabels, rotation=0, fontsize=9)
    ax2.set_ylabel("Schur Mode")
    ax2.set_xlabel("Time Step")
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"gate_driver_sample_{sample_idx}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved Gate Driver analysis: {save_path}")


# =========================================================
#  Logic: Scan & CSV Export
# =========================================================

def scan_and_export_csv(model, network, routes, agent_ids, times, config):
    print("Scanning all routes to generate candidate list...")
    results = []
    
    limit = min(len(routes), 5000) 
    tokenizer = Tokenization(network)
    
    # ★★★ 設定: 広場とみなすノードID ★★★
    TARGET_NODE_ID = 21 
    
    for i in range(limit):
        if i % 500 == 0: print(f"Processing {i}/{limit}...")
        
        raw_r = routes[i]
        valid_len = np.sum(raw_r != 38)
        if valid_len < 5: continue 
        
        path = raw_r[:valid_len]
        path_list = path.tolist()
        
        # 1. Basic Stats
        start_node = path[0]
        end_node = path[-1]
        unique_nodes = len(np.unique(path))
        has_plaza = (TARGET_NODE_ID in path)
        
        # 2. Plaza Stats & Segmentation
        # itertools.groupby で [非2, 2, 2, 非2, 非2...] を [非2の塊, 2の塊, 非2の塊...] にまとめる
        groups = [(k, list(g)) for k, g in itertools.groupby(path_list, lambda x: x == TARGET_NODE_ID)]
        
        max_stay = 0
        segments_data = {}
        
        if has_plaza:
            # 広場(TARGET)であるグループのインデックスを探す
            stay_indices = [idx for idx, (is_target, _) in enumerate(groups) if is_target]
            
            # 各滞在について Pre / Stay / Post を抽出
            for count, grp_idx in enumerate(stay_indices):
                stay_num = count + 1 # 1, 2, 3...
                
                # Stay部分
                stay_nodes = groups[grp_idx][1]
                if len(stay_nodes) > max_stay: max_stay = len(stay_nodes)
                segments_data[f"stay_{stay_num}"] = ", ".join(map(str, stay_nodes))
                
                # Pre部分 (一つ前のグループが非ターゲットなら取得、なければ空)
                if grp_idx > 0:
                    pre_nodes = groups[grp_idx-1][1]
                else:
                    pre_nodes = [] # いきなり広場から始まった場合
                segments_data[f"pre_{stay_num}"] = ", ".join(map(str, pre_nodes))
                
                # Post部分 (一つ後のグループが非ターゲットなら取得、なければ空)
                if grp_idx < len(groups) - 1:
                    post_nodes = groups[grp_idx+1][1]
                else:
                    post_nodes = [] # 広場で終わった場合
                segments_data[f"post_{stay_num}"] = ", ".join(map(str, post_nodes))

        # 3. Model Likelihood
        rt = torch.tensor(raw_r, dtype=torch.long).unsqueeze(0).to(DEVICE)
        tm = torch.tensor(times[i], dtype=torch.long).unsqueeze(0).to(DEVICE)
        ag = torch.tensor(agent_ids[i], dtype=torch.long).unsqueeze(0).to(DEVICE)
        
        inp = tokenizer.tokenization(rt, mode="simple").long().to(DEVICE)
        tgt = tokenizer.tokenization(rt, mode="next").long().to(DEVICE)
        stay = tokenizer.calculate_stay_counts(inp)
        
        with torch.no_grad():
            logits, _, _, _ = model(inp, stay, ag, time_tensor=tm)
            loss = F.cross_entropy(logits.view(-1, config['vocab_size']), tgt.view(-1), reduction='none')
            mask = (tgt.view(-1) != network.N) & (tgt.view(-1) != 38)
            if mask.sum() > 0:
                nll = loss[mask].mean().item()
            else:
                nll = 99.9
        
        # データの構築
        row = {
            "id": i,
            "len": valid_len,
            "start": start_node,
            "end": end_node,
            "unique": unique_nodes,
            "has_plaza": has_plaza,
            "plaza_stay_len": max_stay,
            "nll": round(nll, 4),
            "full_trajectory": ", ".join(map(str, path_list)) # 全体も残す
        }
        # 動的に作ったセグメント列を結合
        row.update(segments_data)
        
        results.append(row)
        
    df = pd.DataFrame(results)
    
    # 列の並び順を整える（可変の stay_X が後ろに来るように）
    # 固定列を先に定義
    fixed_cols = ["id", "nll", "len", "start", "end", "unique", "has_plaza", "plaza_stay_len", "full_trajectory"]
    # DataFrameにある列のうち、固定列以外（＝動的列）を抽出してソート
    dynamic_cols = [c for c in df.columns if c not in fixed_cols]
    
    # pre_1, stay_1, post_1, pre_2... の順に並ぶようにソート
    # そのままだと辞書順になるので、stay番号でソートする関数
    def sort_key(col_name):
        # col_name is like "pre_1", "stay_12"
        parts = col_name.split('_')
        if len(parts) < 2: return (999, col_name)
        type_order = {"pre": 0, "stay": 1, "post": 2} # 表示順序
        return (int(parts[1]), type_order.get(parts[0], 9))

    dynamic_cols.sort(key=sort_key)
    
    # 再配置
    final_cols = fixed_cols + dynamic_cols
    # 存在しない列が含まれるとエラーになるので、dfにあるものだけで構成
    final_cols = [c for c in final_cols if c in df.columns]
    
    df = df[final_cols]
    
    csv_path = os.path.join(OUT_DIR, "candidate_routes_segmented.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved segmented candidate list to: {csv_path}")
    
# =========================================================
#  Logic: Visualization
# =========================================================

def analyze_and_plot(model, network, route, agent_id, time_arr, schur, sample_idx):
    # Prepare
    tokenizer = Tokenization(network)
    rt = torch.tensor(route, dtype=torch.long).unsqueeze(0).to(DEVICE)
    # pad除去して長さを確定
    valid_len = np.sum(route != 38)
    rt = rt[:, :valid_len]
    
    tm = torch.tensor(time_arr, dtype=torch.long).unsqueeze(0).to(DEVICE)
    ag = torch.tensor(agent_id, dtype=torch.long).unsqueeze(0).to(DEVICE)
    
    inp = tokenizer.tokenization(rt, mode="simple").long().to(DEVICE)
    tgt = tokenizer.tokenization(rt, mode="next").long().to(DEVICE)
    stay = tokenizer.calculate_stay_counts(inp)
    
    # Forward
    with torch.no_grad():
        logits, z_hat, z_pred, u_all, debug = model(
            inp, stay, ag, time_tensor=tm, return_debug=True
        )
        # To Numpy
        z_seq = z_hat[0].cpu().numpy()
        u_seq = u_all[0].cpu().numpy()
        probs = F.softmax(logits[0], dim=-1).cpu().numpy()
        
        gate_vals = debug["delta_gate"][0].cpu().numpy().flatten()
        bias_vals = debug["delta_bias"][0].cpu().numpy()

    # BOS補正: route(T)に対してmodel出力は(T+1)になることがあるので合わせる
    # inputがBOS始まりなので、z_seq[0]はBOS直後の状態。route[0]はBOS。
    # 可視化したいのは「実際の移動(Token 1〜)」とそれに対応する状態。
    # ここでは単純に長さをminで合わせます
    L = min(len(z_seq), valid_len)
    z_seq = z_seq[:L]
    u_seq = u_seq[:L]
    probs = probs[:L]
    gate_vals = gate_vals[:L]
    bias_vals = bias_vals[:L]
    tokens = route[:L] # Raw tokens
    
    # Schur Analysis
    z_schur = schur.transform(z_seq)
    z_norms = schur.get_block_norms(z_schur)
    
    # Input Analysis
    B_np = model.B.detach().cpu().numpy()
    
    # Total Force
    Bu_total = u_seq @ B_np.T
    Bu_total_schur = schur.transform(Bu_total)
    Bu_total_norms = schur.get_block_norms(Bu_total_schur)
    
    # Plaza Force (Fix for Red Bubbles)
    plaza_dim = 4
    u_plaza_only = np.zeros_like(u_seq)
    u_plaza_only[:, -plaza_dim:] = u_seq[:, -plaza_dim:]
    
    Bu_plaza = u_plaza_only @ B_np.T
    Bu_plaza_schur = schur.transform(Bu_plaza)
    Bu_plaza_norms = schur.get_block_norms(Bu_plaza_schur)
    
    # Plotting
    fig = plt.figure(figsize=(16, 22))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1.2, 2, 2], hspace=0.35)
    
    # 1. Timeline
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(gate_vals, label="Gate g(z)", color="purple", ls="--", alpha=0.8)
    # Tokens
    for t in range(L):
        node = tokens[t]
        color = "red" if node == 2 else "black"
        # 広場滞在中(2->2)は背景を赤くするなどの工夫も可
        ax1.text(t, 0.5, str(node), color=color, ha='center', fontweight='bold',
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        # Max Bias
        mb = bias_vals[t].max()
        if mb > 0.01:
            ax1.bar(t, mb, color='orange', alpha=0.3, width=1.0)
            
    ax1.set_xlim(-0.5, L-0.5)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_title(f"Sample {sample_idx}: Timeline")
    ax1.legend(loc="upper right")
    
    # 2. Probability
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    sns.heatmap(probs.T, ax=ax2, cmap="Blues", cbar_kws={'label': 'Prob'})
    ax2.scatter(np.arange(L)+0.5, tokens+0.5, marker="x", color="red", s=40)
    ax2.set_ylabel("Node ID")
    
    # 3. Z Dynamics (Schur Energy)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    sns.heatmap(z_norms.T, ax=ax3, cmap="viridis", cbar_kws={'label': 'Energy'})
    yticklabels = [b["desc"] for b in schur.blocks]
    ax3.set_yticks(np.arange(len(yticklabels))+0.5)
    ax3.set_yticklabels(yticklabels, rotation=0, fontsize=9)
    ax3.set_title("Latent Energy (Real Schur Modes)")
    
    # 4. Input Forcing (Corrected)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    sns.heatmap(Bu_total_norms.T, ax=ax4, cmap="Greys", cbar_kws={'label': 'Total Force'}, alpha=0.3, vmin=0)
    
    # ★★★ Red Bubble Correction ★★★
    # Only show bubbles where the agent is actually AT the plaza (Node 2)
    # "広場があるノードだけでいいはず" -> マスクを作成
    is_at_plaza = (tokens == 2) | (tokens == 21)   # [T] (Boolean mask)
    
    num_blocks = len(schur.blocks)
    t_grid, b_grid = np.meshgrid(np.arange(L), np.arange(num_blocks))
    plaza_vals = Bu_plaza_norms.T 
    
    # マスク適用: 広場にいない場所の値は強制的にNaNまたは0にして描画させない
    # plaza_vals shape: [NumBlocks, T]
    # mask shape needs broadcast
    mask_2d = np.tile(is_at_plaza, (num_blocks, 1)) # [NumBlocks, T]
    
    # 広場にいて、かつForceがある程度大きいものだけプロット
    plot_mask = mask_2d & (plaza_vals > 0.01)
    
    if plot_mask.any():
        sc = ax4.scatter(
            t_grid[plot_mask] + 0.5,
            b_grid[plot_mask] + 0.5,
            s=plaza_vals[plot_mask] * 300, 
            c=plaza_vals[plot_mask],
            cmap="Reds",
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
            label="Plaza Force (Active Only)"
        )
        plt.colorbar(sc, ax=ax4, label="Plaza Force")
    
    ax4.set_yticks(np.arange(len(yticklabels))+0.5)
    ax4.set_yticklabels(yticklabels, rotation=0, fontsize=9)
    ax4.set_title("Input Forcing (Red Bubble = Active Plaza Force)")
    ax4.set_xlabel("Time Step")
    
    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, f"result_sample_{sample_idx}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")

    analyze_gate_drivers(model, z_seq, gate_vals, schur, sample_idx, OUT_DIR)

# =========================================================
#  Main Entry
# =========================================================

def main():
    # 1. Load Resources
    base_adj = load_adjacency_from_map()
    expanded_adj = expand_adjacency_matrix(base_adj)
    # Fix: Correct dummy size
    dummy = torch.zeros((len(expanded_adj), 1))
    network = Network(expanded_adj, dummy)
    
    ckpt = torch.load(model_path, map_location=DEVICE)
    c = ckpt['config']
    
    # Adjacency for Model
    dist_mat = compute_hop_dist(base_adj)
    
    model = KoopmanRoutesFormer(
        vocab_size=c['vocab_size'], token_emb_dim=c['token_emb_dim'],
        d_model=c['d_model'], nhead=c['nhead'], num_layers=c['num_layers'],
        d_ff=c['d_ff'], z_dim=c['z_dim'], pad_token_id=c['pad_token_id'],
        dist_mat_base=dist_mat, base_N=N_BASE,
        num_agents=c.get('num_agents', 1), agent_emb_dim=c.get('agent_emb_dim', 16),
        max_stay_count=500, stay_emb_dim=c.get('stay_emb_dim', 16)
    )
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.to(DEVICE)
    model.eval()
    
    # 2. Load Data
    print(f"Loading Data: {NPZ_PATH}")
    data = np.load(NPZ_PATH)
    routes = data['route_arr']
    agent_ids = data['agent_ids'] if 'agent_ids' in data else np.zeros(len(routes))
    times = data['time_arr'] if 'time_arr' in data else np.full(len(routes), 202511221200)
    
    # 3. Execute Mode
    if MODE == "SCAN":
        scan_and_export_csv(model, network, routes, agent_ids, times, c)
    
    elif MODE == "PLOT":
        schur = SchurAnalyzer(model.A.detach().cpu().numpy())
        print("\n=== Schur Modes ===")
        for b in schur.blocks: print(f"  {b['desc']}")
        
        print(f"\nProcessing Target IDs: {TARGET_IDS}")
        for idx in TARGET_IDS:
            if idx >= len(routes):
                print(f"ID {idx} is out of range.")
                continue
            print(f"Plotting Sample {idx}...")
            analyze_and_plot(model, network, routes[idx], agent_ids[idx], times[idx], schur, idx)

if __name__ == "__main__":
    main()