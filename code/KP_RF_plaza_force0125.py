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
# 分析対象の定義
TARGETS = [
    {"name": "Node_11", "node_id": 11, "stay_token": 30},
    {"name": "Node_14", "node_id": 14, "stay_token": 33},
    {"name": "Node_02", "node_id": 2,  "stay_token": 21}
]

# 出力先
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
# パス設定はユーザー環境に合わせて適宜変更してください
OUT_DIR = f"/home/mizutani/projects/RF/runs/20260124_214854/plaza_force_{run_id}"
os.makedirs(OUT_DIR, exist_ok=True)

# データパス
NPZ_PATH = '/home/mizutani/projects/RF/data/input_real_m5.npz'
ADJ_PATH = '/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt'
MODEL_PATH = '/home/mizutani/projects/RF/runs/20260124_214854/model_weights_20260124_214854.pth'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
#  Helper Classes
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
    return model, c

# =========================================================
#  Analysis Part 0: Delta Logits (Global Effect)
# =========================================================

def analyze_delta_logits(model, network, out_dir):
    """
    広場埋め込み(Event=1)が、Event=0に対して各ノードへの遷移ロジットをどう変化させるかを可視化
    Delta Logits = W_out * B * (u_on - u_off)
    これは現在の状態や滞在ステップ数に依存しない定数ベクトルとなる。
    """
    print("Analyzing Delta Logits (Global Plaza Effect)...")
    
    # 1. 入力埋め込みの差分 (Delta u) を構築
    # u = [token, stay, agent, holiday, timezone, event]
    # 他の要素は共通なので差分をとると消える。Event部分の差分だけが残る。
    
    # Event Embeddings
    evt_off = torch.tensor([0], device=DEVICE)
    evt_on  = torch.tensor([1], device=DEVICE)
    
    emb_off = model.event_embedding(evt_off) # [1, emb_dim]
    emb_on  = model.event_embedding(evt_on)  # [1, emb_dim]
    
    delta_evt = emb_on - emb_off # [1, emb_dim]
    
    # 全入力次元における差分ベクトルを作成
    # 構造: [Token(D_tok), Stay(D_stay), Agent(D_ag), Hol(D_hol), Tz(D_tz), Event(D_evt)]
    # Eventは末尾にあると仮定（KoopmanRoutesFormerの実装順序準拠）
    
    # 各エンベディングの次元を取得
    d_tok = model.token_embedding.embedding_dim
    d_stay = model.stay_embedding.embedding_dim
    d_ag = model.agent_embedding.embedding_dim
    d_hol = model.holiday_embedding.embedding_dim
    d_tz = model.time_zone_embedding.embedding_dim
    d_evt = model.event_embedding.embedding_dim
    
    # ゼロパディングを作成
    zeros_prefix_dim = d_tok + d_stay + d_ag + d_hol + d_tz
    zeros_prefix = torch.zeros((1, zeros_prefix_dim), device=DEVICE)
    
    # Delta u: [1, total_input_dim]
    delta_u = torch.cat([zeros_prefix, delta_evt], dim=-1)
    
    # 2. Delta Logits の計算
    # Logits = W * z + b
    # z_next = A*z + B*u
    # Delta Logits = W * (B * delta_u)  (バイアス項やA*z項はキャンセルされる)
    
    B_mat = model.B # [z_dim, input_dim]
    W_out = model.to_logits.weight # [vocab_size, z_dim]
    
    # delta_z = delta_u @ B^T  => [1, z_dim]
    delta_z = F.linear(delta_u, B_mat)
    
    # delta_logits = delta_z @ W^T => [1, vocab_size]
    delta_logits = F.linear(delta_z, W_out).detach().cpu().numpy().flatten()
    
    # 3. 可視化
    vocab_size = len(delta_logits)
    indices = np.arange(vocab_size)
    
    # ラベル作成 helper
    tokenizer = Tokenization(network)
    def get_label(idx):
        if idx in tokenizer.SPECIAL_TOKENS.values():
            for k, v in tokenizer.SPECIAL_TOKENS.items():
                if v == idx: return k
        return str(idx)
        
    labels = [get_label(i) for i in indices]
    
    plt.figure(figsize=(15, 6))
    
    # 棒グラフで表示
    # 正の値：確率を上げる方向、負の値：確率を下げる方向
    bars = plt.bar(indices, delta_logits, color='skyblue', edgecolor='black')
    
    # 0ライン
    plt.axhline(0, color='gray', linewidth=0.8)
    
    plt.title("Impact of Plaza Embedding on Logits (Delta Logits)\n(Positive = Increased Transition Probability)", fontsize=14)
    plt.xlabel("Token ID (Target Node)", fontsize=12)
    plt.ylabel("Change in Logits", fontsize=12)
    plt.xticks(indices, labels, rotation=90, fontsize=8)
    plt.xlim(-1, vocab_size)
    
    # グリッド
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, "plaza_effect_delta_logits.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Delta Logits visualization to {save_path}")


# =========================================================
#  Analysis Part 1: Input Forcing Diff (Bu)
# =========================================================

def analyze_forcing_diff(model, target, schur, out_dir):
    """
    広場あり(Event=1)となし(Event=0)の入力Buの差分を、滞在ステップごとに比較
    1~7ステップに限定、スケール統一
    """
    stay_token = target["stay_token"]
    name = target["name"]
    
    print(f"Analyzing Forcing Diff for {name}...")
    
    # 滞在ステップ数 1~7
    stay_steps = np.arange(1, 8)
    
    # 共通入力
    token_t = torch.tensor([stay_token], device=DEVICE)
    agent_t = torch.tensor([0], device=DEVICE)
    holiday_t = torch.tensor([1], device=DEVICE)
    time_t = torch.tensor([0], device=DEVICE)
    
    norms_diff = []
    norms_off = []
    norms_on = []
    
    B_np = model.B.detach().cpu().numpy()

    with torch.no_grad():
        for s in stay_steps:
            stay_t = torch.tensor([s], device=DEVICE)
            
            # Event OFF
            event_off = torch.tensor([0], device=DEVICE)
            emb_token = model.token_embedding(token_t)
            emb_stay = model.stay_embedding(stay_t)
            emb_agent = model.agent_embedding(agent_t)
            emb_hol = model.holiday_embedding(holiday_t)
            emb_time = model.time_zone_embedding(time_t)
            emb_evt_off = model.event_embedding(event_off)
            
            u_off = torch.cat([emb_token, emb_stay, emb_agent, emb_hol, emb_time, emb_evt_off], dim=-1)
            u_off_np = u_off.cpu().numpy().flatten()
            
            # Event ON
            event_on = torch.tensor([1], device=DEVICE)
            emb_evt_on = model.event_embedding(event_on)
            u_on = torch.cat([emb_token, emb_stay, emb_agent, emb_hol, emb_time, emb_evt_on], dim=-1)
            u_on_np = u_on.cpu().numpy().flatten()
            
            # Calculate Bu
            Bu_off = u_off_np @ B_np.T
            Bu_on  = u_on_np  @ B_np.T
            Bu_diff = Bu_on - Bu_off
            
            # Project to Schur
            diff_schur = schur.transform(Bu_diff)
            off_schur = schur.transform(Bu_off)
            on_schur = schur.transform(Bu_on)
            
            norms_diff.append(schur.get_block_norms(diff_schur).flatten())
            norms_off.append(schur.get_block_norms(off_schur).flatten())
            norms_on.append(schur.get_block_norms(on_schur).flatten())
            
    # 配列化
    norms_diff = np.array(norms_diff).T # [NumBlocks, Steps]
    norms_off = np.array(norms_off).T
    norms_on = np.array(norms_on).T
    
    # スケール統一
    max_val_abs = max(norms_off.max(), norms_on.max())
    vmin_abs, vmax_abs = 0, max_val_abs
    
    max_val_diff = norms_diff.max()
    vmin_diff, vmax_diff = 0, max_val_diff
    
    yticklabels = [b["desc"] for b in schur.blocks]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    # 1. Event OFF
    sns.heatmap(norms_off, ax=axes[0], cmap="viridis", vmin=vmin_abs, vmax=vmax_abs, cbar_kws={'label': 'Force Energy'})
    axes[0].set_title(f"{name}: Bu Energy (Event OFF)")
    axes[0].set_xlabel("Stay Steps (Compressed 1-7)")
    axes[0].set_ylabel("Schur Mode")
    axes[0].set_yticklabels(yticklabels, rotation=0)
    
    # 2. Event ON
    sns.heatmap(norms_on, ax=axes[1], cmap="viridis", vmin=vmin_abs, vmax=vmax_abs, cbar_kws={'label': 'Force Energy'})
    axes[1].set_title(f"{name}: Bu Energy (Event ON)")
    axes[1].set_xlabel("Stay Steps (Compressed 1-7)")
    axes[1].set_yticklabels(yticklabels, rotation=0)
    
    # 3. Diff (ON - OFF)
    sns.heatmap(norms_diff, ax=axes[2], cmap="magma", vmin=vmin_diff, vmax=vmax_diff, cbar_kws={'label': 'Diff Energy'})
    axes[2].set_title(f"{name}: Diff Energy (Effect of Plaza)")
    axes[2].set_xlabel("Stay Steps (Compressed 1-7)")
    axes[2].set_yticklabels(yticklabels, rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"forcing_diff_{name}.png"))
    plt.close()


def analyze_forcing_decomposition(model, target, schur, out_dir):
    """
    Buを「Base成分（場所・滞在・時間など）」と「Event成分」に分解して可視化する。
    """
    stay_token = target["stay_token"]
    name = target["name"]
    
    print(f"Analyzing Forcing Decomposition for {name}...")
    
    # 滞在ステップ数 1~7
    stay_steps = np.arange(1, 8)
    
    # 固定条件
    token_t = torch.tensor([stay_token], device=DEVICE)
    agent_t = torch.tensor([0], device=DEVICE)
    holiday_t = torch.tensor([1], device=DEVICE)
    time_t = torch.tensor([0], device=DEVICE)
    
    norms_base = []
    norms_event = []
    
    B_np = model.B.detach().cpu().numpy()

    with torch.no_grad():
        for s in stay_steps:
            stay_t = torch.tensor([s], device=DEVICE)
            
            # --- 1. Base State (Event OFF) ---
            event_off = torch.tensor([0], device=DEVICE)
            
            emb_token = model.token_embedding(token_t)
            emb_stay = model.stay_embedding(stay_t)
            emb_agent = model.agent_embedding(agent_t)
            emb_hol = model.holiday_embedding(holiday_t)
            emb_time = model.time_zone_embedding(time_t)
            emb_evt_off = model.event_embedding(event_off)
            
            # Concatenate
            u_base = torch.cat([emb_token, emb_stay, emb_agent, emb_hol, emb_time, emb_evt_off], dim=-1)
            u_base_np = u_base.cpu().numpy().flatten()
            
            # --- 2. Event State (Event ON) ---
            event_on = torch.tensor([1], device=DEVICE)
            emb_evt_on = model.event_embedding(event_on)
            
            u_total = torch.cat([emb_token, emb_stay, emb_agent, emb_hol, emb_time, emb_evt_on], dim=-1)
            u_total_np = u_total.cpu().numpy().flatten()
            
            # --- 3. Calculate Forces ---
            # Base Force
            Bu_base = u_base_np @ B_np.T
            
            # Event Force (Diff)
            Bu_event = u_total_np @ B_np.T - Bu_base
            
            # Project to Schur
            base_schur = schur.transform(Bu_base)
            event_schur = schur.transform(Bu_event)
            
            norms_base.append(schur.get_block_norms(base_schur).flatten())
            norms_event.append(schur.get_block_norms(event_schur).flatten())
            
    # 配列化
    norms_base = np.array(norms_base).T   # [NumBlocks, Steps]
    norms_event = np.array(norms_event).T # [NumBlocks, Steps]
    
    yticklabels = [b["desc"] for b in schur.blocks]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # 1. Base Force
    sns.heatmap(norms_base, ax=axes[0], cmap="viridis", cbar_kws={'label': 'Force Energy'})
    axes[0].set_title(f"{name}: Base Force (Location/Stay/Time)")
    axes[0].set_xlabel("Stay Steps (1-7)")
    axes[0].set_ylabel("Schur Mode")
    axes[0].set_yticklabels(yticklabels, rotation=0)
    
    # 2. Event Force (Pure Effect)
    sns.heatmap(norms_event, ax=axes[1], cmap="magma", cbar_kws={'label': 'Event Pure Force'})
    axes[1].set_title(f"{name}: Event Force (Added by Plaza Embedding)")
    axes[1].set_xlabel("Stay Steps (1-7)")
    axes[1].set_yticklabels(yticklabels, rotation=0)
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"forcing_decomp_{name}.png")
    plt.savefig(save_path)
    plt.close()

# =========================================================
#  Analysis Part 2: Time-Series Contribution Analysis
# =========================================================

def run_trajectory_analysis_series(model, network, dataset_dict, target, schur, out_dir):
    """
    対象ノードを「経由」した経路全体の時系列分析 (Version 4: z_pred_next visualization)
    Step 0(Start)を除外し、Step 1以降について、
    [Current Input] -> [z_pred_next] -> [Contrib to Next] -> [Next Prob]
    の順で可視化する。
    """
    routes = dataset_dict["routes"]
    holidays = dataset_dict["holidays"]
    timezones = dataset_dict["timezones"]
    events = dataset_dict["events"]
    agent_ids = dataset_dict["agent_ids"]
    
    name = target["name"]
    target_stay_token = target["stay_token"]
    
    print(f"Scanning samples for {name} (Pass-through)...")
    
    found_samples = []
    limit = 5 
    
    for r_idx, r in enumerate(routes):
        valid_len = np.sum(r != 38)
        if valid_len < 5: continue
        path = r[:valid_len]
        
        # 始点がターゲットならスキップ (途中経由のみ)
        if path[0] == target_stay_token:
            continue 
            
        indices = np.where(path == target_stay_token)[0]
        if len(indices) == 0: continue
        
        # 広場イベントありサンプルを優先
        evt_seq = events[r_idx][:valid_len]
        is_event_active = False
        
        for idx in indices:
            if evt_seq[idx] == 1:
                is_event_active = True
                break
        
        found_samples.append({
            "id": r_idx, "path": path, "is_event": is_event_active
        })
        
        if len(found_samples) >= limit: break
    
    if len(found_samples) == 0:
        print(f"No pass-through samples found for {name}.")
        return

    # Analyze
    tokenizer = Tokenization(network)
    W_out = model.to_logits.weight.detach().cpu().numpy() # [Vocab, z_dim]
    
    special_tokens = [tokenizer.SPECIAL_TOKENS[k] for k in ["<p>","<e>","<b>","<m>"]]
    
    for sample in found_samples:
        r_idx = sample["id"]
        path = sample["path"]
        is_evt = sample["is_event"]
        
        # Prepare Batch
        rt_b = torch.tensor(path, dtype=torch.long).unsqueeze(0).to(DEVICE)
        ag_b = torch.tensor(agent_ids[r_idx], dtype=torch.long).unsqueeze(0).to(DEVICE)
        h_b  = torch.tensor(holidays[r_idx][:len(path)], dtype=torch.long).unsqueeze(0).to(DEVICE)
        tz_b = torch.tensor(timezones[r_idx][:len(path)], dtype=torch.long).unsqueeze(0).to(DEVICE)
        e_b  = torch.tensor(events[r_idx][:len(path)], dtype=torch.long).unsqueeze(0).to(DEVICE)
        
        # Tokenize (align context)
        inp_tokens = tokenizer.tokenization(rt_b, mode="simple").long().to(DEVICE)
        tgt_tokens = tokenizer.tokenization(rt_b, mode="next").long().to(DEVICE)
        stay_counts = tokenizer.calculate_stay_counts(inp_tokens)
        
        B_size, T = inp_tokens.shape
        def align_ctx(ctx, target_len):
            out = torch.zeros((B_size, target_len), dtype=torch.long, device=DEVICE)
            copy_len = min(ctx.shape[1], target_len - 1)
            if copy_len > 0: out[:, 1 : 1+copy_len] = ctx[:, :copy_len]
            return out
            
        h_in = align_ctx(h_b, T)
        tz_in = align_ctx(tz_b, T)
        e_in = align_ctx(e_b, T)
        
        # Forward
        with torch.no_grad():
            # ★変更: 返り値を正しく受け取る
            # (logits, z_hat, z_pred_next, u_all)
            logits_seq, z_hat, z_pred_next, _ = model(inp_tokens, stay_counts, ag_b, h_in, tz_in, e_in)
            
            # ★変更: 可視化対象を z_pred_next にする
            # z_pred_next[t] は input[t] に対する「次ステップ予測状態」
            z_seq = z_pred_next[0].cpu().numpy() # [T, z_dim]
            
            probs_seq = F.softmax(logits_seq[0], dim=-1).cpu().numpy() # [T, Vocab]
            
        # --- Time-Series Analysis Data Prep ---
        
        # Step 0 (Start <b>) はカットする。t=1から開始。
        plot_indices = range(1, T) # 1, 2, ..., T-1
        
        # Data Accumulators
        z_norms_list = []
        contrib_list = []
        probs_list = []
        labels_list = []
        ground_truth_list = []

        tgt_np = tgt_tokens[0].cpu().numpy()
        inp_np = inp_tokens[0].cpu().numpy()
        W_schur = W_out @ schur.Z # [Vocab, z_dim]
        
        for t in plot_indices:
            # 1. Labels (Current Input)
            curr_token = inp_np[t]
            if curr_token == 38: break
            labels_list.append(curr_token)
            
            # 2. Z Dynamics (Current Z)
            # ここでの z_seq[t] は z_pred_next[t]
            # つまり、input[t] を受けて予測された「次の状態」
            z_t = z_seq[t]
            z_t_schur = schur.transform(z_t)
            z_norms = schur.get_block_norms(z_t_schur).flatten()
            z_norms_list.append(z_norms)
            
            # 3. Contribution (to Next Target Logit)
            target_token = tgt_np[t]
            ground_truth_list.append(target_token)
            
            if target_token == 38:
                contrib_list.append(np.zeros(len(schur.blocks)))
            else:
                step_contribs = []
                for b in schur.blocks:
                    idx = b["idx"]
                    size = b["size"]
                    z_part = z_t_schur[idx : idx+size]
                    W_part = W_schur[target_token, idx : idx+size]
                    val = np.dot(z_part, W_part)
                    step_contribs.append(val)
                contrib_list.append(step_contribs)
                
            # 4. Probability (Next Prob)
            probs_list.append(probs_seq[t])

        # Convert to Numpy
        z_norms_arr = np.array(z_norms_list)
        contrib_arr = np.array(contrib_list)
        probs_arr = np.array(probs_list)
        
        if len(z_norms_arr) == 0: continue
        
        vocab_to_plot = network.N + 5
        probs_plot = probs_arr[:, :vocab_to_plot]

        # --- Plotting ---
        def get_token_label(tok):
            if tok == target_stay_token: return f"[{tok}]" 
            if tok in tokenizer.SPECIAL_TOKENS.values(): 
                for k, v in tokenizer.SPECIAL_TOKENS.items():
                    if v == tok: return k
                return "*"
            return str(tok)
            
        x_tick_labels = [get_token_label(tk) for tk in labels_list]
        yticklabels = [b["desc"] for b in schur.blocks]
        num_steps = len(labels_list)
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 18), sharex=True)
        
        # 1. Z Dynamics (Top)
        sns.heatmap(z_norms_arr.T, ax=axes[0], cmap="viridis", cbar_kws={'label': 'Z Energy'})
        axes[0].set_title(f"Sample {r_idx}: Predicted Next State (z_pred_next) Dynamics (Event={'ON' if is_evt else 'OFF'})")
        axes[0].set_ylabel("Schur Mode")
        axes[0].set_yticks(np.arange(len(yticklabels))+0.5)
        axes[0].set_yticklabels(yticklabels, rotation=0)

        # 2. Contribution to Ground Truth Logit (Middle)
        sns.heatmap(contrib_arr.T, ax=axes[1], cmap="coolwarm", center=0, cbar_kws={'label': 'Contrib to True Logit'})
        axes[1].set_title("Mode Contribution to ACTUAL Next Token Logit")
        axes[1].set_ylabel("Schur Mode")
        axes[1].set_yticks(np.arange(len(yticklabels))+0.5)
        axes[1].set_yticklabels(yticklabels, rotation=0)
        
        # 3. Prediction Probability (Bottom)
        sns.heatmap(probs_plot.T, ax=axes[2], cmap="Blues", cbar_kws={'label': 'Prob'})
        axes[2].set_title("Next Token Prediction Probability")
        axes[2].set_ylabel("Token ID")
        
        # Plot Ground Truth markers
        x_coords = np.arange(num_steps) + 0.5
        y_coords = np.array(ground_truth_list) + 0.5
        valid_mask = y_coords < vocab_to_plot
        axes[2].scatter(x_coords[valid_mask], y_coords[valid_mask], color='red', marker='x', s=40)
        
        # X Axis
        axes[2].set_xticks(np.arange(num_steps) + 0.5)
        axes[2].set_xticklabels(x_tick_labels, rotation=0, fontsize=9)
        axes[2].set_xlabel("Current Token (Input)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"series_{name}_sample_{r_idx}.png"))
        plt.close()


# =========================================================
#  Main
# =========================================================
def main():
    print("Loading Resources...")
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

    print("Loading Model...")
    model, c = load_model_safe(MODEL_PATH, network, base_N, dist_mat)
    
    print("Loading Data...")
    data = np.load(NPZ_PATH)
    dataset_dict = {
        "routes": data['route_arr'],
        "agent_ids": data['agent_ids'] if 'agent_ids' in data else np.zeros(len(data['route_arr'])),
        "holidays": data['holiday_arr'],
        "timezones": data['time_zone_arr'],
        "events": data['event_arr']
    }

    # Schur Analysis
    A_np = model.A.detach().cpu().numpy()
    schur = SchurAnalyzer(A_np)
    
    # 0. Global Plaza Effect (Delta Logits)
    # ターゲットやサンプルに依存しないため、最初に1回だけ実行
    analyze_delta_logits(model, network, OUT_DIR)
    
    # Run Analysis for each target
    for target in TARGETS:
        # 1. Forcing Diff
        analyze_forcing_diff(model, target, schur, OUT_DIR)
        analyze_forcing_decomposition(model, target, schur, OUT_DIR)
        
        # 2. Trajectory Series Analysis
        run_trajectory_analysis_series(model, network, dataset_dict, target, schur, OUT_DIR)
        
    print(f"Deep Analysis V4 (Plaza Effect) Completed. Saved to {OUT_DIR}")

if __name__ == "__main__":
    main()