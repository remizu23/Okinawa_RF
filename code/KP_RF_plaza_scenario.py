#!/usr/bin/env python3
"""
Scenario rollouts (Fixed Config Version)
"""
from __future__ import annotations
import os
import argparse
from typing import List, Optional, Tuple
import numpy as np
import torch
from KP_RF import KoopmanRoutesFormer

# ==========================================
# ★★★ 設定エリア (ここを書き換えてください) ★★★
# ==========================================
class Config:
    # 1. モデルの重みファイル
    model_path = "/home/mizutani/projects/RF/runs/20260121_145835/model_weights_20260121_145835.pth"
    
    # 6. 出力フォルダ
    out_dir = f"/home/mizutani/projects/RF/runs/simulation_2_13_m4_2hop"


    # 2. データセット
    npz_path = "/home/mizutani/projects/RF/data/input_real_m4.npz"
    
    # 3. 隣接行列
    base_adj_path = "/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt"
    
    # 4. 比較したい広場のIDリスト (例: [2, 5] なら、ノード2の場合とノード5の場合をシミュレーション)
    plaza_ids = [2, 13] 
    
    # 5. シミュレーション設定
    prefix_steps = 3   # 最初の実データを使う時間
    horizon_steps = 30 # その後生成させる時間
    n_rollouts = 10     # 1つのデータにつき20回生成して平均をとる
    max_prefixes = 500  # 計算時間短縮のため500件だけ使う (0なら全件)
    
    
    # その他
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    temperature = 1.0
    top_p = 0.9
    nhead = 4
    d_ff = 128
    num_layers = 3

# ==========================================

PAD_ID = 38; EOS_ID = 39; BOS_ID = 40; MASK_ID = 41
N_BASE = 19; N_EXPANDED = 38; VOCAB_SIZE_DEFAULT = 42

def expand_adjacency_matrix(base_adj):
    base_adj = base_adj.to(dtype=torch.bool); n = base_adj.size(0); eye = torch.eye(n, dtype=torch.bool)
    return torch.cat([torch.cat([base_adj, eye], 1), torch.cat([base_adj, eye], 1)], 0)

def compute_hop_dist(base_adj):
    base_adj = base_adj.to(dtype=torch.bool).cpu(); n = base_adj.size(0)
    dist = torch.full((n, n), 1_000_000_000, dtype=torch.long)
    for s in range(n):
        dist[s, s] = 0; q = [s]; head = 0
        while head < len(q):
            u = q[head]; head += 1; du = dist[s, u].item()
            for v in torch.where(base_adj[u])[0].tolist():
                if dist[s, v].item() > du + 1: dist[s, v] = du + 1; q.append(v)
    return dist

def trim_and_tokenize(route_arr):
    out = []
    for row in route_arr:
        row = row.tolist()
        try: pad_pos = row.index(PAD_ID); row = row[:pad_pos]
        except ValueError: pass
        while len(row) > 0 and row[-1] == PAD_ID: row.pop()
        out.append(row)
    return out

def make_input_prefix(seq, prefix_steps):
    inp = [BOS_ID] + seq
    return inp[: max(1, min(len(inp), prefix_steps + 1))]

def pad_batch(seqs, pad_value):
    max_len = max((len(s) for s in seqs), default=0)
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for i, s in enumerate(seqs):
        if len(s) == 0: continue
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out

def stay_counts_from_tokens(tokens):
    device = tokens.device; tok = tokens.cpu().numpy(); bsz, tlen = tok.shape
    counts = np.zeros_like(tok, dtype=np.int64); special = {PAD_ID, EOS_ID, BOS_ID, MASK_ID}
    for b in range(bsz):
        current = None; c = 0
        for t in range(tlen):
            v = int(tok[b, t])
            if v in special: current = None; c = 0; counts[b, t] = 0; continue
            if current is None or v != current: current = v; c = 1
            else: c += 1
            counts[b, t] = c
    return torch.tensor(counts, device=device, dtype=torch.long)

def base_id_from_token(token):
    out = torch.full_like(token, -1)
    m = (token >= 0) & (token < N_BASE); out[m] = token[m]
    s = (token >= N_BASE) & (token < N_EXPANDED); out[s] = token[s] - N_BASE
    return out

def make_allowed_next(curr_tok, expanded_adj, vocab_size):
    """
    【修正版】1ステップごとの生成用マスク関数
    curr_tok: [Batch] (1次元)
    戻り値: [Batch, Vocab]
    """
    device = curr_tok.device
    bsz = curr_tok.size(0)
    
    # [Batch, Vocab] で初期化
    allowed = torch.zeros((bsz, vocab_size), dtype=torch.bool, device=device)
    
    # --- 1. 通常ノード (0~37) ---
    normal = (curr_tok >= 0) & (curr_tok < N_EXPANDED)
    if normal.any():
        # 隣接行列から取得 [K, 38]
        rows = expanded_adj.to(device=device, dtype=torch.bool)
        idx = curr_tok[normal].long()
        allowed_rows = rows[idx]
        
        # 安全な代入
        allowed_normal = allowed[normal]
        allowed_normal[:, :N_EXPANDED] = allowed_rows
        allowed[normal] = allowed_normal

    # --- 2. BOS ---
    bos = (curr_tok == BOS_ID)
    if bos.any():
        allowed[bos, :N_EXPANDED] = True

    # --- 3. EOS (パディング以外なら遷移可能) ---
    not_pad = (curr_tok != PAD_ID)
    if not_pad.any():
        # ★ここが修正ポイント: 1次元なので単純なインデックス参照でOK
        allowed[not_pad, EOS_ID] = True

    # --- 4. 禁止トークン ---
    allowed[:, PAD_ID] = False
    allowed[:, BOS_ID] = False
    if MASK_ID < vocab_size:
        allowed[:, MASK_ID] = False
        
    return allowed

def sample_next(prob, allowed, temperature, top_p):
    p = prob.clone() * allowed.float(); p = p / (p.sum(dim=-1, keepdim=True) + 1e-12)
    if temperature != 1.0: logits = torch.log(p + 1e-12) / temperature; p = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_p, sorted_idx = torch.sort(p, descending=True, dim=-1)
        cdf = torch.cumsum(sorted_p, dim=-1)
        keep = cdf <= top_p; keep[:, 0] = True
        sorted_p = sorted_p * keep.float(); sorted_p = sorted_p / (sorted_p.sum(dim=-1, keepdim=True) + 1e-12)
        next_in_sorted = torch.multinomial(sorted_p, num_samples=1).squeeze(1)
        return sorted_idx.gather(1, next_in_sorted.unsqueeze(1)).squeeze(1)
    return torch.multinomial(p, num_samples=1).squeeze(1)

def write_csv(path, header, rows):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

def main():
    args = Config()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    
    plaza_ids = args.plaza_ids # list

    print("Loading data...")
    npz = np.load(args.npz_path)
    route_arr = npz["route_arr"]
    time_arr = npz["time_arr"] if "time_arr" in npz else None
    agent_ids = npz["agent_ids"] if "agent_ids" in npz else np.zeros(len(route_arr), dtype=np.int64)

    seqs = trim_and_tokenize(route_arr)
    prefixes = [make_input_prefix(s, args.prefix_steps) for s in seqs]
    if args.max_prefixes > 0:
        prefixes = prefixes[: args.max_prefixes]
        agent_ids = agent_ids[: args.max_prefixes]
        if time_arr is not None: time_arr = time_arr[: args.max_prefixes]

    print("Loading adjacency...")
    base_adj = torch.load(args.base_adj_path, map_location="cpu")
    if isinstance(base_adj, dict) and "adj" in base_adj: base_adj = base_adj["adj"]
    if not isinstance(base_adj, torch.Tensor): base_adj = torch.tensor(base_adj)
    base_adj = base_adj.to(dtype=torch.bool)
    dist_mat = compute_hop_dist(base_adj)
    expanded_adj = expand_adjacency_matrix(base_adj)

    print("Loading model...")
    ckpt = torch.load(args.model_path, map_location="cpu")
    
    # ★修正: 正しいキー(model_state_dict)を探すロジックに変更
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    if "to_logits.weight" in state: vocab_size = state["to_logits.weight"].shape[0]
    else: vocab_size = VOCAB_SIZE_DEFAULT
    token_emb_dim = state["token_embedding.weight"].shape[1]
    z_dim = state["to_z.weight"].shape[0]
    d_model = state["input_proj.weight"].shape[0]

    model = KoopmanRoutesFormer(
        vocab_size=vocab_size,
        token_emb_dim=token_emb_dim,
        d_model=d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        z_dim=z_dim,
        pad_token_id=PAD_ID,
        
        # ★★★ ここを追加！ ★★★
        dist_mat_base=dist_mat, 
        base_N=dist_mat.shape[0],
        
        num_agents=int(state["agent_embedding.weight"].shape[0]) if "agent_embedding.weight" in state else 1,
        agent_emb_dim=int(state["agent_embedding.weight"].shape[1]) if "agent_embedding.weight" in state else 16,
        max_stay_count=(int(state["stay_embedding.weight"].shape[0]) - 1) if "stay_embedding.weight" in state else 500,
        stay_emb_dim=int(state["stay_embedding.weight"].shape[1]) if "stay_embedding.weight" in state else 16,
        # use_year_embedding=("year_2025_embedding" in state),
    )
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    occ_by_plaza = {}

    for plaza in plaza_ids:
        print(f"Simulating scenario: Plaza at Node {plaza}...")
        if hasattr(model, "set_plaza_dist"):
            model.set_plaza_dist(dist_mat.to(device), plaza)
        else:
            print("Warning: set_plaza_dist missing.")

        occ = torch.zeros((args.horizon_steps + 1, N_BASE), device=device)
        total_samples = 0

        for start in range(0, len(prefixes), args.batch_size):
            end = min(len(prefixes), start + args.batch_size)
            pref_b = prefixes[start:end]
            agent_b = torch.tensor(agent_ids[start:end], device=device, dtype=torch.long)
            time_b = torch.tensor(time_arr[start:end], device=device, dtype=torch.long) if time_arr is not None else None

            pref_rep = pref_b * args.n_rollouts
            agent_rep = agent_b.repeat_interleave(args.n_rollouts)
            time_rep = time_b.repeat_interleave(args.n_rollouts) if time_b is not None else None
            seqs_live = [p[:] for p in pref_rep]

            last_tok0 = torch.tensor([s[-1] for s in seqs_live], device=device, dtype=torch.long)
            base0 = base_id_from_token(last_tok0)
            for b in base0[base0 >= 0].tolist(): occ[0, b] += 1
            ended = torch.zeros((len(seqs_live),), device=device, dtype=torch.bool)

            for t in range(1, args.horizon_steps + 1):
                tok_in = pad_batch(seqs_live, PAD_ID).to(device)
                stay_b = stay_counts_from_tokens(tok_in)
                with torch.no_grad():
                    logits, _, _, _, dbg = model(tok_in, stay_b, agent_rep, time_tensor=time_rep, return_debug=True)
                    base_logits = dbg["base_logits"]; delta_bias = dbg["delta_bias"]

                lengths = torch.tensor([len(s) for s in seqs_live], device=device, dtype=torch.long)
                last_pos = lengths - 1
                batch_idx = torch.arange(len(seqs_live), device=device)
                curr_tok = tok_in[batch_idx, last_pos]
                allowed = make_allowed_next(curr_tok, expanded_adj, vocab_size)
                logits_next = (base_logits + delta_bias)[batch_idx, last_pos, :]
                probs = torch.softmax(logits_next.masked_fill(~allowed, -1e9), dim=-1)
                next_tok = sample_next(probs, allowed, temperature=args.temperature, top_p=args.top_p)
                next_tok = torch.where(ended, torch.full_like(next_tok, EOS_ID), next_tok)

                for i, nt in enumerate(next_tok.tolist()):
                    if ended[i]: continue
                    seqs_live[i].append(int(nt))
                    if nt == EOS_ID: ended[i] = True
                
                base_now = base_id_from_token(next_tok)
                for b in base_now[base_now >= 0].tolist(): occ[t, b] += 1
            total_samples += len(seqs_live)

        occ = occ / max(1, total_samples)
        occ_by_plaza[plaza] = occ.detach().cpu()
        rows = [[t, b, float(occ[t, b].item())] for t in range(occ.size(0)) for b in range(N_BASE)]
        write_csv(os.path.join(args.out_dir, f"occupancy_by_node_plaza{plaza}.csv"), ["t", "base_node", "presence_prob"], rows)

    if len(plaza_ids) >= 2:
        a, b = plaza_ids[0], plaza_ids[1]
        diff = occ_by_plaza[a] - occ_by_plaza[b]
        rows = [[t, n0, float(diff[t, n0].item())] for t in range(diff.size(0)) for n0 in range(N_BASE)]
        write_csv(os.path.join(args.out_dir, f"diff_plaza{a}_minus_plaza{b}.csv"), ["t", "base_node", "presence_prob_diff"], rows)

    print("[Done] Scenario Rollout Completed.")

if __name__ == "__main__":
    main()