#!/usr/bin/env python3
"""
Critical plaza-attraction analysis (Fixed Config Version)
"""
from __future__ import annotations
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import torch
from KP_RF import KoopmanRoutesFormer

# ==========================================
# ★★★ 設定エリア (ここを書き換えてください) ★★★
# ==========================================
class Config:
    # 1. モデルの重みファイル (.pth)
    model_path = "/home/mizutani/projects/RF/runs/20260121_145835/model_weights_20260121_145835.pth"
    
    # 2. データセット (.npz)
    npz_path = "/home/mizutani/projects/RF/data/input_real_m4.npz"
    
    # 3. 隣接行列 (.pt)
    base_adj_path = "/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt"
    
    # 4. 広場とするノードID (デフォルト: 2)
    plaza_id = 2
    
    # 5. 出力フォルダ名
    out_dir = "/home/mizutani/projects/RF/runs/20260121_145835/plaza_effect"
    
    # その他設定
    batch_size = 64
    max_seqs = 0  # 0なら全データ使用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # モデルのハイパーパラメータ (学習時と同じにする)
    nhead = 4
    d_ff = 128
    num_layers = 3

# ==========================================

# 定数定義
PAD_ID = 38
EOS_ID = 39
BOS_ID = 40
MASK_ID = 41
N_BASE = 19
N_EXPANDED = 38
VOCAB_SIZE_DEFAULT = 42

def expand_adjacency_matrix(base_adj: torch.Tensor) -> torch.Tensor:
    base_adj = base_adj.to(dtype=torch.bool)
    n = base_adj.size(0)
    eye = torch.eye(n, dtype=torch.bool)
    mm = base_adj; ms = eye; sm = base_adj; ss = eye
    top = torch.cat([mm, ms], dim=1)
    bottom = torch.cat([sm, ss], dim=1)
    return torch.cat([top, bottom], dim=0)

def compute_hop_dist(base_adj: torch.Tensor) -> torch.Tensor:
    base_adj = base_adj.to(dtype=torch.bool).cpu()
    n = base_adj.size(0)
    dist = torch.full((n, n), 1_000_000_000, dtype=torch.long)
    for s in range(n):
        dist[s, s] = 0
        q = [s]; head = 0
        while head < len(q):
            u = q[head]; head += 1
            du = dist[s, u].item()
            nbrs = torch.where(base_adj[u])[0].tolist()
            for v in nbrs:
                if dist[s, v].item() > du + 1:
                    dist[s, v] = du + 1
                    q.append(v)
    return dist

def trim_and_tokenize(route_arr: np.ndarray) -> List[List[int]]:
    seqs: List[List[int]] = []
    for row in route_arr:
        row = row.tolist()
        try:
            pad_pos = row.index(PAD_ID)
            row = row[:pad_pos]
        except ValueError: pass
        while len(row) > 0 and row[-1] == PAD_ID: row.pop()
        seqs.append(row)
    return seqs

def make_teacher_forcing_pair(seq: List[int]) -> Tuple[List[int], List[int]]:
    return [BOS_ID] + seq, seq + [EOS_ID]

def pad_batch(seqs: List[List[int]], pad_value: int) -> torch.Tensor:
    max_len = max((len(s) for s in seqs), default=0)
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for i, s in enumerate(seqs):
        if len(s) == 0: continue
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out

def stay_counts_from_tokens(tokens: torch.Tensor) -> torch.Tensor:
    device = tokens.device
    tok = tokens.cpu().numpy()
    bsz, tlen = tok.shape
    counts = np.zeros_like(tok, dtype=np.int64)
    special = {PAD_ID, EOS_ID, BOS_ID, MASK_ID}
    for b in range(bsz):
        current = None; c = 0
        for t in range(tlen):
            v = int(tok[b, t])
            if v in special:
                current = None; c = 0; counts[b, t] = 0
                continue
            if current is None or v != current:
                current = v; c = 1
            else: c += 1
            counts[b, t] = c
    return torch.tensor(counts, device=device, dtype=torch.long)

def base_id_from_token(tokens: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(tokens, -1)
    m = (tokens >= 0) & (tokens < N_BASE)
    out[m] = tokens[m]
    s = (tokens >= N_BASE) & (tokens < N_EXPANDED)
    out[s] = tokens[s] - N_BASE
    return out

def make_adjacency_choice_mask(curr_tokens: torch.Tensor, expanded_adj: torch.Tensor, vocab_size: int) -> torch.Tensor:
    device = curr_tokens.device
    bsz, tlen = curr_tokens.shape
    allowed = torch.zeros((bsz, tlen, vocab_size), dtype=torch.bool, device=device)
    normal = (curr_tokens >= 0) & (curr_tokens < N_EXPANDED)
    if normal.any():
        rows = expanded_adj.to(device=device, dtype=torch.bool)
        idx = curr_tokens[normal].long()
        allowed_rows = rows[idx]
        allowed_normal = allowed[normal]
        allowed_normal[:, :N_EXPANDED] = allowed_rows
        allowed[normal] = allowed_normal
    bos = (curr_tokens == BOS_ID)
    if bos.any():
        # bos: [B,T]
        b_idx, t_idx = torch.where(bos)          # 1次元 index に分解
        allowed[b_idx, t_idx, 0:N_EXPANDED] = True

    not_pad = curr_tokens != PAD_ID
    if not_pad.any():
        b_idx, t_idx = torch.where(not_pad)
        allowed[b_idx, t_idx, EOS_ID] = True

    allowed[:, :, PAD_ID] = False; allowed[:, :, BOS_ID] = False
    if MASK_ID < vocab_size: allowed[:, :, MASK_ID] = False
    return allowed

def masked_softmax(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.masked_fill(~allowed, -1e9)
    return torch.softmax(masked_logits, dim=-1)

@dataclass
class Agg:
    n: int = 0
    sum_tow_on: float = 0.0; sum_tow_off: float = 0.0
    sum_same_on: float = 0.0; sum_same_off: float = 0.0
    sum_away_on: float = 0.0; sum_away_off: float = 0.0
    sum_gate: float = 0.0
    sum_logp_on: float = 0.0; sum_logp_off: float = 0.0

    def add(self, tow_on, tow_off, same_on, same_off, away_on, away_off, gate, logp_on, logp_off):
        self.n += int(tow_on.numel())
        self.sum_tow_on += float(tow_on.sum().item()); self.sum_tow_off += float(tow_off.sum().item())
        self.sum_same_on += float(same_on.sum().item()); self.sum_same_off += float(same_off.sum().item())
        self.sum_away_on += float(away_on.sum().item()); self.sum_away_off += float(away_off.sum().item())
        self.sum_gate += float(gate.sum().item())
        self.sum_logp_on += float(logp_on.sum().item()); self.sum_logp_off += float(logp_off.sum().item())

    def as_row(self) -> Dict[str, float]:
        if self.n == 0: return {}
        toward_on = self.sum_tow_on / self.n; toward_off = self.sum_tow_off / self.n
        same_on = self.sum_same_on / self.n; same_off = self.sum_same_off / self.n
        away_on = self.sum_away_on / self.n; away_off = self.sum_away_off / self.n
        gate_mean = self.sum_gate / self.n
        avg_logp_on = self.sum_logp_on / self.n; avg_logp_off = self.sum_logp_off / self.n
        return {
            "n_steps": self.n,
            "toward_on": toward_on, "toward_off": toward_off, "delta_toward": toward_on - toward_off,
            "same_on": same_on, "same_off": same_off,
            "away_on": away_on, "away_off": away_off,
            "gate_mean": gate_mean,
            "avg_logp_on": avg_logp_on, "avg_logp_off": avg_logp_off, "delta_logp": avg_logp_on - avg_logp_off,
        }

def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    import csv
    if not rows: return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def main():
    args = Config() # Use the Config class instead of argparse
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading data from {args.npz_path}...")
    npz = np.load(args.npz_path)
    route_arr = npz["route_arr"]
    time_arr = npz["time_arr"] if "time_arr" in npz else None
    agent_ids = npz["agent_ids"] if "agent_ids" in npz else np.zeros(len(route_arr), dtype=np.int64)

    if args.max_seqs > 0:
        route_arr = route_arr[: args.max_seqs]
        agent_ids = agent_ids[: args.max_seqs]
        if time_arr is not None: time_arr = time_arr[: args.max_seqs]

    seqs = trim_and_tokenize(route_arr)
    pairs = [make_teacher_forcing_pair(s) for s in seqs]
    input_seqs = [p[0] for p in pairs]
    target_seqs = [p[1] for p in pairs]

    print(f"Loading adjacency from {args.base_adj_path}...")
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
        
        # ★★★ 追加: ここで距離行列を渡して、バイアス層を作成させる ★★★
        dist_mat_base=dist_mat,
        base_N=dist_mat.shape[0], # または N_BASE
        
        num_agents=int(state["agent_embedding.weight"].shape[0]) if "agent_embedding.weight" in state else 1,
        agent_emb_dim=int(state["agent_embedding.weight"].shape[1]) if "agent_embedding.weight" in state else 16,
        max_stay_count=(int(state["stay_embedding.weight"].shape[0]) - 1) if "stay_embedding.weight" in state else 500,
        stay_emb_dim=int(state["stay_embedding.weight"].shape[1]) if "stay_embedding.weight" in state else 16,
    )

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing: print("[load_state_dict] missing:", missing[:5], "...")
    
    model.to(device)
    model.eval()
    
    # KP_RF.pyに set_plaza_dist が実装されている前提
    if hasattr(model, "set_plaza_dist"):
        model.set_plaza_dist(dist_mat.to(device), args.plaza_id)
    else:
        print("Warning: model.set_plaza_dist not found. Results may be inaccurate.")

    move_ids = torch.arange(0, N_BASE, device=device, dtype=torch.long)
    agg_overall = Agg()
    agg_by_entrance = defaultdict(Agg)
    agg_by_curr_dist = defaultdict(Agg)

    print("Running analysis...")
    n = len(input_seqs)
    for start in range(0, n, args.batch_size):
        end = min(n, start + args.batch_size)
        in_batch = input_seqs[start:end]
        tg_batch = target_seqs[start:end]
        tok_in = pad_batch(in_batch, PAD_ID).to(device)
        tok_tg = pad_batch(tg_batch, PAD_ID).to(device)
        agent_b = torch.tensor(agent_ids[start:end], device=device, dtype=torch.long)
        time_b = torch.tensor(time_arr[start:end], device=device, dtype=torch.long) if time_arr is not None else None
        stay_b = stay_counts_from_tokens(tok_in)

        # KP_RF.py の forward が return_debug=True に対応している前提
        with torch.no_grad():
            try:
                logits, z_hat, _, _, dbg = model(tok_in, stay_b, agent_b, time_tensor=time_b, return_debug=True)
            except TypeError:
                print("Error: Your model.forward() does not accept 'return_debug=True'. Please update KP_RF.py.")
                return

            base_logits = dbg["base_logits"]
            delta_bias = dbg["delta_bias"]
            gate = dbg["delta_gate"]

        curr_tok = tok_in
        true_next = tok_tg
        allowed = make_adjacency_choice_mask(curr_tok, expanded_adj, vocab_size)
        prob_off = masked_softmax(base_logits, allowed)
        prob_on = masked_softmax(base_logits + delta_bias, allowed)

        curr_base = base_id_from_token(curr_tok)
        valid_pos = (curr_tok != PAD_ID) & (curr_tok != EOS_ID) & (curr_tok != BOS_ID) & (curr_tok != MASK_ID)
        valid_pos = valid_pos & (true_next != PAD_ID) & (curr_base >= 0)

        if not valid_pos.any(): continue

        d_curr = dist_mat.to(device)[curr_base.clamp(min=0), args.plaza_id]
        d_move = dist_mat.to(device)[move_ids, args.plaza_id]
        delta = d_move.view(1, 1, -1) - d_curr.unsqueeze(-1)
        bin_id = torch.where(delta < 0, 0, torch.where(delta == 0, 1, 2)).long()

        p_off_move = prob_off.index_select(-1, move_ids)
        p_on_move = prob_on.index_select(-1, move_ids)

        tow_off = (p_off_move * (bin_id == 0)).sum(-1)
        tow_on = (p_on_move * (bin_id == 0)).sum(-1)
        same_off = (p_off_move * (bin_id == 1)).sum(-1)
        same_on = (p_on_move * (bin_id == 1)).sum(-1)
        away_off = (p_off_move * (bin_id == 2)).sum(-1)
        away_on = (p_on_move * (bin_id == 2)).sum(-1)

        true_next_clamped = true_next.clamp(min=0, max=vocab_size - 1)
        logp_off = torch.log(torch.gather(prob_off, -1, true_next_clamped.unsqueeze(-1)).squeeze(-1) + 1e-12)
        logp_on = torch.log(torch.gather(prob_on, -1, true_next_clamped.unsqueeze(-1)).squeeze(-1) + 1e-12)

        v = valid_pos
        agg_overall.add(tow_on[v], tow_off[v], same_on[v], same_off[v], away_on[v], away_off[v], gate.squeeze(-1)[v], logp_on[v], logp_off[v])

        # Grouping
        entrance_base = base_id_from_token(tok_in[:, 1:2]).squeeze(1)
        entrance_base = torch.where(entrance_base >= 0, entrance_base, torch.zeros_like(entrance_base))
        ent_bt = entrance_base.view(-1, 1).expand_as(curr_base)
        ent_v = ent_bt[v].long(); d_v = d_curr[v].long()

        for key in ent_v.unique().tolist():
            m = ent_v == key
            if m.any(): agg_by_entrance[int(key)].add(tow_on[v][m], tow_off[v][m], same_on[v][m], same_off[v][m], away_on[v][m], away_off[v][m], gate.squeeze(-1)[v][m], logp_on[v][m], logp_off[v][m])
        for key in d_v.unique().tolist():
            m = d_v == key
            if m.any(): agg_by_curr_dist[int(key)].add(tow_on[v][m], tow_off[v][m], same_on[v][m], same_off[v][m], away_on[v][m], away_off[v][m], gate.squeeze(-1)[v][m], logp_on[v][m], logp_off[v][m])

    # Write Outputs
    write_csv(os.path.join(args.out_dir, "overall.csv"), [{"group": "overall", **agg_overall.as_row()}])
    
    rows_ent = []
    for e in sorted(agg_by_entrance.keys()): rows_ent.append({"entrance_base": e, **agg_by_entrance[e].as_row()})
    write_csv(os.path.join(args.out_dir, "by_entrance.csv"), rows_ent)

    rows_dist = []
    for d0 in sorted(agg_by_curr_dist.keys()): rows_dist.append({"curr_dist_to_plaza": d0, **agg_by_curr_dist[d0].as_row()})
    write_csv(os.path.join(args.out_dir, "by_curr_dist.csv"), rows_dist)

    print(f"[Done] Analysis written to: {args.out_dir}")
    print("Overall Result:", agg_overall.as_row())

if __name__ == "__main__":
    main()