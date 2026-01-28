#!/usr/bin/env python3
"""
Ablation check (Fixed Config Version)
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch
from KP_RF import KoopmanRoutesFormer

# ==========================================
# ★★★ 設定エリア (ここを書き換えてください) ★★★
# ==========================================
class Config:
    # 1. モデルの重みファイル
    model_path = "/home/mizutani/projects/RF/runs/20260121_145835/model_weights_20260121_145835.pth"
    
    # 2. データセット
    npz_path = "/home/mizutani/projects/RF/data/input_real_m4.npz"
    
    # 3. 隣接行列
    base_adj_path = "/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt"
    
    # 4. 広場ID
    plaza_id = 2
    
    # その他
    batch_size = 64
    max_seqs = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
        row = row.tolist(); 
        try: pad_pos = row.index(PAD_ID); row = row[:pad_pos]
        except ValueError: pass
        while len(row) > 0 and row[-1] == PAD_ID: row.pop()
        out.append(row)
    return out

def make_teacher_forcing_pair(seq): return [BOS_ID] + seq, seq + [EOS_ID]

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

def make_choice_mask(curr_tokens, expanded_adj, vocab_size):
    # print("DEBUG curr_tokens.shape =", tuple(curr_tokens.shape))
    device = curr_tokens.device
    bsz, tlen = curr_tokens.shape
    # print("DEBUG tlen =", tlen, "N_EXPANDED =", N_EXPANDED)
    allowed = torch.zeros((bsz, tlen, vocab_size), dtype=torch.bool, device=curr_tokens.device)
    # print("DEBUG allowed.shape =", tuple(allowed.shape))
    
    # マスク作成
    normal = (curr_tokens >= 0) & (curr_tokens < N_EXPANDED)
    
    if normal.any():
        # 1. 隣接行列を正しいデバイスへ
        rows = expanded_adj.to(device=device, dtype=torch.bool)
        
        # 2. 該当するトークンのインデックスを取得
        idx = curr_tokens[normal].long()
        
        # 3. 許可される遷移先を取得 [K, 38]
        allowed_rows = rows[idx]
        
        # 4. 代入先のテンソルを一時的に取り出す [K, 42]
        allowed_normal = allowed[normal]
        
        # 5. 前半部分(0~37)に代入
        allowed_normal[:, :N_EXPANDED] = allowed_rows
        
        # 6. 元のテンソルに戻す
        allowed[normal] = allowed_normal

    # BOS, EOS等の処理（変更なし）
    bos = (curr_tokens == BOS_ID)
    if bos.any():
        # bos: [B,T]
        b_idx, t_idx = torch.where(bos)          # 1次元 index に分解
        allowed[b_idx, t_idx, 0:N_EXPANDED] = True
    
    not_pad = (curr_tokens != PAD_ID)
    if not_pad.any():
        b_idx, t_idx = torch.where(not_pad)
        allowed[b_idx, t_idx, EOS_ID] = True
    
    allowed[:, :, PAD_ID] = False
    allowed[:, :, BOS_ID] = False
    if MASK_ID < vocab_size:
        allowed[:, :, MASK_ID] = False
        
    return allowed
    
def main():
    args = Config()
    device = torch.device(args.device)

    print("Loading data...")
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
    if "to_logits.weight" in state: 
        vocab_size = state["to_logits.weight"].shape[0]
    else: vocab_size = VOCAB_SIZE_DEFAULT
    token_emb_dim = state["token_embedding.weight"].shape[1]
    z_dim = state["to_z.weight"].shape[0]
    d_model = state["input_proj.weight"].shape[0]

    # チェックポイント内に delta 系パラメータがあるかデバッグ（num=4のはず）
    delta_keys = [k for k in state.keys() if "delta" in k.lower() or "plaza" in k.lower()]
    print("ckpt delta/plaza keys:", delta_keys[:50])
    print("num:", len(delta_keys))


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
    print("missing keys:", len(missing))
    print("unexpected keys:", len(unexpected))
    print("missing examples:", missing[:30])
    print("unexpected examples:", unexpected[:30])

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    if hasattr(model, "set_plaza_dist"): model.set_plaza_dist(dist_mat.to(device), args.plaza_id)

    sum_logp_on = 0.0; sum_logp_off = 0.0
    sum_acc_on = 0; sum_acc_off = 0
    n_steps = 0

    print("Running ablation check...")
    for start in range(0, len(input_seqs), args.batch_size):
        end = min(len(input_seqs), start + args.batch_size)
        tok_in = pad_batch(input_seqs[start:end], PAD_ID).to(device)
        tok_tg = pad_batch(target_seqs[start:end], PAD_ID).to(device)
        stay = stay_counts_from_tokens(tok_in)
        agent_b = torch.tensor(agent_ids[start:end], device=device, dtype=torch.long)
        time_b = torch.tensor(time_arr[start:end], device=device, dtype=torch.long) if time_arr is not None else None

        with torch.no_grad():
            _, _, _, _, dbg = model(tok_in, stay, agent_b, time_tensor=time_b, return_debug=True)
            base_logits = dbg["base_logits"]; delta_bias = dbg["delta_bias"]

        if start == 0:
            print("base_logits:", base_logits.shape, "delta_bias:", delta_bias.shape)
            print("delta_bias abs mean/max:", delta_bias.abs().mean().item(), delta_bias.abs().max().item())
            v = ((tok_in != PAD_ID) & (tok_in != EOS_ID) & (tok_in != BOS_ID) & (tok_in != MASK_ID) & (tok_tg != PAD_ID))
            if v.any():
                db = delta_bias[v]  # [N, vocab]
                print("delta_bias vocab-std mean:", db.std(dim=-1).mean().item())


        allowed = make_choice_mask(tok_in, expanded_adj, vocab_size)
        prob_off = torch.softmax(base_logits.masked_fill(~allowed, -1e9), dim=-1)
        prob_on = torch.softmax((base_logits + delta_bias).masked_fill(~allowed, -1e9), dim=-1)

        valid = (tok_in != PAD_ID) & (tok_in != EOS_ID) & (tok_in != BOS_ID) & (tok_in != MASK_ID)
        valid = valid & (tok_tg != PAD_ID)
        if not valid.any(): continue

        true_next = tok_tg.clamp(min=0, max=vocab_size - 1)
        p_true_off = torch.gather(prob_off, -1, true_next.unsqueeze(-1)).squeeze(-1)
        p_true_on = torch.gather(prob_on, -1, true_next.unsqueeze(-1)).squeeze(-1)

        logp_off = torch.log(p_true_off + 1e-12)
        logp_on = torch.log(p_true_on + 1e-12)

        pred_off = prob_off.argmax(dim=-1); pred_on = prob_on.argmax(dim=-1)
        acc_off = (pred_off == true_next); acc_on = (pred_on == true_next)

        n = int(valid.sum().item())
        n_steps += n
        sum_logp_off += float(logp_off[valid].sum().item())
        sum_logp_on += float(logp_on[valid].sum().item())
        sum_acc_off += int(acc_off[valid].sum().item())
        sum_acc_on += int(acc_on[valid].sum().item())

    print("n_steps:", n_steps)
    if n_steps > 0:
        print("avg_logp_off:", sum_logp_off / n_steps)
        print("avg_logp_on :", sum_logp_on / n_steps)
        print(f"delta_logp  : {(sum_logp_on - sum_logp_off) / n_steps:.6f} (Positive is Better)")
        print("acc_off     :", sum_acc_off / n_steps)
        print("acc_on      :", sum_acc_on / n_steps)
        print(f"delta_acc   : {(sum_acc_on - sum_acc_off) / n_steps:.6f} (Positive is Better)")

if __name__ == "__main__":
    main()