#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
広場滞在（event=1）を含む prefix を固定し、
その「広場滞在ステップだけ」 event を 1→0 に反転した counterfactual を作る。

同一prefix（同一の入力トークン列）に対して
  - ON: 元の event_arr
  - OFF: t0 (event=1の最初のステップ) だけ event=0 にしたもの
を teacher-forcing で forward し、広場滞在後の「次トークン分布」を比較する。

出力:
  1) token 分布差: ΔP_token(t, tok)
  2) location 分布差: ΔP_loc(t, node)  (move+stay を同一ノードに集約)
  3) ΔP_end(t), ΔEntropy(t), ΔNeff(t) など
  4) サンプルごとのCSV/集約CSV/簡易プロット

重要:
  - softmax の前に "拡張隣接行列" による隣接制約マスクを適用する
  - 「次トークン分布」は全トークン（move, stay, specials）で出す
  - 「地点分布」は move/stay を base node に集約し、special は別扱い
"""

import os
# argparse は削除
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

# user modules
from network import Network, expand_adjacency_matrix
from tokenization import Tokenization
from KP_RF import KoopmanRoutesFormer  # あなたのモデル

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- token conventions (ユーザー説明に合わせる) ----
BASE_N = 19
MOVE_START = 0
MOVE_END = 18
STAY_START = 19
STAY_END = 37
PAD = 38
END = 39
BEGIN = 40
MASK = 41  # unused

# ================= SETTINGS (HARDCODED) =================
NPZ_PATH = "/home/mizutani/projects/RF/data/input_real_m4_emb.npz"           # 実際のパスに変更してください
ADJ_PATH = "/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt"         # 実際のパスに変更してください
MODEL_PATH = "/home/mizutani/projects/RF/runs/20260123_210513/model_weights_20260123_210513.pth"     # 実際のパスに変更してください
OUT_DIR = "/home/mizutani/projects/RF/runs/20260123_210513/plaza_analysis"             # 出力先ディレクトリ
MAX_SAMPLES = 50                        # event=1 を含むサンプルから最大いくつ解析するか
HORIZON_STEPS = 20                      # 広場訪問後に何ステップ先まで比較するか
FOCUS_NODES_STR = "2,11,14"             # 地点分布・トークン分布で追跡したいノードID, カンマ区切り
# ========================================================


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def load_adj(adj_path: str) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    adj_path から隣接行列をロードし、
      - base_adj: [19,19]（baseノード）
      - expanded_adj: [38,38]（move+stay）
    を返す。
    """
    adj = torch.load(adj_path, weights_only=True)
    adj = adj.cpu()
    if adj.shape[0] == 38:
        base_adj = adj[:BASE_N, :BASE_N].clone()
        expanded_adj = adj.clone()
        base_n = BASE_N
    elif adj.shape[0] == BASE_N:
        base_adj = adj.clone()
        expanded_adj = expand_adjacency_matrix(adj)
        base_n = BASE_N
    else:
        # ここは不明確なので明示的に落とす（推測で動かさない）
        raise ValueError(f"Unexpected adjacency shape: {adj.shape}. Expected 19 or 38.")
    return base_adj, expanded_adj, base_n


def build_network(expanded_adj: torch.Tensor) -> Network:
    dummy_feat = torch.zeros((expanded_adj.shape[0], 1), dtype=torch.float32)
    # Network の引数仕様が「adj, feat」だと仮定（あなたの既存コードに合わせる）
    return Network(expanded_adj, dummy_feat)


def load_npz(npz_path: str) -> Dict[str, np.ndarray]:
    data = np.load(npz_path)
    out = dict(data)
    return out


def safe_get(arr_dict: Dict[str, np.ndarray], key: str, default=None):
    return arr_dict[key] if key in arr_dict else default


def align_ctx_1d(seq_1d: np.ndarray, target_len: int) -> torch.Tensor:
    """
    tokenization で <b> が先頭に入るので、
    context 系 (holiday/timezone/event) も同様に
      out[0] = 0
      out[1:1+L] = seq[:L]
    として長さ target_len に揃える。
    """
    out = np.zeros((target_len,), dtype=np.int64)
    L = min(len(seq_1d), target_len - 1)
    if L > 0:
        out[1:1+L] = seq_1d[:L]
    return torch.tensor(out, dtype=torch.long, device=DEVICE).unsqueeze(0)  # [1,T]


def token_to_base_node(tok: int) -> int:
    """
    move(0..18) -> 0..18
    stay(19..37) -> tok-19
    special は -1
    """
    if MOVE_START <= tok <= MOVE_END:
        return tok
    if STAY_START <= tok <= STAY_END:
        return tok - STAY_START
    return -1


def make_adjacency_mask_for_step(expanded_adj: torch.Tensor, curr_tok: int, vocab_size: int) -> torch.Tensor:
    """
    curr_tok（入力トークン）から、次トークンに許される集合を作り、
    logits に -inf を入れるマスクを返す。
    - expanded_adj は 38x38 を想定（move+stay）
    - vocab_size はモデルの vocab
    方針:
      - move/stay トークンは expanded_adj に従う
      - END は常に許可（ここはモデル設計依存だが、回遊終了比較が重要なので許可）
      - PAD/BEGIN/MASK は生成禁止（確率を見たいなら後で別途許可に変えてもよい）
    """
    allow = torch.zeros((vocab_size,), dtype=torch.bool, device=DEVICE)

    # move/stay領域
    if 0 <= curr_tok < 38:
        nbr = expanded_adj[curr_tok] > 0
        # nbr は 38次元、対応部分だけ True
        allow[:38] = nbr.to(DEVICE)
    else:
        # curr_tok が special のときは、隣接制約の定義が不明確なので、
        # ここでは「move/stay 全許可 + END許可」にする（曖昧さをコードで固定）
        allow[:38] = True

    # special tokens
    allow[END] = True
    # PAD/BEGIN/MASK は禁止のまま（False）
    allow[PAD] = False
    allow[BEGIN] = False
    if vocab_size > MASK:
        allow[MASK] = False

    # mask logits: disallow -> -inf
    # 返り値は「加算するマスク」（許可=0, 禁止=-inf）
    mask = torch.zeros((vocab_size,), dtype=torch.float32, device=DEVICE)
    mask[~allow] = float("-inf")
    return mask


def entropy_from_probs(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log(p)))


def neff_from_probs(p: np.ndarray, eps: float = 1e-12) -> float:
    # exp(H)
    return float(np.exp(entropy_from_probs(p, eps=eps)))


@dataclass
class SampleResult:
    sample_id: int
    t0_inp_index: int   # inp_tokens 上のインデックス（<b>込み）
    horizon: int
    delta_end: np.ndarray       # [H]
    delta_entropy: np.ndarray   # [H]
    delta_neff: np.ndarray      # [H]
    # token-wise と location-wise は巨大になり得るので必要なものだけ保存する
    delta_loc: np.ndarray       # [H, 19]
    delta_tok_focus: Dict[int, np.ndarray]  # tok -> [H]


def forward_logits(model, inp_tokens, stay_counts, agent_ids, holiday_in, tz_in, event_in):
    """
    model forward を呼んで logits を取る。
    返り値 logits: [1,T,V]
    """
    with torch.no_grad():
        logits, z_hat, z_pred, u_all = model(inp_tokens, stay_counts, agent_ids, holiday_in, tz_in, event_in)
    return logits


def analyze_one_sample(
    model,
    tokenizer: Tokenization,
    expanded_adj: torch.Tensor,
    route_tokens: np.ndarray,
    holiday_seq: np.ndarray,
    tz_seq: np.ndarray,
    event_seq: np.ndarray,
    agent_id: int,
    sample_id: int,
    horizon_steps: int,
    focus_nodes: List[int],
    out_dir: str,
) -> SampleResult or None:
    """
    1サンプルに対し、最初の event=1 ステップを t0 として
      ON: event 그대로
      OFF: t0 だけ 0
    の2forwardで、t0以降 horizon_steps だけ「次トークン分布差」を取る。

    teacher-forcing なので、入力トークン列は ON/OFF 共通。
    """
    # 有効長（PAD=38で切る）
    valid_len = int(np.sum(route_tokens != PAD))
    if valid_len < 5:
        return None

    route = route_tokens[:valid_len]
    holidays = holiday_seq[:valid_len]
    tz = tz_seq[:valid_len]
    events = event_seq[:valid_len]

    # t0 = 最初の event=1 位置
    idxs = np.where(events == 1)[0]
    if len(idxs) == 0:
        return None
    t0 = int(idxs[0])  # route 上（<b>なし）のインデックス

    # tokenization（inp_tokensは <b> + route）
    rt_b = torch.tensor(route, dtype=torch.long, device=DEVICE).unsqueeze(0)  # [1,L]
    inp_tokens = tokenizer.tokenization(rt_b, mode="simple").long().to(DEVICE)  # [1,T]
    stay_counts = tokenizer.calculate_stay_counts(inp_tokens)  # [1,T]

    B, T = inp_tokens.shape
    # context align
    holiday_in = align_ctx_1d(holidays, T)
    tz_in = align_ctx_1d(tz, T)

    # event ON
    event_on = align_ctx_1d(events, T)

    # event OFF（t0 だけ 0）
    events_cf = events.copy()
    events_cf[t0] = 0
    event_off = align_ctx_1d(events_cf, T)

    agent_ids = torch.tensor([agent_id], dtype=torch.long, device=DEVICE)

    # forward
    logits_on = forward_logits(model, inp_tokens, stay_counts, agent_ids, holiday_in, tz_in, event_on)  # [1,T,V]
    logits_off = forward_logits(model, inp_tokens, stay_counts, agent_ids, holiday_in, tz_in, event_off)

    V = logits_on.shape[-1]

    # 解析対象は「広場滞在(t0)の直後」から
    # 注意: inp_tokens は先頭に <b> があるので、route[t0] は inp_tokens[t0+1]
    t0_inp = t0 + 1

    H = min(horizon_steps, (T - 1) - t0_inp)  # 予測は position t の logits が「次」を表す想定
    if H <= 0:
        return None

    # Δ指標の格納
    delta_end = np.zeros((H,), dtype=np.float32)
    delta_entropy = np.zeros((H,), dtype=np.float32)
    delta_neff = np.zeros((H,), dtype=np.float32)
    delta_loc = np.zeros((H, BASE_N), dtype=np.float32)

    # focus token: move/stay 全体を “地点” として見る場合は loc の方で見るので、
    # ここでは「指定ノードへの遷移確率（地点）」を loc で出す。
    # 加えて、ユーザが「トークン分布も欲しい」ので focus_nodes を move と stay 両方に展開し、
    # それぞれの tok の ΔP を保存する（巨大にならない範囲）。
    delta_tok_focus: Dict[int, np.ndarray] = {}
    focus_toks = []
    for n in focus_nodes:
        if 0 <= n < BASE_N:
            focus_toks.append(n)           # move token
            focus_toks.append(STAY_START + n)  # stay token
    focus_toks = sorted(set([t for t in focus_toks if 0 <= t < V]))
    for t in focus_toks:
        delta_tok_focus[t] = np.zeros((H,), dtype=np.float32)

    # step loop
    for h in range(H):
        pos = t0_inp + h  # inp_tokens の位置（現在トークン）
        curr_tok = int(inp_tokens[0, pos].item())

        # adjacency mask (softmax前)
        mask_vec = make_adjacency_mask_for_step(expanded_adj, curr_tok, V)  # [V]

        logit_on = logits_on[0, pos] + mask_vec
        logit_off = logits_off[0, pos] + mask_vec

        p_on = F.softmax(logit_on, dim=-1).detach().cpu().numpy()
        p_off = F.softmax(logit_off, dim=-1).detach().cpu().numpy()
        dp = p_on - p_off

        # END
        if END < V:
            delta_end[h] = float(dp[END])

        # token entropy / Neff（全トークン分布）
        delta_entropy[h] = float(entropy_from_probs(p_on) - entropy_from_probs(p_off))
        delta_neff[h] = float(neff_from_probs(p_on) - neff_from_probs(p_off))

        # location distribution（move+stay を node に集約）
        # ここでは special は無視し、ノード0..18への確率質量だけ集約する
        loc_on = np.zeros((BASE_N,), dtype=np.float32)
        loc_off = np.zeros((BASE_N,), dtype=np.float32)
        for tok in range(min(V, 42)):  # vocabが42以上なら十分。小さければある範囲のみ。
            base = token_to_base_node(tok)
            if base >= 0:
                loc_on[base] += p_on[tok]
                loc_off[base] += p_off[tok]
        delta_loc[h] = loc_on - loc_off

        # focus tok ΔP を保存（トークン分布側）
        for ft in focus_toks:
            delta_tok_focus[ft][h] = float(dp[ft])

    # --- save per-sample CSV ---
    # (1) 時系列指標
    df_series = pd.DataFrame({
        "h": np.arange(H),
        "delta_P_end": delta_end,
        "delta_entropy_token": delta_entropy,
        "delta_neff_token": delta_neff,
    })
    df_series.to_csv(os.path.join(out_dir, f"sample_{sample_id:06d}_series.csv"), index=False)

    # (2) 地点分布差
    df_loc = pd.DataFrame(delta_loc, columns=[f"node_{i:02d}" for i in range(BASE_N)])
    df_loc.insert(0, "h", np.arange(H))
    df_loc.to_csv(os.path.join(out_dir, f"sample_{sample_id:06d}_delta_loc.csv"), index=False)

    # (3) focus token差
    if len(focus_toks) > 0:
        df_tok = pd.DataFrame({"h": np.arange(H)})
        for ft in focus_toks:
            df_tok[f"tok_{ft:02d}"] = delta_tok_focus[ft]
        df_tok.to_csv(os.path.join(out_dir, f"sample_{sample_id:06d}_delta_tok_focus.csv"), index=False)

    # --- quick plots (optional, lightweight) ---
    # location heatmap: [H,19] -> transpose for heatmap-like image without seaborn
    fig = plt.figure(figsize=(12, 4))
    plt.imshow(delta_loc.T, aspect="auto")
    plt.colorbar(label="ΔP(location) = P_on - P_off")
    plt.yticks(np.arange(BASE_N), [f"{i:02d}" for i in range(BASE_N)])
    plt.xlabel("h (steps after plaza-visit)")
    plt.ylabel("node")
    plt.title(f"Sample {sample_id}: Δ Location Prob after plaza (teacher-forcing)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"sample_{sample_id:06d}_delta_loc.png"))
    plt.close(fig)

    fig = plt.figure(figsize=(10, 3))
    plt.plot(delta_end, label="ΔP(end)")
    plt.plot(delta_entropy, label="ΔEntropy(token)")
    plt.plot(delta_neff, label="ΔNeff(token)")
    plt.axhline(0, linewidth=1)
    plt.legend()
    plt.xlabel("h")
    plt.title(f"Sample {sample_id}: summary deltas")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"sample_{sample_id:06d}_summary.png"))
    plt.close(fig)

    return SampleResult(
        sample_id=sample_id,
        t0_inp_index=t0_inp,
        horizon=H,
        delta_end=delta_end,
        delta_entropy=delta_entropy,
        delta_neff=delta_neff,
        delta_loc=delta_loc,
        delta_tok_focus=delta_tok_focus,
    )


def load_model_from_ckpt(model_path: str, network: Network, dist_mat_base: torch.Tensor, base_N: int) -> Tuple[KoopmanRoutesFormer, dict]:
    ckpt = torch.load(model_path, map_location=DEVICE)
    c = ckpt["config"]
    sd = ckpt["model_state_dict"]

    # infer agent/stay sizes if available
    det_num_agents = sd["agent_embedding.weight"].shape[0] if "agent_embedding.weight" in sd else c.get("num_agents", 1)
    det_max_stay = (sd["stay_embedding.weight"].shape[0] - 1) if "stay_embedding.weight" in sd else c.get("max_stay_count", 500)

    model = KoopmanRoutesFormer(
        vocab_size=c["vocab_size"],
        token_emb_dim=c.get("token_emb_dim", c.get("d_ie", 64)),
        d_model=c.get("d_model", c.get("d_ie", 64)),
        nhead=c.get("nhead", 4),
        num_layers=c.get("num_layers", 3),
        d_ff=c.get("d_ff", 128),
        z_dim=c["z_dim"],
        pad_token_id=network.N,          # あなたの既存コードに合わせた
        dist_mat_base=dist_mat_base,     # 使わないならダミーでOK
        base_N=base_N,
        holiday_emb_dim=c.get("holiday_emb_dim", c.get("holiday_emb_dim", 4)),
        time_zone_emb_dim=c.get("time_zone_emb_dim", c.get("time_zone_emb_dim", 4)),
        event_emb_dim=c.get("event_emb_dim", c.get("event_emb_dim", 4)),
        num_agents=det_num_agents,
        agent_emb_dim=c.get("agent_emb_dim", 16),
        max_stay_count=det_max_stay,
        stay_emb_dim=c.get("stay_emb_dim", 16),
    )
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE)
    model.eval()
    return model, c


def main():

    ensure_dir(OUT_DIR)

    # load adjacency & network
    base_adj, expanded_adj, base_n = load_adj(ADJ_PATH)
    network = build_network(expanded_adj)

    # dist_mat_base は今回の解析では使いませんが、モデル init 要件として渡す
    dist_dummy = torch.zeros((base_n, base_n), dtype=torch.long)

    # load model
    model, cfg = load_model_from_ckpt(MODEL_PATH, network, dist_dummy, base_n)

    # tokenizer
    tokenizer = Tokenization(network)

    # data
    data = load_npz(NPZ_PATH)
    routes = data["route_arr"]
    holidays = data["holiday_arr"]
    tz = data["time_zone_arr"]
    events = data["event_arr"]
    agent_ids = data["agent_ids"] if "agent_ids" in data else np.zeros((len(routes),), dtype=np.int64)

    focus_nodes = []
    for s in FOCUS_NODES_STR.split(","):
        s = s.strip()
        if s == "":
            continue
        focus_nodes.append(int(s))
    focus_nodes = [n for n in focus_nodes if 0 <= n < BASE_N]

    # iterate samples
    results: List[SampleResult] = []
    n = len(routes)
    for i in range(n):
        if len(results) >= MAX_SAMPLES:
            break

        res = analyze_one_sample(
            model=model,
            tokenizer=tokenizer,
            expanded_adj=expanded_adj.to(DEVICE),
            route_tokens=routes[i],
            holiday_seq=holidays[i],
            tz_seq=tz[i],
            event_seq=events[i],
            agent_id=int(agent_ids[i]),
            sample_id=i,
            horizon_steps=HORIZON_STEPS,
            focus_nodes=focus_nodes,
            out_dir=OUT_DIR,
        )
        if res is not None:
            results.append(res)

    if len(results) == 0:
        print("No samples with event=1 were found. Nothing to analyze.")
        return

    # ---- aggregate over samples (align by h) ----
    H = min([r.horizon for r in results])
    # truncate to common horizon
    delta_end_mat = np.stack([r.delta_end[:H] for r in results], axis=0)          # [S,H]
    delta_ent_mat = np.stack([r.delta_entropy[:H] for r in results], axis=0)
    delta_neff_mat = np.stack([r.delta_neff[:H] for r in results], axis=0)
    delta_loc_mat = np.stack([r.delta_loc[:H] for r in results], axis=0)          # [S,H,19]

    # mean / stderr
    def mean_stderr(x):
        m = x.mean(axis=0)
        se = x.std(axis=0, ddof=1) / np.sqrt(x.shape[0])
        return m, se

    mean_end, se_end = mean_stderr(delta_end_mat)
    mean_ent, se_ent = mean_stderr(delta_ent_mat)
    mean_neff, se_neff = mean_stderr(delta_neff_mat)
    mean_loc = delta_loc_mat.mean(axis=0)  # [H,19]

    # save aggregate CSV
    df_ag = pd.DataFrame({
        "h": np.arange(H),
        "mean_delta_P_end": mean_end,
        "se_delta_P_end": se_end,
        "mean_delta_entropy_token": mean_ent,
        "se_delta_entropy_token": se_ent,
        "mean_delta_neff_token": mean_neff,
        "se_delta_neff_token": se_neff,
    })
    df_ag.to_csv(os.path.join(OUT_DIR, "aggregate_series.csv"), index=False)

    df_loc = pd.DataFrame(mean_loc, columns=[f"node_{i:02d}" for i in range(BASE_N)])
    df_loc.insert(0, "h", np.arange(H))
    df_loc.to_csv(os.path.join(OUT_DIR, "aggregate_delta_loc_mean.csv"), index=False)

    # aggregate plots
    fig = plt.figure(figsize=(12, 4))
    plt.imshow(mean_loc.T, aspect="auto")
    plt.colorbar(label="Mean ΔP(location) = P_on - P_off")
    plt.yticks(np.arange(BASE_N), [f"{i:02d}" for i in range(BASE_N)])
    plt.xlabel("h")
    plt.ylabel("node")
    plt.title(f"Aggregate mean Δ Location Prob (N={len(results)})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "aggregate_delta_loc_mean.png"))
    plt.close(fig)

    fig = plt.figure(figsize=(10, 3))
    plt.plot(mean_end, label="mean ΔP(end)")
    plt.fill_between(np.arange(H), mean_end-se_end, mean_end+se_end, alpha=0.2)
    plt.plot(mean_ent, label="mean ΔEntropy(token)")
    plt.fill_between(np.arange(H), mean_ent-se_ent, mean_ent+se_ent, alpha=0.2)
    plt.plot(mean_neff, label="mean ΔNeff(token)")
    plt.fill_between(np.arange(H), mean_neff-se_neff, mean_neff+se_neff, alpha=0.2)
    plt.axhline(0, linewidth=1)
    plt.legend()
    plt.xlabel("h")
    plt.title(f"Aggregate deltas (teacher-forcing, N={len(results)})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "aggregate_series.png"))
    plt.close(fig)

    print(f"Done. analyzed {len(results)} samples.")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()