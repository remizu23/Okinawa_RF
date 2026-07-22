#!/usr/bin/env python3
"""Koopman / linear latent dynamics: modal excitation maps.

固定 prefix から得た z_0 を固有空間で操作し、将来各時刻の
move / stay の score・probability を地図上に可視化する。

使い方:
  1. 下記 CONFIG / SCENARIOS を編集（MODE_JOBS は手動指定時のみ）
  2. python plot_modal_maps.py
  → checkpoint から M2 なら A0+A1、M0/M1 なら shared A を自動選択
  → auto_mode_jobs=True なら |λ|≥閾値の全モードを自動ジョブ化
  → 全 SCENARIOS × 全 operator × 全 MODE_JOBS を実行し、
    out_dir/{scenario_name}/{operator}/ に図と meta JSON を保存

出力図（各 1 枚の 3行×T列マルチパネル）:
  - real_mode_probability / real_mode_score
  - complex_amplitude_probability / complex_amplitude_score
  - complex_phase_probability / complex_phase_score
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
import torch

from network import Network, expand_adjacency_matrix
from tokenization import Tokenization
from DKP_RF import KoopmanRoutesFormer
from context_modes import ContextAblationConfig, DEFAULT_CONTEXT_CONFIG

# ===========================================================================
#  Config（ここを編集して実行）
# ===========================================================================

CONFIG: dict[str, Any] = {
    # 必須
    "checkpoint": (
        "/home/mizutani/projects/RF/runs/2607_condition_ablation/cond_ablation_20260721_002552/007_timezone_M2_p5_z32/train/model_weights_20260721_015751.pth"
    ),
    "out_dir": (
        "/home/mizutani/projects/RF/runs/2607_condition_ablation/cond_ablation_20260721_002552/007_timezone_M2_p5_z32/modal_maps"
    ),

    # データパス
    "adj_path": "/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt",
    "loc_csv": "/home/mizutani/projects/RF/data/ble_location_v2.csv",
    # 実モードで α=None（σベース）のときだけ使用
    "data_npz": "/home/mizutani/projects/RF/data/input_real_m5.npz",
    "split_npz": "/home/mizutani/projects/RF/data/common_split_indices_m5.npz",

    # 共通ロールアウト設定
    "horizon": 5,
    "output_type": "both",  # "score" | "probability" | "both"

    # モードジョブ自動生成（True なら MODE_JOBS は使わない）
    "auto_mode_jobs": True,
    "eig_abs_min": 0.3,   # |λ| >= この閾値のモードを可視化
    # auto_mode_jobs=True 時のデフォルト励起量（real の alpha=None なら alpha_sigma*σ_i）
    "auto_alpha": None,            # 例: 1.0 で直接指定
    "auto_gamma": 1.5,
    "auto_delta": float(np.pi / 2),

    # 実モード: alpha=None のとき α = alpha_sigma * σ_i
    "alpha_sigma": 1.0,
    "sigma_split": "train",
    "sigma_max_samples": 256,

    # 動作フラグ
    "list_modes": False,   # True にすると、図は作らず 固有値一覧だけ表示して終了。mode_index を決める前の確認用。
    "no_basemap": False,  # False なら contextily で OSM 背景を試す（失敗時フォールバック）
    "cpu": False,  # True なら 強制 CPU。False なら CUDA が使えれば GPU。
    "strict": False,  # False（デフォルト）: ある scenario×job が失敗してもログを出して 次のジョブを続行。True: 1件でも失敗したら 例外で即停止。
}

# scen2 と同様: 複数 prefix を列挙（コメントアウトで切替）
SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "6,6,25,14,33",
        "prefix": [6, 6, 25, 14, 33],
        "time": 20240101,
        "agent_id": 0,
        "holiday": 1,
        "time_zone": 0,
    },
    {
        "name": "16,35,35,14,33",
        "prefix": [16, 35, 35, 14, 33],
        "time": 20240101,
        "agent_id": 0,
        "holiday": 1,
        "time_zone": 0,
    },
    # {
    #     "name": "0,1,2,21,21",
    #     "prefix": [0, 1, 2, 21, 21],
    #     "time": 20240101,
    #     "agent_id": 0,
    #     "holiday": 1,
    #     "time_zone": 0,
    # },
]

# auto_mode_jobs=False のときのみ使用（手動指定）
MODE_JOBS: list[dict[str, Any]] = [
    {
        "name": "real_m0",
        "mode_type": "real",
        "mode_index": 0,
        "alpha": 1.0,
    },
    {
        "name": "amp_m3",
        "mode_type": "complex_amplitude",
        "mode_index": 3,
        "gamma": 1.5,
    },
    {
        "name": "phase_m3",
        "mode_type": "complex_phase",
        "mode_index": 3,
        "delta": float(np.pi / 2),
    },
]

# 地図から除外するノード（ble_loc_id / トークン node id）
EXCLUDED_NODE_IDS: list[int] = [7]

DEFAULT_GAMMA = 1.5
DEFAULT_DELTA = float(np.pi / 2)
DEFAULT_ALPHA_SIGMA = 1.0
DEFAULT_SIGMA_MAX_SAMPLES = 256
EIG_REAL_TOL = 1e-8
CONJ_TOL = 1e-6


# ===========================================================================
# Softmax / token helpers
# ===========================================================================

def softmax_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum()
    return e / s if s != 0 else np.full_like(e, 1.0 / len(e))


def extract_move_stay_values(
    values_1d: np.ndarray,
    *,
    num_nodes: int = 19,
    stay_offset: int = 19,
) -> dict[str, np.ndarray]:
    """語彙ベクトルから move/stay 成分を取り出す（special は捨てる）。"""
    v = np.asarray(values_1d, dtype=float).reshape(-1)
    move = v[:num_nodes].copy()
    stay = v[stay_offset : stay_offset + num_nodes].copy()
    return {"move": move, "stay": stay}


# ===========================================================================
# Load model / data / geo
# ===========================================================================

def _pick_cfg(c: dict, *keys, default=None):
    for k in keys:
        if k in c:
            return c[k]
    return default


def _infer_model_hparams(ckpt_config: dict, state_dict: dict, base_N: int) -> dict:
    c = ckpt_config if isinstance(ckpt_config, dict) else {}

    if "A" in state_dict:
        inferred_z_dim = int(state_dict["A"].shape[0])
    elif "A0" in state_dict:
        inferred_z_dim = int(state_dict["A0"].shape[0])
    else:
        raise KeyError("state_dict に A / A0 がありません")

    inferred_vocab_size = int(state_dict["token_embedding.weight"].shape[0])
    inferred_token_emb_dim = int(state_dict["token_embedding.weight"].shape[1])
    inferred_d_model = int(state_dict["input_proj.weight"].shape[0])
    inferred_num_agents = int(state_dict["agent_embedding.weight"].shape[0])
    inferred_agent_emb_dim = int(state_dict["agent_embedding.weight"].shape[1])
    inferred_max_stay_count = int(state_dict["stay_embedding.weight"].shape[0] - 1)
    inferred_stay_emb_dim = int(state_dict["stay_embedding.weight"].shape[1])
    inferred_holiday_emb_dim = int(state_dict["holiday_embedding.weight"].shape[1])
    inferred_time_zone_emb_dim = int(state_dict["time_zone_embedding.weight"].shape[1])
    inferred_event_emb_dim = int(state_dict["event_embedding.weight"].shape[1])

    if "transformer_block.layers.0.linear1.weight" in state_dict:
        inferred_d_ff = int(state_dict["transformer_block.layers.0.linear1.weight"].shape[0])
    else:
        inferred_d_ff = int(_pick_cfg(c, "d_ff", default=4 * inferred_d_model))

    transformer_layer_ids = set()
    for k in state_dict.keys():
        if k.startswith("transformer_block.layers."):
            parts = k.split(".")
            if len(parts) > 3 and parts[3].isdigit():
                transformer_layer_ids.add(int(parts[3]))
    inferred_num_layers = (max(transformer_layer_ids) + 1) if transformer_layer_ids else int(
        _pick_cfg(c, "num_layers", default=3)
    )

    nhead_from_cfg = _pick_cfg(c, "nhead", "head_num", default=None)
    if nhead_from_cfg is None:
        for cand in (8, 4, 2, 1):
            if inferred_d_model % cand == 0:
                nhead_from_cfg = cand
                break
    resolved_nhead = int(nhead_from_cfg)

    return {
        "vocab_size": int(_pick_cfg(c, "vocab_size", default=inferred_vocab_size)),
        "token_emb_dim": int(_pick_cfg(c, "token_emb_dim", "d_ie", default=inferred_token_emb_dim)),
        "d_model": int(_pick_cfg(c, "d_model", "d_ie", default=inferred_d_model)),
        "nhead": resolved_nhead,
        "num_layers": int(_pick_cfg(c, "num_layers", "B_de", default=inferred_num_layers)),
        "d_ff": int(_pick_cfg(c, "d_ff", default=inferred_d_ff)),
        "z_dim": int(_pick_cfg(c, "z_dim", default=inferred_z_dim)),
        "pad_token_id": int(_pick_cfg(c, "pad_token_id", default=38)),
        "base_N": int(_pick_cfg(c, "base_N", default=base_N)),
        "num_agents": int(_pick_cfg(c, "num_agents", default=inferred_num_agents)),
        "agent_emb_dim": int(_pick_cfg(c, "agent_emb_dim", default=inferred_agent_emb_dim)),
        "max_stay_count": int(_pick_cfg(c, "max_stay_count", default=inferred_max_stay_count)),
        "stay_emb_dim": int(_pick_cfg(c, "stay_emb_dim", default=inferred_stay_emb_dim)),
        "holiday_emb_dim": int(_pick_cfg(c, "holiday_emb_dim", default=inferred_holiday_emb_dim)),
        "time_zone_emb_dim": int(_pick_cfg(c, "time_zone_emb_dim", default=inferred_time_zone_emb_dim)),
        "event_emb_dim": int(_pick_cfg(c, "event_emb_dim", default=inferred_event_emb_dim)),
    }


def setup_network(adj_path: str | Path):
    adj_matrix = torch.load(adj_path, weights_only=True)
    if adj_matrix.shape[0] == 38:
        base_N = 19
        base_for_expand = adj_matrix[:base_N, :base_N]
    else:
        base_N = int(adj_matrix.shape[0])
        base_for_expand = adj_matrix

    expanded_adj = expand_adjacency_matrix(base_for_expand)
    dummy_feat = torch.zeros((base_N, 1))
    node_features = torch.cat([dummy_feat, dummy_feat], dim=0)
    network = Network(expanded_adj, node_features)
    tokenizer = Tokenization(network)
    return network, tokenizer, base_for_expand.detach().cpu().numpy(), base_N


def load_model_and_data(
    checkpoint_path: str | Path,
    adj_path: str | Path,
    device: torch.device,
):
    """モデル・tokenizer・隣接行列を読み込む。"""
    network, tokenizer, base_adj_np, base_N = setup_network(adj_path)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    c = ckpt.get("config", {}) or {}
    h = _infer_model_hparams(c, state_dict, base_N)

    context_ablation = ContextAblationConfig.from_dict(c.get("context_ablation"))
    stay_u_threshold = int(c.get("stay_u_threshold", 3))
    max_prefix_len = int(c.get("max_prefix_len", c.get("fixed_prefix_length", 5)))
    encoder_type = str(c.get("encoder_type", "transformer"))
    use_aux_loss = bool(c.get("use_aux_loss", False))

    model = KoopmanRoutesFormer(
        vocab_size=h["vocab_size"],
        token_emb_dim=h["token_emb_dim"],
        d_model=h["d_model"],
        nhead=h["nhead"],
        num_layers=h["num_layers"],
        d_ff=h["d_ff"],
        z_dim=h["z_dim"],
        pad_token_id=h["pad_token_id"],
        base_N=h["base_N"],
        num_agents=h["num_agents"],
        agent_emb_dim=h["agent_emb_dim"],
        max_stay_count=h["max_stay_count"],
        stay_emb_dim=h["stay_emb_dim"],
        holiday_emb_dim=h["holiday_emb_dim"],
        time_zone_emb_dim=h["time_zone_emb_dim"],
        event_emb_dim=h["event_emb_dim"],
        use_aux_loss=use_aux_loss,
        encoder_type=encoder_type,
        max_prefix_len=max_prefix_len,
        context_ablation=context_ablation,
        stay_u_threshold=stay_u_threshold,
        context_config=DEFAULT_CONTEXT_CONFIG,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    meta = {
        "hparams": h,
        "config": c,
        "context_ablation": context_ablation.to_dict(),
        "encoder_type": encoder_type,
        "max_prefix_len": max_prefix_len,
        "checkpoint_path": str(checkpoint_path),
    }
    return model, tokenizer, base_adj_np, base_N, meta


def load_node_coordinates(
    loc_csv: str | Path,
    num_nodes: int = 19,
    excluded_node_ids: list[int] | None = None,
) -> dict[int, tuple[float, float]]:
    """ble_loc_id -> (lon, lat). ノード 0..num_nodes-1（除外ノード除く）。"""
    excluded = set(excluded_node_ids if excluded_node_ids is not None else EXCLUDED_NODE_IDS)
    df = pd.read_csv(loc_csv)
    coords: dict[int, tuple[float, float]] = {}
    for _, row in df.iterrows():
        if pd.isna(row.get("ble_loc_id")):
            continue
        nid = int(row["ble_loc_id"])
        if 0 <= nid < num_nodes and nid not in excluded:
            coords[nid] = (float(row["lon"]), float(row["lat"]))
    missing = [
        i for i in range(num_nodes)
        if i not in excluded and i not in coords
    ]
    if missing:
        raise ValueError(f"座標が見つからないノード: {missing} ({loc_csv})")
    return coords


def get_active_node_ids(
    num_nodes: int,
    excluded_node_ids: list[int] | None = None,
) -> list[int]:
    excluded = set(excluded_node_ids if excluded_node_ids is not None else EXCLUDED_NODE_IDS)
    return [i for i in range(num_nodes) if i not in excluded]


def load_split_route_indices(
    split: str,
    split_npz: str | Path,
) -> np.ndarray:
    data = np.load(split_npz, allow_pickle=True)
    alts = {
        "train": ["train_sequences", "train_idx", "train_indices", "train"],
        "val": ["val_sequences", "val_idx", "val_indices", "val"],
        "test": ["test_sequences", "test_idx", "test_indices", "test"],
    }[split.lower()]
    for a in alts:
        if a in data:
            return np.asarray(data[a])
    raise KeyError(f"split npz に {split} 用キーがありません: {list(data.keys())}")


def _route_real_len(route_1d: np.ndarray, pad_token_id: int = 38) -> int:
    pad_idx = np.where(np.asarray(route_1d) == pad_token_id)[0]
    return int(pad_idx[0]) if len(pad_idx) else int(len(route_1d))


def scenario_to_sample(scenario: dict[str, Any]) -> dict[str, Any]:
    """SCENARIOS 要素から encode 用サンプル辞書を作る。"""
    if "prefix" not in scenario:
        raise KeyError(f"scenario に prefix がありません: {scenario}")
    tokens = [int(t) for t in scenario["prefix"]]
    if not tokens:
        raise ValueError(f"prefix が空です: {scenario.get('name')}")
    hol = int(scenario.get("holiday", 0))
    tz = int(scenario.get("time_zone", 0))
    return {
        "source": "scenario",
        "name": str(scenario.get("name") or ",".join(str(t) for t in tokens)),
        "tokens": tokens,
        "time": int(scenario.get("time", 20240101)),
        "agent_id": int(scenario.get("agent_id", 0)),
        "holidays": [hol] * len(tokens),
        "timezones": [tz] * len(tokens),
        "events": [0] * len(tokens),
        "prefix_length": len(tokens),
    }


# ===========================================================================
# Encode / operator / eigen / perturb / rollout
# ===========================================================================

def encode_prefix_to_z0(
    model: KoopmanRoutesFormer,
    tokenizer: Tokenization,
    sample: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    tokens_list = sample["tokens"]
    seq_len = len(tokens_list)
    tokens = torch.tensor([tokens_list], dtype=torch.long, device=device)
    stay_counts = tokenizer.calculate_stay_counts(tokens).to(device)
    agent_ids = torch.tensor([sample["agent_id"]], dtype=torch.long, device=device)
    holidays = torch.tensor([sample["holidays"]], dtype=torch.long, device=device)
    time_zones = torch.tensor([sample["timezones"]], dtype=torch.long, device=device)
    events = torch.tensor([sample["events"]], dtype=torch.long, device=device)

    with torch.no_grad():
        z0, _ = model.encode_prefix(
            tokens, stay_counts, agent_ids, holidays, time_zones, events, prefix_mask=None
        )
    return z0.detach().cpu().numpy()[0].astype(np.float64)


def resolve_operator_variants(model: KoopmanRoutesFormer) -> list[tuple[str, np.ndarray]]:
    """checkpoint から実行すべき作用素を返す。

    - M2 (dual-A): A0 と A1 の両方
    - M0/M1 (単一 A): shared のみ
    """
    if bool(getattr(model, "uses_dual_A", False)):
        a0 = model.A0.detach().cpu().numpy().astype(np.float64)
        da = model.delta_A.detach().cpu().numpy().astype(np.float64)
        return [("a0", a0.copy()), ("a1", (a0 + da).copy())]
    return [("shared", model.A.detach().cpu().numpy().astype(np.float64).copy())]


def get_operator_matrix(
    model: KoopmanRoutesFormer,
    operator_variant: str,
    condition_value: float | None = None,
) -> np.ndarray:
    """作用素行列を返す（resolve_operator_variants の tag 指定用）。"""
    variant = operator_variant.lower()
    for tag, a in resolve_operator_variants(model):
        if tag == variant:
            return a.copy()
    uses_dual = bool(getattr(model, "uses_dual_A", False))
    if variant == "u" and uses_dual:
        if condition_value is None:
            raise ValueError("operator_variant=u には condition_value が必要です")
        a0 = model.A0.detach().cpu().numpy().astype(np.float64)
        da = model.delta_A.detach().cpu().numpy().astype(np.float64)
        return (a0 + float(condition_value) * da).copy()
    raise ValueError(f"未知の operator_variant: {operator_variant}")


def eigendecompose_operator(A: np.ndarray) -> dict[str, np.ndarray]:
    """A = V Λ V^{-1}。|λ| 降順ソート。"""
    A = np.asarray(A, dtype=np.float64)
    eigvals, eigvecs = scipy.linalg.eig(A)
    sort_idx = np.argsort(np.abs(eigvals))[::-1]
    eigvals = eigvals[sort_idx]
    V = eigvecs[:, sort_idx]
    V_inv = scipy.linalg.inv(V)
    return {"eigvals": eigvals, "V": V, "V_inv": V_inv, "A": A}


def find_conjugate_index(eigvals: np.ndarray, mode_idx: int, tol: float = CONJ_TOL) -> int:
    target = np.conj(eigvals[mode_idx])
    for j, ev in enumerate(eigvals):
        if j == mode_idx:
            continue
        if abs(ev - target) < tol:
            return int(j)
    raise ValueError(
        f"mode_idx={mode_idx} (λ={eigvals[mode_idx]}) の共役ペアが見つかりません"
    )


def is_real_eigenvalue(lam: complex, tol: float = EIG_REAL_TOL) -> bool:
    return abs(lam.imag) < tol


def _complex_representative_index(eigvals: np.ndarray, mode_idx: int) -> int:
    """複素共役ペアの代表 index（Im >= 0 を優先、同符号なら小さい方）。"""
    conj_idx = find_conjugate_index(eigvals, mode_idx)
    if eigvals[mode_idx].imag >= 0 and eigvals[conj_idx].imag <= 0:
        return int(mode_idx)
    if eigvals[conj_idx].imag >= 0 and eigvals[mode_idx].imag <= 0:
        return int(conj_idx)
    return int(min(mode_idx, conj_idx))


def build_mode_jobs_from_eigenvalues(
    eigvals: np.ndarray,
    *,
    eig_abs_min: float,
    alpha: float | None = None,
    gamma: float = DEFAULT_GAMMA,
    delta: float = DEFAULT_DELTA,
) -> list[dict[str, Any]]:
    """|λ| >= eig_abs_min のモードについて自動 MODE_JOBS を生成。

    - 実固有値: real を1件
    - 複素共役ペア: complex_amplitude + complex_phase（ペアは1回だけ）
    """
    jobs: list[dict[str, Any]] = []
    used: set[int] = set()
    threshold = float(eig_abs_min)

    for i, lam in enumerate(eigvals):
        if i in used:
            continue
        if abs(lam) < threshold:
            # |λ| 降順ソート済みなので以降はすべて閾値未満
            break

        if is_real_eigenvalue(lam):
            used.add(i)
            jobs.append({
                "name": f"auto_real_m{i}",
                "mode_type": "real",
                "mode_index": int(i),
                "alpha": alpha,
            })
            continue

        conj_idx = find_conjugate_index(eigvals, i)
        rep = _complex_representative_index(eigvals, i)
        used.add(i)
        used.add(conj_idx)
        jobs.append({
            "name": f"auto_amp_m{rep}",
            "mode_type": "complex_amplitude",
            "mode_index": int(rep),
            "gamma": float(gamma),
        })
        jobs.append({
            "name": f"auto_phase_m{rep}",
            "mode_type": "complex_phase",
            "mode_index": int(rep),
            "delta": float(delta),
        })

    return jobs


def print_eigenvalue_table(eigvals: np.ndarray, *, operator_tag: str) -> None:
    print(f"\n--- Eigenvalues ({operator_tag}) ---")
    print("mode_idx | kind    | λ")
    for i, lam in enumerate(eigvals):
        kind = "real" if is_real_eigenvalue(lam) else "complex"
        print(f"{i:7d} | {kind:7s} | {lam.real:.6f}{lam.imag:+.6f}j  |λ|={abs(lam):.6f}")


def _to_real_z(z_complex: np.ndarray, tag: str = "z0'") -> np.ndarray:
    imag_max = float(np.max(np.abs(np.asarray(z_complex).imag)))
    if imag_max > 1e-6:
        print(
            f"[WARN] {tag} に数値誤差由来の虚部 max|Im|={imag_max:.3e} が残ったため実部を採用します"
        )
    return np.asarray(z_complex.real, dtype=np.float64)


def perturb_real_mode(
    z0: np.ndarray,
    V: np.ndarray,
    V_inv: np.ndarray,
    mode_idx: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_tilde = V_inv @ z0.astype(np.complex128)
    z_tilde_p = z_tilde.copy()
    z_tilde_p[mode_idx] = z_tilde_p[mode_idx] + alpha
    z0_p = _to_real_z(V @ z_tilde_p)
    return z0_p, z_tilde, z_tilde_p


def perturb_complex_amplitude(
    z0: np.ndarray,
    V: np.ndarray,
    V_inv: np.ndarray,
    mode_idx: int,
    conj_idx: int,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_tilde = V_inv @ z0.astype(np.complex128)
    z_tilde_p = z_tilde.copy()
    c = z_tilde[mode_idx]
    z_tilde_p[mode_idx] = gamma * c
    z_tilde_p[conj_idx] = gamma * np.conj(c)
    # 共役整合性を強制（数値誤差対策）
    z_tilde_p[conj_idx] = np.conj(z_tilde_p[mode_idx])
    z0_p = _to_real_z(V @ z_tilde_p)
    return z0_p, z_tilde, z_tilde_p


def perturb_complex_phase(
    z0: np.ndarray,
    V: np.ndarray,
    V_inv: np.ndarray,
    mode_idx: int,
    conj_idx: int,
    delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_tilde = V_inv @ z0.astype(np.complex128)
    z_tilde_p = z_tilde.copy()
    c = z_tilde[mode_idx]
    phase = np.exp(1j * float(delta))
    z_tilde_p[mode_idx] = c * phase
    z_tilde_p[conj_idx] = np.conj(c) * np.conj(phase)
    z_tilde_p[conj_idx] = np.conj(z_tilde_p[mode_idx])
    z0_p = _to_real_z(V @ z_tilde_p)
    return z0_p, z_tilde, z_tilde_p


def rollout_scores_and_probabilities(
    z0: np.ndarray,
    model: KoopmanRoutesFormer,
    A: np.ndarray,
    horizon: int,
) -> dict[str, np.ndarray]:
    """z_1..z_T について score / probability を返す。

    モデル実装に合わせ z_{t+1} = A @ z_t （列ベクトル表記）。
    """
    W = model.to_logits.weight.detach().cpu().numpy().astype(np.float64)
    b = model.to_logits.bias.detach().cpu().numpy().astype(np.float64)
    A = np.asarray(A, dtype=np.float64)
    z = np.asarray(z0, dtype=np.float64).copy()

    scores = []
    probs = []
    zs = []
    for _ in range(horizon):
        z = A @ z
        zs.append(z.copy())
        logits = W @ z + b
        scores.append(logits)
        probs.append(softmax_np(logits))

    return {
        "scores": np.stack(scores, axis=0),  # [T, V]
        "probs": np.stack(probs, axis=0),
        "z_traj": np.stack(zs, axis=0),  # [T, z_dim] = z_1..z_T
    }


def estimate_mode_sigma(
    model: KoopmanRoutesFormer,
    tokenizer: Tokenization,
    V_inv: np.ndarray,
    mode_idx: int,
    *,
    split: str,
    data_npz: str | Path,
    split_npz: str | Path,
    prefix_length: int,
    device: torch.device,
    pad_token_id: int = 38,
    max_samples: int = DEFAULT_SIGMA_MAX_SAMPLES,
) -> float:
    """指定 split 上の z̃_{0, mode_idx} の標準偏差（簡易: 先頭 max_samples 件）。"""
    trip = np.load(data_npz)
    route_arr = trip["route_arr"]
    time_arr = trip["time_arr"]
    agent_arr = trip["agent_ids"] if "agent_ids" in trip else np.zeros(len(route_arr), dtype=int)
    holiday_arr = trip["holiday_arr"]
    timezone_arr = trip["time_zone_arr"]
    event_arr = trip["event_arr"]
    route_indices = load_split_route_indices(split, split_npz)

    coeffs = []
    n_use = min(int(max_samples), len(route_indices))
    for si in range(n_use):
        route_idx = int(route_indices[si])
        r = route_arr[route_idx]
        real_len = _route_real_len(r, pad_token_id)
        if real_len < prefix_length:
            continue
        sample = {
            "tokens": [int(x) for x in r[:prefix_length]],
            "time": int(time_arr[route_idx]),
            "agent_id": int(agent_arr[route_idx]),
            "holidays": [int(x) for x in holiday_arr[route_idx, :prefix_length]],
            "timezones": [int(x) for x in timezone_arr[route_idx, :prefix_length]],
            "events": [int(x) for x in event_arr[route_idx, :prefix_length]],
        }
        z0 = encode_prefix_to_z0(model, tokenizer, sample, device)
        z_tilde = V_inv @ z0.astype(np.complex128)
        coeffs.append(z_tilde[mode_idx])

    if len(coeffs) < 2:
        print("[WARN] σ 推定サンプルが不足。alpha=1.0 をそのまま使います")
        return 1.0

    coeffs = np.asarray(coeffs)
    # 実モード想定: 実部の std。複素でも |c| の std は使わず、実モード用。
    sigma = float(np.std(coeffs.real))
    if not np.isfinite(sigma) or sigma < 1e-12:
        sigma = 1.0
        print("[WARN] σ がほぼ 0 のため 1.0 にフォールバック")
    return sigma


# ===========================================================================
# Plotting
# ===========================================================================

_BASEMAP_WARNED = False


def _set_map_axis_limits(
    ax,
    coords: dict[int, tuple[float, float]],
    *,
    pad_frac: float = 0.10,
) -> tuple[float, float, float, float]:
    """coords から表示範囲を設定し (xmin, xmax, ymin, ymax) を返す。"""
    lons = np.array([coords[i][0] for i in coords])
    lats = np.array([coords[i][1] for i in coords])
    lon_min, lon_max = float(lons.min()), float(lons.max())
    lat_min, lat_max = float(lats.min()), float(lats.max())
    lon_span = max(lon_max - lon_min, 1e-6)
    lat_span = max(lat_max - lat_min, 1e-6)
    pad_lon = max(lon_span * pad_frac, 0.00025)
    pad_lat = max(lat_span * pad_frac, 0.00025)
    xmin, xmax = lon_min - pad_lon, lon_max + pad_lon
    ymin, ymax = lat_min - pad_lat, lat_max + pad_lat
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    return xmin, xmax, ymin, ymax


def _style_basemap(ax, *, alpha: float = 0.28) -> None:
    """OSM タイルをモノクロ・半透明にしてマーカーが目立つようにする。"""
    for im in ax.get_images():
        data = im.get_array()
        if data is None:
            im.set_alpha(alpha)
            continue
        arr = np.asarray(data, dtype=float)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            im.set_alpha(alpha)
            continue
        rgb = arr[..., :3]
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        gray = np.clip(gray, 0.0, 1.0)
        rgba = np.zeros((*gray.shape, 4), dtype=float)
        rgba[..., 0] = gray
        rgba[..., 1] = gray
        rgba[..., 2] = gray
        rgba[..., 3] = alpha
        im.set_data(rgba)
        im.set_alpha(1.0)
        im.set_zorder(0)


def _try_add_basemap(ax) -> bool:
    """軸範囲設定済み ax に OSM タイルを重ねる（モノクロ・薄め）。"""
    global _BASEMAP_WARNED
    try:
        import contextily as ctx  # type: ignore
        # Positron はもともと淡いが、さらにモノクロ化してマーカーを強調
        source = ctx.providers.CartoDB.Positron
        ctx.add_basemap(
            ax,
            crs="EPSG:4326",
            source=source,
            attribution=False,
            zoom="auto",
        )
        _style_basemap(ax, alpha=0.70)
        return True
    except Exception:
        try:
            import contextily as ctx  # type: ignore
            ctx.add_basemap(
                ax,
                crs="EPSG:4326",
                source=ctx.providers.OpenStreetMap.Mapnik,
                attribution=False,
                zoom="auto",
            )
            _style_basemap(ax, alpha=0.70)
            return True
        except Exception as e:
            ax.set_facecolor("#f5f5f0")
            if not _BASEMAP_WARNED:
                print(f"[INFO] OSM 背景をスキップ（以降も同様）: {e}")
                _BASEMAP_WARNED = True
            return False


def _draw_network(
    ax,
    coords: dict[int, tuple[float, float]],
    adj: np.ndarray,
    active_node_ids: list[int],
):
    active = set(active_node_ids)
    for i in active_node_ids:
        for j in range(i + 1, adj.shape[0]):
            if j not in active:
                continue
            if adj[i, j] > 0 or adj[j, i] > 0:
                xi, yi = coords[i]
                xj, yj = coords[j]
                ax.plot([xi, xj], [yi, yj], color="#888888", lw=0.7, zorder=1, alpha=0.55)


def plot_single_map_panel(
    ax,
    coords: dict[int, tuple[float, float]],
    adj: np.ndarray,
    move_vals: np.ndarray,
    stay_vals: np.ndarray,
    *,
    active_node_ids: list[int],
    cmap,
    vmin: float,
    vmax: float,
    title: str,
    show_basemap: bool,
    marker_size: float = 32.0,
    is_diverging: bool = False,
    size_by_abs: bool = False,
    draw_labels: bool = True,
    lon_offset: float = 0.0,
):
    node_ids = [i for i in active_node_ids if i in coords]
    lons = np.array([coords[i][0] for i in node_ids], dtype=float)
    lats = np.array([coords[i][1] for i in node_ids], dtype=float)
    move_v = np.array([move_vals[i] for i in node_ids], dtype=float)
    stay_v = np.array([stay_vals[i] for i in node_ids], dtype=float)

    if lon_offset <= 0:
        lon_span = float(lons.max() - lons.min()) if len(lons) else 0.001
        lon_offset = max(lon_span * 0.022, 8e-6)

    lat_span = float(lats.max() - lats.min()) if len(lats) else 0.001
    label_lat_offset = max(lat_span * 0.018, 6e-5)

    _set_map_axis_limits(ax, coords)

    if show_basemap:
        ok = _try_add_basemap(ax)
        if not ok:
            ax.set_facecolor("#f5f5f0")
    else:
        ax.set_facecolor("#f5f5f0")

    _draw_network(ax, coords, adj, node_ids)

    move_x = lons - lon_offset
    stay_x = lons + lon_offset

    if size_by_abs:
        abs_all = np.concatenate([np.abs(move_v), np.abs(stay_v)])
        scale = max(float(abs_all.max()), 1e-12)
        m_sizes = 12.0 + 48.0 * (np.abs(move_v) / scale)
        s_sizes = 12.0 + 48.0 * (np.abs(stay_v) / scale)
    else:
        m_sizes = np.full(len(node_ids), marker_size, dtype=float)
        s_sizes = np.full(len(node_ids), marker_size, dtype=float)

    ax.scatter(
        move_x, lats, c=move_v, s=m_sizes, marker="o",
        cmap=cmap, vmin=vmin, vmax=vmax, edgecolors="k", linewidths=0.35, zorder=3,
    )
    ax.scatter(
        stay_x, lats, c=stay_v, s=s_sizes, marker="D",
        cmap=cmap, vmin=vmin, vmax=vmax, edgecolors="k", linewidths=0.35, zorder=3,
    )

    if draw_labels:
        for nid, x, y in zip(node_ids, lons, lats):
            ax.text(
                x, y + label_lat_offset,
                str(nid),
                fontsize=4.5,
                ha="center",
                va="bottom",
                color="#1a1a1a",
                zorder=5,
                bbox=dict(
                    boxstyle="round,pad=0.12",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.82,
                ),
            )

    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")


def plot_modal_map_grid(
    *,
    baseline_by_t: list[dict[str, np.ndarray]],
    perturbed_by_t: list[dict[str, np.ndarray]],
    coords: dict[int, tuple[float, float]],
    adj: np.ndarray,
    active_node_ids: list[int],
    value_kind: str,  # "probability" | "score"
    mode_type: str,
    title_meta: str,
    out_path: str | Path,
    show_basemap: bool = True,
) -> str:
    """3行×T列のマルチパネル図を保存。

    色スケール:
      - baseline 行: 行内・全 t 共有（行独立）
      - perturbed 行: 行内・全 t 共有（行独立）
      - Δ 行: 0中心・全 t 共有
    """
    T = len(baseline_by_t)
    assert T == len(perturbed_by_t)

    diff_by_t = [
        {
            "move": perturbed_by_t[t]["move"] - baseline_by_t[t]["move"],
            "stay": perturbed_by_t[t]["stay"] - baseline_by_t[t]["stay"],
        }
        for t in range(T)
    ]

    def _row_abs_lim(rows: list[dict[str, np.ndarray]]) -> tuple[float, float]:
        vals = np.concatenate([np.concatenate([r["move"], r["stay"]]) for r in rows])
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1e-6
        if value_kind == "probability":
            vmin = max(0.0, vmin)
        return vmin, vmax

    def _row_diff_lim(rows: list[dict[str, np.ndarray]]) -> float:
        vals = np.concatenate([np.concatenate([r["move"], r["stay"]]) for r in rows])
        return max(float(np.max(np.abs(vals))), 1e-12)

    base_vmin, base_vmax = _row_abs_lim(baseline_by_t)
    pert_vmin, pert_vmax = _row_abs_lim(perturbed_by_t)
    diff_v = _row_diff_lim(diff_by_t)

    seq_cmap = "YlOrRd" if value_kind == "probability" else "viridis"
    div_cmap = "RdBu_r"

    # 列方向（t 間）を詰めて横方向の遷移を追いやすくする
    col_w = 2.2
    fig, axes = plt.subplots(
        3, T,
        figsize=(col_w * T + 0.6, 8.8),
        squeeze=False,
        gridspec_kw={"wspace": 0.006, "hspace": 0.10},
    )
    fig.suptitle(title_meta, fontsize=11, y=0.98)

    row_specs = [
        (baseline_by_t, seq_cmap, base_vmin, base_vmax, False, "baseline"),
        (perturbed_by_t, seq_cmap, pert_vmin, pert_vmax, False, "perturbed"),
        (diff_by_t, div_cmap, -diff_v, diff_v, True, "Δ (pert − base)"),
    ]

    for r, (row_data, cmap, vmin, vmax, is_div, row_name) in enumerate(row_specs):
        # 上2行（baseline / perturbed）はマーカーを小さめ、Δ行はやや大きめ
        row_marker_size = 22.0 if not is_div else 34.0
        for t in range(T):
            ax = axes[r][t]
            plot_single_map_panel(
                ax,
                coords,
                adj,
                row_data[t]["move"],
                row_data[t]["stay"],
                active_node_ids=active_node_ids,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                title=f"t={t + 1}" if r == 0 else "",
                show_basemap=show_basemap,
                marker_size=row_marker_size,
                is_diverging=is_div,
                size_by_abs=is_div,
                draw_labels=True,
            )
        axes[r][0].set_ylabel(row_name, fontsize=10)

    for r, (cmap, vmin, vmax, label) in enumerate([
        (seq_cmap, base_vmin, base_vmax, f"baseline {value_kind}"),
        (seq_cmap, pert_vmin, pert_vmax, f"perturbed {value_kind}"),
        (div_cmap, -diff_v, diff_v, f"Δ{value_kind}"),
    ]):
        cax = fig.add_axes([0.92, 0.68 - r * 0.30, 0.015, 0.22])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label(label, fontsize=8)
        cb.ax.tick_params(labelsize=7)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markeredgecolor="k", markersize=8, label="move"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="gray",
                   markeredgecolor="k", markersize=7, label="stay"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=9, frameon=True)

    plt.subplots_adjust(left=0.035, right=0.87, bottom=0.05, top=0.945, wspace=0.006, hspace=0.10)
    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160, pad_inches=0.03)
    plt.close(fig)
    print(f"Saved figure: {out_path}")
    return out_path


def save_modal_map_metadata(path: str | Path, meta: dict[str, Any]) -> str:
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def _jsonable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag, "abs": abs(obj), "str": str(obj)}
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(type(obj))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=_jsonable)
    print(f"Saved metadata: {path}")
    return path


# ===========================================================================
# Orchestration
# ===========================================================================

def _prefix_tag(tokens: list[int]) -> str:
    return ",".join(str(t) for t in tokens)


def _complex_to_meta(lam: complex) -> dict[str, float | str]:
    return {
        "real": float(lam.real),
        "imag": float(lam.imag),
        "abs": float(abs(lam)),
        "str": f"{lam.real:.6f}{lam.imag:+.6f}j",
    }


def run_one_job(
    *,
    model: KoopmanRoutesFormer,
    tokenizer: Tokenization,
    base_adj: np.ndarray,
    base_N: int,
    coords: dict[int, tuple[float, float]],
    active_node_ids: list[int],
    load_meta: dict[str, Any],
    A: np.ndarray,
    eigvals: np.ndarray,
    V: np.ndarray,
    V_inv: np.ndarray,
    sample: dict[str, Any],
    mode_job: dict[str, Any],
    cfg: dict[str, Any],
    out_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """1 scenario × 1 MODE_JOB を実行して図と meta を保存。"""
    z0 = encode_prefix_to_z0(model, tokenizer, sample, device)

    mode_idx = int(mode_job["mode_index"])
    if mode_idx < 0 or mode_idx >= len(eigvals):
        raise IndexError(f"mode_index={mode_idx} が範囲外 (z_dim={len(eigvals)})")

    mode_type = str(mode_job["mode_type"])
    lam = eigvals[mode_idx]
    conj_idx = None
    alpha = None
    gamma = None
    delta = None
    sigma_i = None
    alpha_from_sigma = False

    if mode_type == "real":
        if not is_real_eigenvalue(lam):
            raise ValueError(
                f"mode_type=real ですが mode_index={mode_idx} は複素固有値です: {lam}"
            )
        if mode_job.get("alpha") is not None:
            alpha = float(mode_job["alpha"])
        else:
            alpha_from_sigma = True
            sigma_i = estimate_mode_sigma(
                model, tokenizer, V_inv, mode_idx,
                split=str(cfg.get("sigma_split", "train")),
                data_npz=cfg["data_npz"],
                split_npz=cfg["split_npz"],
                prefix_length=int(sample["prefix_length"]),
                device=device,
                pad_token_id=int(load_meta["hparams"]["pad_token_id"]),
                max_samples=int(cfg.get("sigma_max_samples", DEFAULT_SIGMA_MAX_SAMPLES)),
            )
            alpha = float(cfg.get("alpha_sigma", DEFAULT_ALPHA_SIGMA)) * sigma_i
            print(f"[INFO] α = {cfg.get('alpha_sigma', 1.0)} σ_i = {alpha:.6g} (σ_i={sigma_i:.6g})")
        z0_p, z_t, z_tp = perturb_real_mode(z0, V, V_inv, mode_idx, alpha)

    elif mode_type == "complex_amplitude":
        if is_real_eigenvalue(lam):
            raise ValueError(
                f"mode_type=complex_amplitude ですが mode_index={mode_idx} は実固有値です: {lam}"
            )
        conj_idx = find_conjugate_index(eigvals, mode_idx)
        gamma = float(mode_job.get("gamma", DEFAULT_GAMMA))
        z0_p, z_t, z_tp = perturb_complex_amplitude(z0, V, V_inv, mode_idx, conj_idx, gamma)

    elif mode_type == "complex_phase":
        if is_real_eigenvalue(lam):
            raise ValueError(
                f"mode_type=complex_phase ですが mode_index={mode_idx} は実固有値です: {lam}"
            )
        conj_idx = find_conjugate_index(eigvals, mode_idx)
        delta = float(mode_job.get("delta", DEFAULT_DELTA))
        z0_p, z_t, z_tp = perturb_complex_phase(z0, V, V_inv, mode_idx, conj_idx, delta)
    else:
        raise ValueError(f"未知の mode_type: {mode_type}")

    horizon = int(cfg["horizon"])
    base_roll = rollout_scores_and_probabilities(z0, model, A, horizon)
    pert_roll = rollout_scores_and_probabilities(z0_p, model, A, horizon)

    output_type = str(cfg.get("output_type", "both"))
    output_types = []
    if output_type in ("score", "both"):
        output_types.append("score")
    if output_type in ("probability", "both"):
        output_types.append("probability")

    prefix_tag = _prefix_tag(sample["tokens"])
    op_tag = str(cfg["operator_variant"]).lower()
    mode_tag = {
        "real": "real_mode",
        "complex_amplitude": "complex_amplitude",
        "complex_phase": "complex_phase",
    }[mode_type]
    job_name = str(mode_job.get("name", f"{mode_tag}_m{mode_idx}"))
    saved: dict[str, Any] = {"figures": {}, "metadata": None, "job_name": job_name}
    title = None

    for value_kind in output_types:
        key = "scores" if value_kind == "score" else "probs"
        base_maps = [
            extract_move_stay_values(base_roll[key][t], num_nodes=base_N, stay_offset=base_N)
            for t in range(horizon)
        ]
        pert_maps = [
            extract_move_stay_values(pert_roll[key][t], num_nodes=base_N, stay_offset=base_N)
            for t in range(horizon)
        ]

        fig_stem = f"{mode_tag}_{value_kind}_m{mode_idx}_{op_tag}_h{horizon}_{prefix_tag}"
        fig_path = out_dir / f"{fig_stem}.png"

        if mode_type == "real":
            pert_desc = f"α={alpha:.4g}"
        elif mode_type == "complex_amplitude":
            pert_desc = f"γ={gamma}"
        else:
            pert_desc = f"δ={delta:.4g}"
        title = (
            f"{mode_type} | {value_kind} | mode={mode_idx} λ={lam.real:.3f}{lam.imag:+.3f}j"
            f" | {pert_desc} | op={op_tag} | prefix=[{prefix_tag}]"
        )

        plot_modal_map_grid(
            baseline_by_t=base_maps,
            perturbed_by_t=pert_maps,
            coords=coords,
            adj=base_adj,
            active_node_ids=active_node_ids,
            value_kind=value_kind,
            mode_type=mode_type,
            title_meta=title,
            out_path=fig_path,
            show_basemap=not bool(cfg.get("no_basemap", False)),
        )
        saved["figures"][f"{mode_tag}_{value_kind}"] = str(fig_path)

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(cfg["checkpoint"]),
        "scenario_name": sample.get("name"),
        "mode_job_name": job_name,
        "prefix_source": sample.get("source"),
        "baseline_prefix": sample["tokens"],
        "baseline_prefix_summary": {
            "tokens": sample["tokens"],
            "time": sample["time"],
            "agent_id": sample["agent_id"],
            "holidays": sample["holidays"],
            "timezones": sample["timezones"],
            "events": sample["events"],
            "prefix_length": sample["prefix_length"],
        },
        "excluded_node_ids": list(EXCLUDED_NODE_IDS),
        "active_node_ids": active_node_ids,
        "mode_type": mode_type,
        "mode_index": mode_idx,
        "conjugate_index": conj_idx,
        "eigenvalue": _complex_to_meta(lam),
        "is_real": bool(is_real_eigenvalue(lam)),
        "auto_mode_jobs": bool(cfg.get("auto_mode_jobs", False)),
        "eig_abs_min": float(cfg["eig_abs_min"]) if cfg.get("auto_mode_jobs") else None,
        "alpha": alpha,
        "alpha_sigma_multiplier": (
            float(cfg.get("alpha_sigma", DEFAULT_ALPHA_SIGMA)) if alpha_from_sigma else None
        ),
        "sigma_i": sigma_i,
        "gamma": gamma,
        "delta": delta,
        "operator_variant": op_tag,
        "horizon": horizon,
        "output_type": output_type,
        "z0_norm": float(np.linalg.norm(z0)),
        "z0_perturbed_norm": float(np.linalg.norm(z0_p)),
        "z_tilde0_selected": {
            "real": float(np.real(z_t[mode_idx])),
            "imag": float(np.imag(z_t[mode_idx])),
        },
        "z_tilde0_perturbed_selected": {
            "real": float(np.real(z_tp[mode_idx])),
            "imag": float(np.imag(z_tp[mode_idx])),
        },
        "figure_paths": saved["figures"],
        "figure_title": title,
        "model_meta": {
            "context_ablation": load_meta["context_ablation"],
            "encoder_type": load_meta["encoder_type"],
            "z_dim": int(load_meta["hparams"]["z_dim"]),
            "uses_dual_A": bool(getattr(model, "uses_dual_A", False)),
        },
        "color_scale_policy": {
            "absolute_rows": "independent_per_row_shared_across_time",
            "delta_row": "diverging_zero_centered_shared_across_time",
        },
        "notes": [
            "Open-loop rollout: z_t = A^t z_0, scores for all move/stay tokens.",
            "Probability = softmax over full vocabulary; map shows move/stay only.",
            "Complex modes are always updated as conjugate pairs.",
        ],
    }
    meta_path = out_dir / f"{mode_tag}_m{mode_idx}_{op_tag}_h{horizon}_{prefix_tag}_meta.json"
    save_modal_map_metadata(meta_path, meta)
    saved["metadata"] = str(meta_path)
    return saved


def main():
    code_dir = Path(__file__).resolve().parent
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    cfg = dict(CONFIG)
    auto_jobs = bool(cfg.get("auto_mode_jobs", False))

    if not cfg.get("checkpoint"):
        raise ValueError("CONFIG['checkpoint'] を指定してください")
    if not cfg.get("out_dir"):
        raise ValueError("CONFIG['out_dir'] を指定してください")
    if not SCENARIOS:
        raise ValueError("SCENARIOS が空です")
    if not auto_jobs and not MODE_JOBS and not cfg.get("list_modes"):
        raise ValueError("auto_mode_jobs=False ですが MODE_JOBS が空です")

    use_cpu = bool(cfg.get("cpu", False))
    device = torch.device("cpu" if use_cpu or not torch.cuda.is_available() else "cuda")
    out_root = Path(cfg["out_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    model, tokenizer, base_adj, base_N, load_meta = load_model_and_data(
        cfg["checkpoint"], cfg["adj_path"], device
    )
    coords = load_node_coordinates(cfg["loc_csv"], num_nodes=base_N)
    active_node_ids = get_active_node_ids(base_N)
    operators = resolve_operator_variants(model)

    print("=" * 60)
    print("Modal map visualization")
    print(f"checkpoint: {cfg['checkpoint']}")
    print(f"out_dir:    {out_root}")
    print(f"operators:  {[tag for tag, _ in operators]}")
    print(f"active_nodes: {active_node_ids} (excluded: {EXCLUDED_NODE_IDS})")
    print(f"auto_jobs:  {auto_jobs}")
    if auto_jobs:
        print(f"eig_abs_min: {cfg.get('eig_abs_min')}")
    else:
        print(f"mode_jobs:  {len(MODE_JOBS)} (manual)")
    print(f"scenarios:  {len(SCENARIOS)}")
    print("=" * 60)

    if cfg.get("list_modes"):
        for op_tag, a in operators:
            eigvals = eigendecompose_operator(a)["eigvals"]
            print_eigenvalue_table(eigvals, operator_tag=op_tag)
            if auto_jobs:
                jobs = build_mode_jobs_from_eigenvalues(
                    eigvals,
                    eig_abs_min=float(cfg.get("eig_abs_min", 0.5)),
                    alpha=cfg.get("auto_alpha"),
                    gamma=float(cfg.get("auto_gamma", DEFAULT_GAMMA)),
                    delta=float(cfg.get("auto_delta", DEFAULT_DELTA)),
                )
                print(f"  -> auto jobs ({op_tag}): {len(jobs)}")
                for j in jobs:
                    print(f"     {j['name']}  type={j['mode_type']}  idx={j['mode_index']}")
        return

    success = 0
    failed = 0
    for op_tag, a in operators:
        eigen = eigendecompose_operator(a)
        eigvals, v, v_inv = eigen["eigvals"], eigen["V"], eigen["V_inv"]

        if auto_jobs:
            mode_jobs = build_mode_jobs_from_eigenvalues(
                eigvals,
                eig_abs_min=float(cfg.get("eig_abs_min", 0.5)),
                alpha=cfg.get("auto_alpha"),
                gamma=float(cfg.get("auto_gamma", DEFAULT_GAMMA)),
                delta=float(cfg.get("auto_delta", DEFAULT_DELTA)),
            )
            print(f"\n[OPERATOR] {op_tag}  auto_mode_jobs={len(mode_jobs)}")
        else:
            mode_jobs = MODE_JOBS
            print(f"\n[OPERATOR] {op_tag}  manual_mode_jobs={len(mode_jobs)}")

        if not mode_jobs:
            print(f"  [SKIP] {op_tag}: 閾値条件を満たすモードがありません")
            continue

        op_cfg = dict(cfg)
        op_cfg["operator_variant"] = op_tag

        for scenario in SCENARIOS:
            sample = scenario_to_sample(scenario)
            scen_name = sample["name"]
            scen_out = out_root / scen_name / op_tag
            scen_out.mkdir(parents=True, exist_ok=True)
            print(f"\n[SCENARIO] {scen_name} / {op_tag}  prefix={sample['tokens']}")

            for mode_job in mode_jobs:
                job_name = mode_job.get(
                    "name",
                    f"{mode_job.get('mode_type')}_m{mode_job.get('mode_index')}",
                )
                print(f"  [JOB] {job_name}")
                try:
                    run_one_job(
                        model=model,
                        tokenizer=tokenizer,
                        base_adj=base_adj,
                        base_N=base_N,
                        coords=coords,
                        active_node_ids=active_node_ids,
                        load_meta=load_meta,
                        A=a,
                        eigvals=eigvals,
                        V=v,
                        V_inv=v_inv,
                        sample=sample,
                        mode_job=mode_job,
                        cfg=op_cfg,
                        out_dir=scen_out,
                        device=device,
                    )
                    success += 1
                except Exception as e:
                    failed += 1
                    msg = (
                        f"  [FAIL] scenario={scen_name} op={op_tag} "
                        f"job={job_name}: {e}"
                    )
                    if cfg.get("strict"):
                        raise RuntimeError(msg) from e
                    print(msg)

    print("\n" + "=" * 60)
    print(f"Done. success={success}, failed={failed}")
    print(f"Output root: {out_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
