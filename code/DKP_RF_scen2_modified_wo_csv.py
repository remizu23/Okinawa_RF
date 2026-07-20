"""
※使い方※

例1: グリッド実験結果から複数モデルを選択して実行
python /home/mizutani/projects/RF/code/DKP_RF_scen2_modified_wo_csv.py \
  --session-root /home/mizutani/projects/RF/runs/2604_grid_experiments/grid_20260720_180128 \
  --grid-type model_x_zdim \
  --model-type transformer \
  --z-dims 32,64,128

例2: 特定のモデルパスを指定して実行
python /home/mizutani/projects/RF/code/DKP_RF_scen2_modified_wo_csv.py \
  --model-path /path/to/model_weights_*.pth
# 出力先未指定時 → exp_dir/scen2/ に自動保存


Koopman Mode Decomposition Analysis for DKP_RF (Original Model / No Jumps)

旧モデル（Prefix-only encoding + Autonomous Koopman rollout）用の
固有モード分解による解釈性検証コード

主な機能：
1. A行列の固有値分解と単位円上プロット
2. 各ステップでの潜在状態z_tを固有空間射影
3. 重みベクトルの固有空間寄与分析
4. トークン選択確率と広場有無の差分可視化
5. Greedy生成による5ステップ分の横並び可視化
6. ★追加: Koopman Biplot (z軌跡とトークン重みの同時プロット)
"""

import argparse
import json
import os
import csv
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch.nn.functional as F

# ユーザー定義モジュール
from network import Network, expand_adjacency_matrix
from tokenization import Tokenization
from DKP_RF import KoopmanRoutesFormer


# =========================================================
#  Config & Settings
# =========================================================

# シナリオ定義
SCENARIOS = [
    # {
    #     "name": "2,21,21,5,24",
    #     "prefix": [2,21,21,5,24],  # 初期prefix
    #     "time": 20240101,
    #     "agent_id": 0,
    #     "holiday": 1,
    #     "time_zone": 0,
    #     "plaza_node_tokens": [2, 21],  # 広場として扱うノード
    # },
    # {
    #     "name": "0,1,2,21,21",
    #     "prefix": [0,1,2,21,21],  # 初期prefix
    #     "time": 20240101,
    #     "agent_id": 0,
    #     "holiday": 1,
    #     "time_zone": 0,
    #     "plaza_node_tokens": [2, 21],  # 広場として扱うノード
    # },
    # {
    #     "name": "12,10,5,2,21",
    #     "prefix": [12,10,5,2,21],  # 初期prefix
    #     "time": 20240101,
    #     "agent_id": 0,
    #     "holiday": 1,
    #     "time_zone": 0,
    #     "plaza_node_tokens": [2, 21],  # 広場として扱うノード
    # },
    # {
    #     "name": "16,14,6,2,21",
    #     "prefix": [16,14,6,2,21],
    #     "time": 20240101,
    #     "agent_id": 0,
    #     "holiday": 1,
    #     "time_zone": 0,
    #     "plaza_node_tokens": [2, 21],
    # },
    {
        "name": "6,6,25,14,33",  # 卒論最終で使用1
        "prefix": [6,6,25,14,33],
        "time": 20240101,
        "agent_id": 0,
        "holiday": 1,              
        "time_zone": 0,            
        "plaza_node_tokens": [14, 33]
    },
    {
        "name": "16,35,35,14,33",  # 卒論最終で使用2
        "prefix": [16,35,35,14,33],
        "time": 20240101,
        "holiday": 1,              
        "time_zone": 0,            
        "agent_id": 0,
        "plaza_node_tokens": [14, 33]
    },
    # {
    #     "name": "0,1,2,21,21",
    #     "prefix": [0, 1, 2, 21, 21],  # 初期prefix
    #     "time": 20240101,
    #     "agent_id": 0,
    #     "holiday": 1,
    #     "time_zone": 0,
    #     "plaza_node_tokens": [2, 21],  # 広場として扱うノード
    # },
    # {
    #     "name": "12,10,5,2,21",
    #     "prefix": [12,10,5,2,21],  # 初期prefix
    #     "time": 20240101,
    #     "agent_id": 0,
    #     "holiday": 1,
    #     "time_zone": 0,
    #     "plaza_node_tokens": [2, 21],  # 広場として扱うノード
    # },
    # {
    #     "name": "16,14,6,2,21",
    #     "prefix": [16,14,6,2,21],  # 初期prefix
    #     "time": 20240101,
    #     "agent_id": 0,
    #     "holiday": 1,
    #     "time_zone": 0,
    #     "plaza_node_tokens": [2, 21],  # 広場として扱うノード
    # },
    # {
    #     "name": "6,5,4,11",
    #     "prefix": [6, 5, 4, 11, 30],
    #     "time": 20240101,
    #     "agent_id": 0,
    #     "holiday": 1,
    #     "time_zone": 0,
    #     "plaza_node_tokens": [11, 30],
    # },
    # {
    #     "name": "1,5,6,14",
    #     "prefix": [1, 5, 6, 14, 33],
    #     "time": 20240101,
    #     "agent_id": 0,
    #     "holiday": 1,              
    #     "time_zone": 0,            
    #     "plaza_node_tokens": [14, 33]
    # },
    # {
    #     "name": "18,16,14",
    #     "prefix": [18, 16, 14, 33],
    #     "time": 20240101,
    #     "holiday": 1,              
    #     "time_zone": 0,            
    #     "agent_id": 0,
    #     "plaza_node_tokens": [14, 33]
    # },
]

# デフォルトパス（CLI 未指定時）
DEFAULT_ADJ_PATH = "/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt"
DEFAULT_NUM_ROLLOUT_STEPS = 15
CHECKPOINT_MANIFEST = "checkpoint.json"

# モード分類しきい値（|λ| < thresh を短期減衰モードとみなす）
EIG_ABS_SHORT_TERM_THRESH = 0.5


# =========================================================
#  Utility Functions
# =========================================================

def get_movable_tokens(current_node, adj_matrix, pad_token_id=38, end_token_id=38):
    """現在位置から移動可能なトークンを取得"""
    if isinstance(adj_matrix, torch.Tensor):
        adj_np = adj_matrix.cpu().numpy()
    else:
        adj_np = adj_matrix
    
    neighbors = np.where(adj_np[current_node] > 0)[0]
    movable_tokens = []
    
    # Move tokens
    for neighbor in neighbors:
        if 0 <= neighbor <= 18:
            movable_tokens.append(int(neighbor))
    
    # Stay tokens
    if 0 <= current_node <= 18:
        stay_token = current_node + 19
        movable_tokens.append(stay_token)
    
    movable_tokens.append(end_token_id)
    return sorted(movable_tokens)


def get_token_label(token_id, tokenizer):
    """トークンIDから，グラフ表示用のラベル文字列を取得"""
    if 0 <= token_id <= 18:
        return f"M{token_id}"  #　"{token_id}移動"　みたいな方がいいかも．
    elif 19 <= token_id <= 37:
        return f"S{token_id-19}"
    elif token_id in tokenizer.SPECIAL_TOKENS.values():
        for k, v in tokenizer.SPECIAL_TOKENS.items():
            if v == token_id:
                return k
        return f"?{token_id}"
    else:
        return f"T{token_id}"



def softmax_np(x):
    """numpy版の安定softmax（1次元想定）"""
    x = np.asarray(x)
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum()
    return e / s if s != 0 else np.full_like(e, 1.0 / len(e))


def token_probs_to_move_stay_end_probs(
    token_probs_1d: np.ndarray,
    *,
    num_nodes: int = 19,
    stay_offset: int = 19,
    end_token_id: int = 38,
):
    """token確率ベクトル（語彙次元）を、move/stay/ENDへ集約する。

    - move token: [0..num_nodes-1] → MOVE i
    - stay token: [stay_offset..stay_offset+num_nodes-1] → STAY i
    - END: end_token_id → END

    返り値は長さ (2*num_nodes + 1) の 1次元配列：
    [move0..move18, stay0..stay18, END]
    """
    p = np.asarray(token_probs_1d).reshape(-1)
    if p.shape[0] <= end_token_id:
        raise ValueError(f"token_probs length {p.shape[0]} is smaller than end_token_id={end_token_id}")
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be positive, got {num_nodes}")
    if stay_offset < 0:
        raise ValueError(f"stay_offset must be >=0, got {stay_offset}")

    move_stay_end_p = np.zeros((2 * num_nodes + 1,), dtype=float)

    # move tokens: 0..(num_nodes-1)
    if p.shape[0] < num_nodes:
        raise ValueError(f"token_probs length {p.shape[0]} is smaller than num_nodes={num_nodes}")
    move_stay_end_p[:num_nodes] += p[:num_nodes]

    # stay tokens: stay_offset..(stay_offset+num_nodes-1) (if present)
    stay_end = stay_offset + num_nodes
    if p.shape[0] >= stay_end:
        move_stay_end_p[num_nodes:2 * num_nodes] += p[stay_offset:stay_end]

    # END
    move_stay_end_p[2 * num_nodes] += p[end_token_id]

    return move_stay_end_p


def _write_node_prob_csv(node_by_time: np.ndarray, row_labels: list[str], out_path: str):
    """node_by_time: shape=(num_rows, T), rows correspond to row_labels."""
    node_by_time = np.asarray(node_by_time)
    if node_by_time.ndim != 2:
        raise ValueError(f"node_by_time must be 2D, got shape={node_by_time.shape}")
    if node_by_time.shape[0] != len(row_labels):
        raise ValueError(f"row_labels length mismatch: {len(row_labels)} vs {node_by_time.shape[0]}")

    T = int(node_by_time.shape[1])
    header = ["token"] + [f"t{t}" for t in range(1, T + 1)]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, label in enumerate(row_labels):
            w.writerow([label] + [float(x) for x in node_by_time[i, :].tolist()])


def _infer_model_hparams(ckpt_config, state_dict, base_N):
    """ckpt config が欠けていても state_dict 形状から復元する。"""
    c = ckpt_config if isinstance(ckpt_config, dict) else {}

    def pick(*keys, default=None):
        for k in keys:
            if k in c:
                return c[k]
        return default

    # state_dict から推定できる値
    inferred_vocab_size = int(state_dict["token_embedding.weight"].shape[0])
    inferred_token_emb_dim = int(state_dict["token_embedding.weight"].shape[1])
    inferred_d_model = int(state_dict["input_proj.weight"].shape[0])
    inferred_z_dim = int(state_dict["A"].shape[0])
    inferred_num_agents = int(state_dict["agent_embedding.weight"].shape[0])
    inferred_agent_emb_dim = int(state_dict["agent_embedding.weight"].shape[1])
    inferred_max_stay_count = int(state_dict["stay_embedding.weight"].shape[0] - 1)
    inferred_stay_emb_dim = int(state_dict["stay_embedding.weight"].shape[1])
    inferred_holiday_emb_dim = int(state_dict["holiday_embedding.weight"].shape[1])
    inferred_time_zone_emb_dim = int(state_dict["time_zone_embedding.weight"].shape[1])
    inferred_event_emb_dim = int(state_dict["event_embedding.weight"].shape[1])
    inferred_d_ff = int(state_dict["transformer_block.layers.0.linear1.weight"].shape[0])

    # 層数は state_dict のキー数から推定
    transformer_layer_ids = set()
    for k in state_dict.keys():
        if k.startswith("transformer_block.layers."):
            parts = k.split(".")
            if len(parts) > 3 and parts[3].isdigit():
                transformer_layer_ids.add(int(parts[3]))
    inferred_num_layers = (max(transformer_layer_ids) + 1) if transformer_layer_ids else 3

    # nhead は通常 config に入っている想定。無い場合は安全な候補から推定
    nhead_from_cfg = pick("nhead", "head_num", default=None)
    if nhead_from_cfg is None:
        for cand in (8, 4, 2, 1):
            if inferred_d_model % cand == 0:
                nhead_from_cfg = cand
                break
    resolved_nhead = int(nhead_from_cfg)

    return {
        "vocab_size": int(pick("vocab_size", default=inferred_vocab_size)),
        "token_emb_dim": int(pick("token_emb_dim", "d_ie", default=inferred_token_emb_dim)),
        "d_model": int(pick("d_model", "d_ie", default=inferred_d_model)),
        "nhead": resolved_nhead,
        "num_layers": int(pick("num_layers", "B_de", default=inferred_num_layers)),
        "d_ff": int(pick("d_ff", default=inferred_d_ff)),
        "z_dim": int(pick("z_dim", default=inferred_z_dim)),
        "pad_token_id": int(pick("pad_token_id", default=38)),
        "base_N": int(pick("base_N", default=base_N)),
        "num_agents": int(pick("num_agents", default=inferred_num_agents)),
        "agent_emb_dim": int(pick("agent_emb_dim", default=inferred_agent_emb_dim)),
        "max_stay_count": int(pick("max_stay_count", default=inferred_max_stay_count)),
        "stay_emb_dim": int(pick("stay_emb_dim", default=inferred_stay_emb_dim)),
        "holiday_emb_dim": int(pick("holiday_emb_dim", default=inferred_holiday_emb_dim)),
        "time_zone_emb_dim": int(pick("time_zone_emb_dim", default=inferred_time_zone_emb_dim)),
        "event_emb_dim": int(pick("event_emb_dim", default=inferred_event_emb_dim)),
    }


# =========================================================
#  Eigen Analyzer Class
# =========================================================

class KoopmanEigenAnalyzer:  # 計算用の行列を準備する．固有ベクトル，固有空間分解後のWなど．
    """Koopman演算子Aの固有値分解と解析"""
    
    def __init__(self, model):
        self.model = model
        self.z_dim = model.z_dim
        
        self.A_np = model.A.detach().cpu().numpy()  
        # detatchで学習グラフから切り離し，ここから先は単なる数字として扱う．これ以上操作を記録しない．微分対象ではない．
        # 勾配情報を残すと，この後の解析用の操作の履歴も残そうとしてメモリを食うし，numpy変換もできないため．
        
        eigvals, eigvecs = scipy.linalg.eig(self.A_np)
        # scipyのlinalg関数は，上記のように使うことで固有値，固有ベクトルを呼び出せる

        sort_idx = np.argsort(np.abs(eigvals))[::-1]
        # ↑ eigvals を降順に（←[::-1]のため）並べるための番号指示書

        self.eigvals = eigvals[sort_idx]
        # eigvals を sort_idx 順に並び替える．
        # self.eigvals とし，このオブジェクトの属性としてこれを保存する．

        self.V = eigvecs[:, sort_idx]
        # eigvecs の二次元配列についても，行(=各要素)についてはそのままに，列をsort_idxで並び替え
        # →左から，固有値の大きい順に対応して固有ベクトルが並んだ配列になる．
        # ([縦ベクトル1],[縦ベクトル2],...)という形の配列．次元数(行)*固有値数(列)．

        self.V_inv = scipy.linalg.inv(self.V)
        # Vの逆行列を保存．

        self.W = model.to_logits.weight.detach().cpu().numpy()
        self.b = model.to_logits.bias.detach().cpu().numpy()
        self.W_modal = self.W @ self.V
        # 重みベクトルWの固有空間投影．形式的な説明は下記．
        # z = V * α で固有空間分解すると，
        # logits = Wz + b = W(V * α) + b
        # となるので，W * V を W_modal として保存している．
        
        print(f"Eigenvalue decomposition complete: Max |λ| = {np.abs(self.eigvals).max():.4f}")
    

    def transform_to_eigenspace(self, z): # zの固有投影αを求める関数．
        if z.ndim == 1:
            return self.V_inv @ z
            # V * α = z なので（V=([縦ベクトル1],[縦ベクトル2],...)），
            # α = V_inv * z とする．
        else: # 不使用
            return (self.V_inv @ z.T).T
    

    def plot_eigenvalues(self, save_path):  # 固有値の複素平面プロット
        fig, ax = plt.subplots(figsize=(8, 8))
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1, alpha=0.3)
        
        ax.scatter(self.eigvals.real, self.eigvals.imag, 
                  c=np.arange(len(self.eigvals)), cmap='coolwarm',
                  s=100, edgecolors='black', linewidth=1.5, zorder=5)
        
        for i, ev in enumerate(self.eigvals):
            ax.annotate(f'λ{i}', (ev.real, ev.imag), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Real'); ax.set_ylabel('Imaginary')
        ax.set_title('Eigenvalues of Koopman Matrix A')
        ax.axhline(0, color='black', alpha=0.3); ax.axvline(0, color='black', alpha=0.3)
        ax.grid(True, alpha=0.3); ax.axis('equal')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# =========================================================
#  Scenario Analysis
# =========================================================

def encode_prefix_with_plaza(model, tokenizer, prefix, agent_id, holiday, time_zone, 
                              plaza_tokens, use_plaza, device):
    """Prefixをエンコードして初期潜在状態z_0を取得"""
    seq_len = len(prefix)
    tokens = torch.tensor([prefix], dtype=torch.long).to(device) # prefixをTensor化　 形状：(1, seq_len)に
    stay_counts = tokenizer.calculate_stay_counts(tokens).to(device) # [3,3,3,5] → [1,2,3,1]
    agent_ids = torch.tensor([agent_id], dtype=torch.long).to(device)
    holidays = torch.tensor([[holiday] * seq_len], dtype=torch.long).to(device) # (1, seq_len)
    time_zones = torch.tensor([[time_zone] * seq_len], dtype=torch.long).to(device) # (1, seq_len)
    
    events = torch.zeros((1, seq_len), dtype=torch.long).to(device) # events初期化

    # use_plaza が True か False かで，with/without ver. の prefix 埋め込みを区別して作っている．
    if use_plaza:
        for pos, tok in enumerate(prefix): # for pos, tok in enumerate(prefix) で位置＆要素をループできる．
            if tok in plaza_tokens: # 「plaza_tokens(引数で取得)に合致するtok(要素)の場合に，」
                events[0, pos] = 1  # 「その時のposについて，events[0, pos] = 1 にする．」
    
    with torch.no_grad(): # 計算グラフから切り離す．
        z_0, _ = model.encode_prefix(  # transformerでprefixをencodeする，KPRFクラスで定義した関数．2つ目の引数h_lastは不使用．
            tokens, stay_counts, agent_ids,
            holidays, time_zones, events
        )
    return z_0[0]


def greedy_rollout_with_analysis(model, analyzer, tokenizer, network, adj_matrix,
                                  scenario, num_steps, device, out_dir):
    """Greedyロールアウトしながら解析

    ★更新:
    - これまでは「with」で生成した経路（current_node）に合わせて movable を作り、
      その上で with/without の確率分布を比較していた。
    - 今回は「with経路」と「without経路」をそれぞれ生成し、
      それぞれの経路に整合した movable 上で同じフォーマットの比較図を2枚出力する。
      (tag='withpath' / 'wopath')
    """
    print(f"\n{'='*60}")
    print(f"Analyzing Scenario: {scenario['name']}")
    print(f"{'='*60}")

    prefix = scenario['prefix']
    agent_id = scenario['agent_id']
    holiday = scenario['holiday']
    time_zone = scenario['time_zone']
    plaza_tokens = scenario.get('plaza_node_tokens', [])

    # 初期エンコード（with/withoutは，use_plaza = True/False で区別．）
    z_with = encode_prefix_with_plaza(model, tokenizer, prefix, agent_id, holiday, time_zone, plaza_tokens, True, device).cpu().numpy()
    z_without = encode_prefix_with_plaza(model, tokenizer, prefix, agent_id, holiday, time_zone, plaza_tokens, False, device).cpu().numpy()

    # np配列に変換（np.asarray）し，1次元にする（[1, z_dim]などで来た時に，[z_dim]にする）保険．
    z_with = np.asarray(z_with).reshape(-1)
    z_without = np.asarray(z_without).reshape(-1)

    # 経路ごとの current_node を別々に持つ．nodeなので19で割った余り．
    current_node_withpath = prefix[-1] % 19
    current_node_wopath = prefix[-1] % 19

    generated_with = []
    generated_without = []

    step_data_withpath = []
    step_data_wopath = []

    # --- 全ノード確率（node0..18 + END）を step ごとに保存 ---
    node_probs_with_steps = []
    node_probs_without_steps = []
    node_probs_diff_steps = []

    ended_with = False
    ended_wo = False

    # A_np = analyzer.model.A.detach().cpu().numpy() としていたが，下記でいい，
    A_np = analyzer.A_np

    # rolloutステップ数分，下記を進める．num_stepsは，NUM_ROLLOUT_STEPSとしてハードコーディングした値が入る．
    for step in range(num_steps):
        print(f"\n--- Step {step+1}/{num_steps} ---")

        z_next_with = A_np @ z_with
        z_next_without = A_np @ z_without

        alpha_with = analyzer.transform_to_eigenspace(z_next_with)
        alpha_without = analyzer.transform_to_eigenspace(z_next_without)

        logits_with = analyzer.W @ z_next_with + analyzer.b
        logits_without = analyzer.W @ z_next_without + analyzer.b

        probs_with = softmax_np(logits_with)
        probs_without = softmax_np(logits_without)

        # --- move/stay/ENDへ集約（ENDは padding token <p>=38 として扱う） ---
        node_p_with = token_probs_to_move_stay_end_probs(
            probs_with, num_nodes=19, stay_offset=19, end_token_id=38
        )
        node_p_wo = token_probs_to_move_stay_end_probs(
            probs_without, num_nodes=19, stay_offset=19, end_token_id=38
        )
        node_probs_with_steps.append(node_p_with)
        node_probs_without_steps.append(node_p_wo)
        node_probs_diff_steps.append(node_p_with - node_p_wo)

        # ==========================
        # 1) with 経路での比較
        # ==========================
        if not ended_with:
            movable_tokens = get_movable_tokens(current_node_withpath, adj_matrix)

            # 「選択肢（movable）の範囲の確率合計」を分母にして，確率を再正規化
            movable_probs_with = probs_with[movable_tokens]
            movable_probs_with = movable_probs_with / max(movable_probs_with.sum(), 1e-12)

            movable_probs_without = probs_without[movable_tokens]
            movable_probs_without = movable_probs_without / max(movable_probs_without.sum(), 1e-12)

            movable_probs_diff = movable_probs_with - movable_probs_without

            next_token_with = movable_tokens[int(np.argmax(movable_probs_with))] # np.argmaxは，配列の最大要素のindex(位置)を返す
            next_token_without_on_withpath = movable_tokens[int(np.argmax(movable_probs_without))]

            generated_with.append(next_token_with)

            print(f"  [withpath] Current node: {current_node_withpath}")
            print(f"  [withpath] Selected (with): {get_token_label(next_token_with, tokenizer)} (p={movable_probs_with.max():.4f})")

            step_data_withpath.append({
                'step': step,
                'z_with': z_next_with.copy(),
                'z_without': z_next_without.copy(),
                'alpha_with': alpha_with.copy(),
                'alpha_without': alpha_without.copy(),
                'movable_tokens': movable_tokens.copy(),
                'movable_probs_with': movable_probs_with.copy(),
                'movable_probs_without': movable_probs_without.copy(),
                'movable_probs_diff': movable_probs_diff.copy(),
                'next_token_with': next_token_with,
                'next_token_without': next_token_without_on_withpath,
            })

            # withpath の current_node（19で割った余り）更新．
            # ただし，endした場合は，ended_with = Trueとして，with経路のループを外れ，withoutの計算に入る．
            if next_token_with < 19:
                current_node_withpath = next_token_with
            elif 19 <= next_token_with <= 37:
                current_node_withpath = next_token_with - 19
            else:
                ended_with = True

        # ==========================
        # 2) w/o 経路での比較
        # ==========================
        if not ended_wo:
            movable_tokens_wo = get_movable_tokens(current_node_wopath, adj_matrix)

            movable_probs_with_on_wo = probs_with[movable_tokens_wo]
            movable_probs_with_on_wo = movable_probs_with_on_wo / max(movable_probs_with_on_wo.sum(), 1e-12)

            movable_probs_wo = probs_without[movable_tokens_wo]
            movable_probs_wo = movable_probs_wo / max(movable_probs_wo.sum(), 1e-12)

            movable_probs_diff_wo = movable_probs_with_on_wo - movable_probs_wo

            next_token_wo = movable_tokens_wo[int(np.argmax(movable_probs_wo))]
            next_token_with_on_wopath = movable_tokens_wo[int(np.argmax(movable_probs_with_on_wo))]

            generated_without.append(next_token_wo)

            print(f"  [wopath] Current node: {current_node_wopath}")
            print(f"  [wopath] Selected (w/o): {get_token_label(next_token_wo, tokenizer)} (p={movable_probs_wo.max():.4f})")

            step_data_wopath.append({
                'step': step,
                'z_with': z_next_with.copy(),
                'z_without': z_next_without.copy(),
                'alpha_with': alpha_with.copy(),
                'alpha_without': alpha_without.copy(),
                'movable_tokens': movable_tokens_wo.copy(),
                'movable_probs_with': movable_probs_with_on_wo.copy(),
                'movable_probs_without': movable_probs_wo.copy(),
                'movable_probs_diff': movable_probs_diff_wo.copy(),
                'next_token_with': next_token_with_on_wopath,
                'next_token_without': next_token_wo,
            })

            # wopath の current_node 更新（w/o が選んだトークンで進む）
            if next_token_wo < 19:
                current_node_wopath = next_token_wo
            elif 19 <= next_token_wo <= 37:
                current_node_wopath = next_token_wo - 19
            else:
                ended_wo = True

        # 潜在更新（自律なので常に進める）
        z_with = z_next_with
        z_without = z_next_without

        # NOTE: END が選ばれた後も「生のモデル確率」を見たいので、ここでは break しない。

    # --- 図出力 ---
    if len(step_data_withpath) > 0:
        visualize_rollout_analysis(analyzer, step_data_withpath, scenario, tokenizer, out_dir, tag="withpath")
        # biplot は従来通り withpath を採用（最小変更）
        visualize_koopman_biplot_grid(analyzer, step_data_withpath, scenario, tokenizer, out_dir)

    if len(step_data_wopath) > 0:
        visualize_rollout_analysis(analyzer, step_data_wopath, scenario, tokenizer, out_dir, tag="wopath")

    print(f"\nGenerated sequence (with plaza): {prefix} -> {generated_with}")
    print(f"Generated sequence (w/o plaza):  {prefix} -> {generated_without}")

    # --- CSV出力（縦=move/stay/END, 横=時間） ---
    if len(node_probs_with_steps) > 0:
        node_labels = (
            [f"move{i}" for i in range(19)]
            + [f"stay{i}" for i in range(19)]
            + ["END(pad_token=38)"]
        )
        mat_with = np.asarray(node_probs_with_steps, dtype=float).T  # (39, T)
        mat_wo = np.asarray(node_probs_without_steps, dtype=float).T  # (39, T)
        mat_diff = np.asarray(node_probs_diff_steps, dtype=float).T  # (39, T)

        csv_with = os.path.join(out_dir, f"{scenario['name']}_move_stay_prob_with.csv")
        csv_wo = os.path.join(out_dir, f"{scenario['name']}_move_stay_prob_without.csv")
        csv_diff = os.path.join(out_dir, f"{scenario['name']}_move_stay_prob_diff_withminuswithout.csv")

        _write_node_prob_csv(mat_with, node_labels, csv_with)
        _write_node_prob_csv(mat_wo, node_labels, csv_wo)
        _write_node_prob_csv(mat_diff, node_labels, csv_diff)

        print(f"Saved move/stay prob CSV (with): {csv_with}")
        print(f"Saved move/stay prob CSV (without): {csv_wo}")
        print(f"Saved move/stay prob CSV (diff): {csv_diff}")



def _add_long_short_inset_heatmaps(fig, parent_ax, long_vals, short_vals, vlim, title_long="Long-term", title_short="Short-term"):
    """③④のモード×トークンヒートマップの下に、長期/短期寄与の合計ヒートマップ（2段）を追加する"""
    # ~入力の説明~
    # fig : matplotlibの図全体
    # parent_ax：すでに描いてあるメインのAxes（上段の大きい図）
    # long_vals, short_vals：トークンごとの値（長期寄与合計、短期寄与合計）
    # vlim：色の上下限（±vlimで揃える）

    # parent_axを上に詰めて下部にスペースを作る
    pos = parent_ax.get_position() # pos = parent_axが図のどこにあるか．
    # pos.x0, pos.y0, pos.width, pos.height のような値を持つ（座標は 図全体を0〜1とした比率）

    inset_h = pos.height * 0.11  # 追加する小さいヒートマップの一段の高さ（pos.heightの11%）
    gap = pos.height * 0.01      # 長期と短期の間のすき間（pos.heightの1%）
    main_h = pos.height - (inset_h * 2 + gap)  # 上記から算出される，元のparent_ax の適正な大きさ

    parent_ax.set_position([pos.x0, pos.y0 + inset_h * 2 + gap, pos.width, main_h])
    # pos.y0を上にずらし，高さを main_h に縮める．この下に，新しい合計ヒートマップを入れていく．

    # 最下段（短期）
    ax_short = fig.add_axes([pos.x0, pos.y0, pos.width, inset_h])
    im_s = ax_short.imshow(short_vals.reshape(1, -1), cmap='coolwarm', aspect='auto', vmin=-vlim, vmax=vlim)
    ax_short.set_yticks([0]); ax_short.set_yticklabels([title_short], fontsize=7)
    ax_short.set_xticks([])

    # 中段（長期）
    ax_long = fig.add_axes([pos.x0, pos.y0 + inset_h + gap, pos.width, inset_h])
    im_l = ax_long.imshow(long_vals.reshape(1, -1), cmap='coolwarm', aspect='auto', vmin=-vlim, vmax=vlim)
    ax_long.set_yticks([0]); ax_long.set_yticklabels([title_long], fontsize=7)
    ax_long.set_xticks([])

    # 枠線を薄く
    for ax in (ax_long, ax_short):
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_alpha(0.4)

    return im_l, im_s  # 画像オブジェクトとして返す．


def visualize_rollout_analysis(analyzer, step_data, scenario, tokenizer, out_dir, tag="withpath"):
    """ロールアウト解析結果を可視化（5ステップ横並び）"""
    num_steps = len(step_data)
    z_dim = analyzer.z_dim
    
    fig = plt.figure(figsize=(num_steps * 6, 28))
    gs = gridspec.GridSpec(6, num_steps, figure=fig, hspace=0.35, wspace=0.25,
                          top=0.96, bottom=0.04, left=0.05, right=0.98)
    fig.suptitle(f"Koopman Mode Decomposition Analysis ({tag}): {scenario['name']}", fontsize=24, weight='bold')
    
    # スケール計算用の収集
    all_alpha_vals = []
    all_W_vals = []
    all_prob_vals = []
    all_diff_vals = []
    all_longshort_vals = []  # 長期/短期合計寄与（with, diff をまとめてスケール計算）

    eig_abs = np.abs(analyzer.eigvals)
    short_mask = eig_abs < EIG_ABS_SHORT_TERM_THRESH
    long_mask = ~short_mask

    # --- export buffer for stacked probability decomposition (Row ⑤) ---
    prob_rows = []

    for data in step_data:
        all_alpha_vals.extend([data['alpha_with'], data['alpha_without']])
        movable = data['movable_tokens']
        all_W_vals.append(analyzer.W_modal[movable, :]) 
        # ↑W_modal は，[[token1横ベクトル],[token2横ベクトル], ...]．そのうちmovableのトークンだけとってくる．
        
        # 長期/短期の合計寄与（③: with / ④: diff）．

        # data['alpha_with']は，zを固有投影した16次元縦ベクトル．それに移動可能なトークンの重み行列W_modal[movable, :]をかけるが，
        # 単なる行列掛け算(@)ではなく要素毎掛け算(*)にすることで，W_modalの (token数，次元数) のサイズのまま，
        # 「各トークン・各次元での寄与」を，そのまま配列 C として保存している．→ ここは実際の数式と違う操作をしている部分！
            # .reshape(1, -1)は一応つけている．data['alpha_with']が，(z_dim,)→(1, z_dim)（＝1行の行列）になる．
            # → Numpyの，＊（要素毎掛け算）を確実にさせる．
        # ここで，寄与は実部のみを採用する（np.real）．
        C_with = np.real(analyzer.W_modal[movable, :] * data['alpha_with'].reshape(1, -1))
        C_diff = np.real(analyzer.W_modal[movable, :] * (data['alpha_with'] - data['alpha_without']).reshape(1, -1))
        
        # 上記のC_with，C_diff（各トークン・各次元での寄与）を，長期次元・短期次元ごとに足し合わせる．
        # long_mask は長さ z_dim の True/False 配列．これに合致する列(=次元)のみ出してきて，その配列のsumをaxis=1(次元方向)に計算
        long_with = C_with[:, long_mask].sum(axis=1) # スカラー量
        short_with = C_with[:, short_mask].sum(axis=1)

        long_diff = C_diff[:, long_mask].sum(axis=1)
        short_diff = C_diff[:, short_mask].sum(axis=1)

        # 【スケールの算出のために下記を作成：】
        all_longshort_vals.extend([long_with, short_with, long_diff, short_diff]) 
        # all_longshort_vals は「4要素の配列が，たくさん入ったリスト」になっていく．スケールの算出のために作成

        all_prob_vals.extend([data['movable_probs_with'], data['movable_probs_without']])
        # 事前に計算していた，movable_probs_with・movable_probs_withoutを格納．スケールの算出のために作成

        all_diff_vals.append(data['movable_probs_diff'])
        # appendはextendと違い一つの要素として追加．
    
    alpha_vmax = max(np.abs(np.concatenate(all_alpha_vals)).max(), 1e-6)
    W_vmax = max(np.abs(np.concatenate(all_W_vals, axis=0)).max(), 1e-6)

    prob_vmax = max(np.concatenate(all_prob_vals).max(), 1e-6)
    diff_vmax = max(np.abs(np.concatenate(all_diff_vals)).max(), 1e-6)
    longshort_vmax = max(np.abs(np.concatenate(all_longshort_vals)).max(), 1e-6) if len(all_longshort_vals) > 0 else 1e-6
    
    eigenmode_labels = [f"{lam.real:.2f}{lam.imag:+.2f}j" for lam in analyzer.eigvals]
    
    # 各Rowのcolorbar用に最後にハンドルを保持
    last_im_alpha = None
    last_im_alpha_wo = None
    last_im_W = None
    last_im_Wdiff = None
    row_alpha_axes = []
    row_alpha_wo_axes = []
    row_W_axes = []
    row_Wdiff_axes = []

    for step_idx, data in enumerate(step_data): # ループステップ数と要素を，同時にループ．
        col = step_idx
        movable = data['movable_tokens']
        token_labels = [get_token_label(t, tokenizer) for t in movable]
        num_movable = len(movable)
        
        # alpha準備
        alpha_with_plot = np.real(np.asarray(data['alpha_with'])).reshape(1, -1)
        alpha_without_plot = np.real(np.asarray(data['alpha_without'])).reshape(1, -1)
        delta_alpha = np.real(np.asarray(data['alpha_with']) - np.asarray(data['alpha_without']))
        
        # Row 0: alpha (with)
        ax1 = fig.add_subplot(gs[0, col])
        im1 = ax1.imshow(alpha_with_plot, cmap='coolwarm', aspect='auto', vmin=-alpha_vmax, vmax=alpha_vmax)
        ax1.set_title(f"Step {step_idx+1}\n① α (with)", fontsize=11)
        ax1.set_xticks([]); ax1.set_yticks([])
        row_alpha_axes.append(ax1)
        last_im_alpha = im1
        
        # Row 1: alpha (w/o)
        ax2 = fig.add_subplot(gs[1, col])
        im2 = ax2.imshow(alpha_without_plot, cmap='coolwarm', aspect='auto', vmin=-alpha_vmax, vmax=alpha_vmax)
        ax2.set_title("② α (w/o)", fontsize=11)
        ax2.set_xticks(range(z_dim)); ax2.set_xticklabels(eigenmode_labels, rotation=90, fontsize=7)
        ax2.set_yticks([])
        row_alpha_wo_axes.append(ax2)
        last_im_alpha_wo = im2

        # Row 2: W contrib (with)
        ax3 = fig.add_subplot(gs[2, col])
        W_contrib = np.real(analyzer.W_modal[movable, :] * data['alpha_with'].reshape(1, -1))
        im3 = ax3.imshow(W_contrib.T, cmap='coolwarm', aspect='auto', vmin=-W_vmax, vmax=W_vmax)
        # ★追加: 長期/短期モードのトークン別合計寄与（with）
        long_sum = W_contrib[:, long_mask].sum(axis=1)
        short_sum = W_contrib[:, short_mask].sum(axis=1)
        _add_long_short_inset_heatmaps(fig, ax3, long_sum, short_sum, longshort_vmax,
                                       title_long=f"Long(|λ|≥{EIG_ABS_SHORT_TERM_THRESH})",
                                       title_short=f"Short(|λ|<{EIG_ABS_SHORT_TERM_THRESH})")
        ax3.set_title("③ Mode×Token (with)", fontsize=11)
        ax3.set_xticks(range(num_movable)); ax3.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=7)
        ax3.set_yticks(range(z_dim)); ax3.set_yticklabels(eigenmode_labels, fontsize=6)
        row_W_axes.append(ax3)
        last_im_W = im3

        # Row 3: W diff
        ax4 = fig.add_subplot(gs[3, col])
        W_diff = np.real(analyzer.W_modal[movable, :] * delta_alpha.reshape(1, -1))
        im4 = ax4.imshow(W_diff.T, cmap='coolwarm', aspect='auto', vmin=-diff_vmax, vmax=diff_vmax)
        # ★追加: 長期/短期モードのトークン別合計寄与（diff）
        long_sum_d = W_diff[:, long_mask].sum(axis=1)
        short_sum_d = W_diff[:, short_mask].sum(axis=1)
        _add_long_short_inset_heatmaps(fig, ax4, long_sum_d, short_sum_d, longshort_vmax,
                                       title_long=f"Long(|λ|≥{EIG_ABS_SHORT_TERM_THRESH})",
                                       title_short=f"Short(|λ|<{EIG_ABS_SHORT_TERM_THRESH})")
        ax4.set_title("④ ΔContrib (with-w/o)", fontsize=11)
        ax4.set_xticks(range(num_movable)); ax4.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=7)
        ax4.set_yticks([])
        row_Wdiff_axes.append(ax4)
        last_im_Wdiff = im4

        # Row 4: Prob (stacked by long/short-mode attribution)
        ax5 = fig.add_subplot(gs[4, col])
        x = np.arange(num_movable)

        # Token-level logit contributions split by long/short modes (same definition as heatmaps)
        C_with_step = np.real(analyzer.W_modal[movable, :] * data['alpha_with'].reshape(1, -1))      # [num_movable, z_dim]
        C_wo_step   = np.real(analyzer.W_modal[movable, :] * data['alpha_without'].reshape(1, -1))   # [num_movable, z_dim]
        long_with_c  = C_with_step[:, long_mask].sum(axis=1)
        short_with_c = C_with_step[:, short_mask].sum(axis=1)
        long_wo_c    = C_wo_step[:, long_mask].sum(axis=1)
        short_wo_c   = C_wo_step[:, short_mask].sum(axis=1)

        # Convert to "割合" per token using SIGNED attribution (long/short can push up or pull down)
        # ratio = long_contrib / (long_contrib + short_contrib)
        # Note: when (long+short) is ~0, ratio becomes unstable; we fall back to 0 (=> all assigned to short)
        eps = 1e-12
        with_total = long_with_c + short_with_c
        wo_total   = long_wo_c   + short_wo_c

        with_long_ratio = np.where(np.abs(with_total) > eps, long_with_c / with_total, 0.0)
        wo_long_ratio   = np.where(np.abs(wo_total)   > eps, long_wo_c   / wo_total,   0.0)

        p_with = data['movable_probs_with']
        p_wo   = data['movable_probs_without']

        # Decompose probability into long/short components (can be negative if that group reduces the logit)
        p_with_long  = p_with * with_long_ratio
        p_with_short = p_with - p_with_long
        p_wo_long    = p_wo   * wo_long_ratio
        p_wo_short   = p_wo   - p_wo_long


        # --- collect per-token decomposition values for export ---
        for _i_tok, _tok in enumerate(movable):
            prob_rows.append({
                "scenario": scenario["name"],
                "tag": tag,
                "step": int(step_idx + 1),
                "token_id": int(_tok),
                "token_label": get_token_label(int(_tok), tokenizer),
                "p_with": float(p_with[_i_tok]),
                "p_without": float(p_wo[_i_tok]),
                "long_contrib_with": float(long_with_c[_i_tok]),
                "short_contrib_with": float(short_with_c[_i_tok]),
                "total_contrib_with": float((long_with_c[_i_tok] + short_with_c[_i_tok])),
                "with_long_ratio": float(with_long_ratio[_i_tok]),
                "p_with_long": float(p_with_long[_i_tok]),
                "p_with_short": float(p_with_short[_i_tok]),
                "long_contrib_without": float(long_wo_c[_i_tok]),
                "short_contrib_without": float(short_wo_c[_i_tok]),
                "total_contrib_without": float((long_wo_c[_i_tok] + short_wo_c[_i_tok])),
                "without_long_ratio": float(wo_long_ratio[_i_tok]),
                "p_without_long": float(p_wo_long[_i_tok]),
                "p_without_short": float(p_wo_short[_i_tok]),
            })

        w = 0.3
        ax5.bar(x - 0.15, p_with_long,  w, label='With (long)',  color='coral')
        ax5.bar(x - 0.15, p_with_short, w, bottom=p_with_long,  label='With (short)', color='orange')
        ax5.bar(x + 0.15, p_wo_long,    w, label='W/o (long)',   color='steelblue')
        ax5.bar(x + 0.15, p_wo_short,   w, bottom=p_wo_long,    label='W/o (short)',  color='lightskyblue')

        ax5.set_xticks(x); ax5.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=7)
        ax5.set_title("⑤ Probabilities (long/short stacked)", fontsize=11); ax5.set_ylim(0, prob_vmax * 1.1)
        if col == 0: ax5.legend(fontsize=7)

        # Row 5: Diff
        ax6 = fig.add_subplot(gs[5, col])
        colors = ['green' if v >= 0 else 'red' for v in data['movable_probs_diff']]
        ax6.bar(x, data['movable_probs_diff'], color=colors)
        ax6.set_xticks(x); ax6.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=7)
        ax6.set_title("⑥ Prob Diff", fontsize=11); ax6.axhline(0, color='k', linewidth=0.5)

    # heatmap凡例（colorbar）を行ごとに1本ずつ追加（列間でスケール統一済み）
    if last_im_alpha is not None and len(row_alpha_axes) > 0:
        fig.colorbar(last_im_alpha, ax=row_alpha_axes, fraction=0.02, pad=0.01)
    if last_im_alpha_wo is not None and len(row_alpha_wo_axes) > 0:
        fig.colorbar(last_im_alpha_wo, ax=row_alpha_wo_axes, fraction=0.02, pad=0.01)
    if last_im_W is not None and len(row_W_axes) > 0:
        fig.colorbar(last_im_W, ax=row_W_axes, fraction=0.02, pad=0.01)
    if last_im_Wdiff is not None and len(row_Wdiff_axes) > 0:
        fig.colorbar(last_im_Wdiff, ax=row_Wdiff_axes, fraction=0.02, pad=0.01)


    # --- export probability decomposition used in Row ⑤ (stacked bar) ---
    if len(prob_rows) > 0:
        csv_path = os.path.join(out_dir, f"{scenario['name']}_prob_decomp_{tag}.csv")
        fieldnames = list(prob_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(prob_rows)
        print(f"Saved: {csv_path}")

        txt_path = os.path.join(out_dir, f"{scenario['name']}_prob_decomp_{tag}.txt")
        with open(txt_path, "w") as f:
            # group by step for readability
            prob_rows_sorted = sorted(prob_rows, key=lambda r: (r["step"], r["token_id"]))
            cur_step = None
            for r in prob_rows_sorted:
                if cur_step != r["step"]:
                    cur_step = r["step"]
                    f.write(f"\n=== {r['scenario']} | {r['tag']} | Step {cur_step} ===\n")
                f.write(
                    f"{r['token_label']} (id={r['token_id']}): "
                    f"p_with={r['p_with']:.6f}, p_wo={r['p_without']:.6f}, "
                    f"with[long={r['long_contrib_with']:+.6e}, short={r['short_contrib_with']:+.6e}, total={r['total_contrib_with']:+.6e}, "
                    f"ratio={r['with_long_ratio']:+.6f}, p_long={r['p_with_long']:+.6f}, p_short={r['p_with_short']:+.6f}], "
                    f"wo[long={r['long_contrib_without']:+.6e}, short={r['short_contrib_without']:+.6e}, total={r['total_contrib_without']:+.6e}, "
                    f"ratio={r['without_long_ratio']:+.6f}, p_long={r['p_without_long']:+.6f}, p_short={r['p_without_short']:+.6f}]\n"
                )
        print(f"Saved: {txt_path}")

    save_path = os.path.join(out_dir, f"{scenario['name']}_rollout_analysis_{tag}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def _make_eigen_2d_pairs(eigvals, max_pairs=None):
    """|λ|の大きい順（=入力の順）で、
    - 複素固有値は共役ペアを1組として確保（回転の2次元表現）
    - 実固有値は未使用の実同士を順にペアリング
    を返す。

    Returns
    -------
    pairs : list[dict]
        dictは以下を持つ：
        - kind: 'complex' or 'real'
        - i: 主インデックス
        - j: 相方インデックス（complexは共役側、realは次の実）
    """
    n = len(eigvals)
    used = np.zeros(n, dtype=bool)
    pairs = []

    def _is_real(ev, tol=1e-12):
        return abs(ev.imag) < tol

    # 1) 複素ペアを優先的に確保（|λ|順）
    for i in range(n):
        if used[i]:
            continue
        ev = eigvals[i]
        if _is_real(ev):
            continue
        # 共役を探す
        conj_ev = np.conj(ev)
        j = None
        for k in range(i + 1, n):
            if used[k]:
                continue
            if abs(eigvals[k] - conj_ev) < 1e-8:
                j = k
                break
        if j is None:
            # 共役が見つからない場合はスキップ（理論上起きにくいが保険）
            continue
        used[i] = True
        used[j] = True
        pairs.append({'kind': 'complex', 'i': i, 'j': j})
        if max_pairs is not None and len(pairs) >= max_pairs:
            return pairs

    # 2) 残りの実固有値を順にペアリング
    real_indices = [i for i in range(n) if (not used[i]) and _is_real(eigvals[i])]
    for k in range(0, len(real_indices) - 1, 2):
        i = real_indices[k]
        j = real_indices[k + 1]
        used[i] = True
        used[j] = True
        pairs.append({'kind': 'real', 'i': i, 'j': j})
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
    return pairs


def visualize_koopman_biplot_grid(analyzer, step_data, scenario, tokenizer, out_dir):
    """Koopman Biplotを、全固有次元を2次元×8枚（=1枚画像）にまとめて出力。

    - |λ|の大きい順（analyzer.eigvalsの順）
    - 複素共役ペアは1枚の2次元プロット（Re/Imで回転を表現）
    - 実固有値は実同士をペアにして2次元プロット
    """
    z_dim = analyzer.z_dim
    traj_alpha_with = np.array([d['alpha_with'] for d in step_data]).reshape(len(step_data), z_dim)
    traj_alpha_without = np.array([d['alpha_without'] for d in step_data]).reshape(len(step_data), z_dim)

    pairs = _make_eigen_2d_pairs(analyzer.eigvals, max_pairs=8)
    if len(pairs) == 0:
        print("No eigen-pairs found for biplot grid.")
        return

    # 3. W (トークン重みベクトル) は従来通り、生成トークン + 初期のmovable を対象
    generated_tokens = set([d['next_token_with'] for d in step_data])
    initial_movables = set(step_data[0]['movable_tokens'])
    target_tokens = sorted(list(generated_tokens.union(initial_movables)))
    W_modal = analyzer.W_modal

    fig, axes = plt.subplots(2, 4, figsize=(4 * 6.5, 2 * 6.5))
    axes = np.asarray(axes).reshape(2, 4)
    fig.suptitle(f"Koopman Biplot Grid: Trajectory & Token Weights\nScenario: {scenario['name']}", fontsize=16)

    for p_idx in range(8):
        r = p_idx // 4
        c = p_idx % 4
        ax = axes[r, c]
        if p_idx >= len(pairs):
            ax.axis('off')
            continue

        p = pairs[p_idx]
        i = p['i']
        j = p['j']
        ev_i = analyzer.eigvals[i]
        ev_j = analyzer.eigvals[j]

        if p['kind'] == 'complex':
            # 2次元表現: alpha_i の (Re, Im)
            x_with = traj_alpha_with[:, i].real
            y_with = traj_alpha_with[:, i].imag
            x_wo = traj_alpha_without[:, i].real
            y_wo = traj_alpha_without[:, i].imag
            # W: 対応するモードiの (Re, Im)
            W_x = W_modal[target_tokens, i].real
            W_y = W_modal[target_tokens, i].imag
            xlabel = f"Re(α{i}) (λ={ev_i.real:.2f}{ev_i.imag:+.2f}j)"
            ylabel = f"Im(α{i}) (λ={ev_j.real:.2f}{ev_j.imag:+.2f}j)"
        else:
            # 実ペア: (alpha_i.real, alpha_j.real)
            x_with = traj_alpha_with[:, i].real
            y_with = traj_alpha_with[:, j].real
            x_wo = traj_alpha_without[:, i].real
            y_wo = traj_alpha_without[:, j].real
            W_x = W_modal[target_tokens, i].real
            W_y = W_modal[target_tokens, j].real
            xlabel = f"Mode {i} (λ={ev_i.real:.2f}{ev_i.imag:+.2f}j)"
            ylabel = f"Mode {j} (λ={ev_j.real:.2f}{ev_j.imag:+.2f}j)"

        # 1) z軌跡 (With / W/o)
        ax.plot(x_with, y_with, 'o-', color='steelblue', label='Trajectory (With)', markersize=3, alpha=0.7)
        ax.plot(x_with[0], y_with[0], 'D', color='green', markersize=6, label='Start')
        ax.plot(x_with[-1], y_with[-1], 'X', color='red', markersize=6, label='End')
        ax.plot(x_wo, y_wo, '--', color='gray', alpha=0.4, label='Trajectory (W/o)')

        # 2) Wベクトル
        traj_max = np.max(np.abs(np.concatenate([x_with, y_with, x_wo, y_wo])))
        W_max = np.max(np.abs(np.concatenate([W_x, W_y]))) if len(target_tokens) > 0 else 0.0
        if traj_max < 1e-12:
            traj_max = 1.0
        if W_max < 1e-12:
            W_max = 1.0
        scale_factor = (traj_max / W_max) * 0.8

        for t_idx, token in enumerate(target_tokens):
            vec_x = W_x[t_idx]
            vec_y = W_y[t_idx]
            ax.arrow(0, 0, vec_x * scale_factor, vec_y * scale_factor,
                     color='coral', alpha=0.35, head_width=traj_max * 0.03, length_includes_head=True)
            ax.text(vec_x * scale_factor * 1.05, vec_y * scale_factor * 1.05, get_token_label(token, tokenizer),
                    color='darkred', fontsize=7, fontweight='bold')

        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)

        # 凡例は左上の1つだけ（従来のスタイルを維持）
        if p_idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_path = os.path.join(out_dir, f"{scenario['name']}_biplot_grid.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved biplot grid: {save_path}")


# =========================================================
#  Model / Grid Helpers
# =========================================================

def _parse_csv_strs(value: str | None) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def _parse_csv_ints(value: str | None) -> list[int]:
    return [int(x) for x in _parse_csv_strs(value)]


def resolve_model_path(train_out_dir: str | Path) -> str:
    """train が書いた checkpoint.json を優先し、なければ最新 .pth にフォールバック。"""
    train_out_dir = Path(train_out_dir)
    manifest_path = train_out_dir / CHECKPOINT_MANIFEST
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        model_path = data.get("model_path")
        if model_path and os.path.exists(model_path):
            return model_path
        raise FileNotFoundError(
            f"checkpoint.json に記載の model_path が存在しません: {model_path}"
        )

    cands = sorted(train_out_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"No .pth found in {train_out_dir}")
    return str(cands[-1])


def default_out_dir(model_path: str | Path) -> str:
    p = Path(model_path)
    if p.parent.name == "train":
        return str(p.parent.parent / "scen2")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(p.parent / f"scen2_{ts}")


def load_model(model_path: str | Path, device: torch.device, base_N: int) -> tuple[KoopmanRoutesFormer, dict]:
    """チェックポイントから KoopmanRoutesFormer を復元（encoder_type / max_prefix_len 対応）。"""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    c = ckpt.get("config", {})
    h = _infer_model_hparams(c, state_dict, base_N)

    max_prefix_len = int(
        c.get("max_prefix_len", c.get("fixed_prefix_length", len(SCENARIOS[0]["prefix"])))
    )
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
    ).to(device)
    print(
        "Resolved model config: "
        f"z_dim={h['z_dim']}, d_model={h['d_model']}, num_layers={h['num_layers']}, "
        f"nhead={h['nhead']}, encoder_type={encoder_type}, max_prefix_len={max_prefix_len}"
    )

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, h


def setup_network(adj_path: str | Path):
    adj_matrix = torch.load(adj_path, weights_only=True)
    if adj_matrix.shape[0] == 38:
        base_N = 19
    else:
        base_N = int(adj_matrix.shape[0])

    expanded_adj = expand_adjacency_matrix(adj_matrix)
    dummy_feat = torch.zeros((len(adj_matrix), 1))
    node_features = torch.cat([dummy_feat, dummy_feat], dim=0)
    network = Network(expanded_adj, node_features)
    tokenizer = Tokenization(network)
    return network, tokenizer, expanded_adj, base_N


def iter_grid_experiments(
    session_root: str | Path,
    grid_types: list[str] | None = None,
    model_types: list[str] | None = None,
    z_dims: list[int] | None = None,
):
    """run_experiment_grid.py の session 配下から scen2 対象実験を列挙する。"""
    session_root = Path(session_root)
    if not session_root.exists():
        raise FileNotFoundError(f"session root が存在しません: {session_root}")

    for exp_dir in sorted(session_root.iterdir()):
        if not exp_dir.is_dir():
            continue

        manifest_path = exp_dir / "experiment.json"
        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            hp = manifest.get("hyperparams", {})
            model_path = manifest.get("model_path")
            grid_types_in = manifest.get("grid_types", [])
            model_type = hp.get("model_type")
            z_dim = hp.get("z_dim")
        else:
            train_dir = exp_dir / "train"
            if not train_dir.exists():
                continue
            model_path = resolve_model_path(train_dir)
            grid_types_in = []
            model_type = None
            z_dim = None
            manifest = {"experiment": exp_dir.name}

        if not model_path or not os.path.exists(model_path):
            print(f"[SKIP] model_path なし: {exp_dir.name}")
            continue

        if grid_types and not any(gt in grid_types for gt in grid_types_in):
            # experiment.json が無い古い run は grid_type フィルタをスキップ
            if grid_types_in:
                continue

        if model_types and model_type is not None and model_type not in model_types:
            continue
        if z_dims and z_dim is not None and int(z_dim) not in z_dims:
            continue

        yield {
            "exp_dir": exp_dir,
            "experiment": manifest.get("experiment", exp_dir.name),
            "model_path": model_path,
            "model_type": model_type,
            "z_dim": z_dim,
            "grid_types": grid_types_in,
        }


def run_analysis(
    model_path: str | Path,
    out_dir: str | Path,
    adj_path: str | Path,
    num_rollout_steps: int,
    device: torch.device,
    scenarios: list[dict] | None = None,
) -> str:
    """1モデル分の scen2 解析を実行する。"""
    model_path = str(model_path)
    out_dir = str(out_dir)
    scenarios = scenarios or SCENARIOS

    print("=" * 60)
    print("Koopman Analysis Pipeline (Original Model)")
    print(f"Model: {model_path}")
    print(f"Output: {out_dir}")
    print("=" * 60)

    os.makedirs(out_dir, exist_ok=True)

    print("\n1. Loading Data...")
    network, tokenizer, expanded_adj, base_N = setup_network(adj_path)

    print(f"\n2. Loading Model: {model_path}")
    model, _ = load_model(model_path, device, base_N)

    analyzer = KoopmanEigenAnalyzer(model)
    analyzer.plot_eigenvalues(os.path.join(out_dir, "eigenvalues.png"))

    print("\n3. Running Scenarios...")
    for scenario in scenarios:
        greedy_rollout_with_analysis(
            model, analyzer, tokenizer, network, expanded_adj,
            scenario, num_rollout_steps, device, out_dir
        )

    print("\n" + "=" * 60)
    print("All done.")
    print(f"Output: {out_dir}")
    print("=" * 60)
    return out_dir


# =========================================================
#  Main Pipeline
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Koopman scen2 解析（単体モデル or run_experiment_grid セッション一括）"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="単体実行時の学習済み .pth パス",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="出力先（未指定時: exp_dir/scen2 または model 隣の scen2_*）",
    )
    parser.add_argument(
        "--session-root",
        type=str,
        default=None,
        help="run_experiment_grid.py の1セッション出力ディレクトリ（一括実行）",
    )
    parser.add_argument(
        "--grid-type",
        type=str,
        default=None,
        help="フィルタ: カンマ区切り (例: model_x_zdim,model_x_prefix)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="フィルタ: カンマ区切り (例: transformer,lstm,mlp_flat)",
    )
    parser.add_argument(
        "--z-dims",
        type=str,
        default=None,
        help="フィルタ: カンマ区切り z_dim (例: 16,32,64,128)",
    )
    parser.add_argument(
        "--adj-path",
        type=str,
        default=DEFAULT_ADJ_PATH,
        help="隣接行列 .pt のパス",
    )
    parser.add_argument(
        "--num-rollout-steps",
        type=int,
        default=DEFAULT_NUM_ROLLOUT_STEPS,
        help="Greedy ロールアウトステップ数",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="一括実行で1件でも失敗したら即停止",
    )
    args = parser.parse_args()

    if not args.model_path and not args.session_root:
        parser.error("--model-path または --session-root のいずれかを指定してください")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_types = _parse_csv_strs(args.grid_type)
    model_types = _parse_csv_strs(args.model_type)
    z_dims = _parse_csv_ints(args.z_dims)

    if args.session_root:
        success = 0
        skipped = 0
        targets = list(
            iter_grid_experiments(
                session_root=args.session_root,
                grid_types=grid_types or None,
                model_types=model_types or None,
                z_dims=z_dims or None,
            )
        )
        if not targets:
            raise RuntimeError(f"対象実験が見つかりません: {args.session_root}")

        print(f"[INFO] session_root: {args.session_root}")
        print(f"[INFO] targets: {len(targets)}")

        for item in targets:
            exp_name = item["experiment"]
            model_path = item["model_path"]
            out_dir = args.out_dir or str(item["exp_dir"] / "scen2")
            print(f"\n[RUN] {exp_name}")
            print(f"      model_path={model_path}")
            print(f"      out_dir={out_dir}")
            try:
                run_analysis(
                    model_path=model_path,
                    out_dir=out_dir,
                    adj_path=args.adj_path,
                    num_rollout_steps=args.num_rollout_steps,
                    device=device,
                )
                success += 1
            except Exception as e:
                msg = f"[FAIL] {exp_name}: {e}"
                if args.strict:
                    raise RuntimeError(msg) from e
                print(msg)
                skipped += 1

        print(f"\n完了: success={success}, skipped_or_failed={skipped}")
        return

    model_path = args.model_path
    out_dir = args.out_dir or default_out_dir(model_path)
    run_analysis(
        model_path=model_path,
        out_dir=out_dir,
        adj_path=args.adj_path,
        num_rollout_steps=args.num_rollout_steps,
        device=device,
    )


if __name__ == "__main__":
    main()
