"""
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

import torch
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from datetime import datetime
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
    {
        "name": "2,21,21,5,24",
        "prefix": [2,21,21,5,24],  # 初期prefix
        "time": 20240101,
        "agent_id": 0,
        "holiday": 1,
        "time_zone": 0,
        "plaza_node_tokens": [2, 21],  # 広場として扱うノード
    },
    {
        "name": "11,30,8,8,9",
        "prefix": [11,30,8,8,9],  # 初期prefix
        "time": 20240101,
        "agent_id": 0,
        "holiday": 1,
        "time_zone": 0,
        "plaza_node_tokens": [11, 30],  # 広場として扱うノード
    },
    {
        "name": "4,4,23,23,11",
        "prefix": [4,4,23,23,11],  # 初期prefix
        "time": 20240101,
        "agent_id": 0,
        "holiday": 1,
        "time_zone": 0,
        "plaza_node_tokens": [11, 30],  # 広場として扱うノード
    },
    {
        "name": "6,6,25,14,33",
        "prefix": [6,6,25,14,33],
        "time": 20240101,
        "agent_id": 0,
        "holiday": 1,
        "time_zone": 0,
        "plaza_node_tokens": [14, 33],
    },
    {
        "name": "6,6,25,14,33",
        "prefix": [6,6,25,14,33],
        "time": 20240101,
        "agent_id": 0,
        "holiday": 1,              
        "time_zone": 0,            
        "plaza_node_tokens": [14, 33]
    },
    {
        "name": "16,35,35,14,33",
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

# パス設定（環境に合わせて変更してください）
MODEL_PATH = "/home/mizutani/projects/RF/runs/20260202_054426/model_weights_20260202_054426.pth"
ADJ_PATH = "/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt"
DATA_PATH = "/home/mizutani/projects/RF/data/input_real_m5.npz"

# 出力先
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(os.path.dirname(MODEL_PATH), f"scen2_{RUN_ID}")
os.makedirs(OUT_DIR, exist_ok=True)

# デバイス
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 解析パラメータ
NUM_ROLLOUT_STEPS = 15  # 何ステップ生成するか

def _normalize_prefix_time_to_12digits(t: int) -> int:
    """
    model.forward_rollout は prefix_times を 12桁想定 (例: 202409290900) で date/hour を取る。
    scen2 側でも揃える。8桁 (YYYYMMDD) の場合は 12:00 扱いで YYYYMMDD1200 にする（暫定）。
    """
    t = int(t)
    if t < 10**8:
        raise ValueError(f"scenario['time'] looks too short: {t}")
    # 8桁: YYYYMMDD
    if t < 10**10:
        return t * 10000 + 1200
    # 10〜12桁はそのまま（あなたの運用に合わせる）
    return t

def _build_fixed_context_from_prefix_time(prefix_time_int: int, base_N: int, device):
    """
    DKP_RF.forward_rollout の holiday/night/event 判定を scen2 側に最小限コピー。
    (HOLIDAYS, night range, EVENTS は forward_rollout と同じ)。
    """
    t_int = torch.tensor([prefix_time_int], device=device, dtype=torch.long)  # [1]
    date_int = t_int // 10000
    hour = (t_int // 100) % 100

    HOLIDAYS = {20240928, 20240929, 20251122, 20251123}
    fixed_h = torch.isin(date_int, torch.tensor(list(HOLIDAYS), device=device)).long()  # [1]

    night_start = 19
    night_end = 2
    fixed_tz = ((hour >= night_start) | (hour < night_end)).long()  # [1]

    EVENTS = [
        (20240929, 9, 16, [14]),
        (20251122, 10, 19, [2, 11]),
        (20251123, 10, 16, [2]),
    ]
    event_node_mask = torch.zeros((1, base_N), device=device, dtype=torch.bool)
    for ev_date, ev_start, ev_end, ev_nodes in EVENTS:
        cond = (date_int == ev_date) & (ev_start <= hour) & (hour < ev_end)
        if cond.any():
            node_idx = torch.tensor(ev_nodes, device=device, dtype=torch.long)
            event_node_mask[cond] = event_node_mask[cond].clone()
            event_node_mask[cond][:, node_idx] = True

    return fixed_h[0], fixed_tz[0], event_node_mask[0]  # scalar, scalar, [base_N] bool

@torch.no_grad()
def _step_dynamics_with_Bu(model, z_t: torch.Tensor, cur_tok: int, fixed_h, fixed_tz, event_node_mask_1d,
                           force_event_zero: bool = False):
    """
    z_{t+1} = z_t A^T + 0.1 * Bu,  Bu = u B^T
    u = [control_token_embedding(cur_tok), holiday_u(h), timezone_u(tz), event_u(e(node))]
    force_event_zero=True なら e_k=0 にして「広場(イベント)埋め込み無し」の Bu を作る。
    """
    device = z_t.device
    base_N = model.base_N

    cur_tok_t = torch.tensor([cur_tok], device=device, dtype=torch.long).clamp(min=0, max=base_N*2-1)  # [1]
    node_id = (cur_tok_t % base_N).item()

    if force_event_zero:
        e_k = torch.zeros((1,), device=device, dtype=torch.long)  # [1]
    else:
        e_k = event_node_mask_1d[node_id].long().view(1)          # [1]

    u_tok = model.control_token_embedding(cur_tok_t)  # [1, u_tok_dim]
    u_h  = model.holiday_embedding_u(torch.tensor([int(fixed_h)], device=device, dtype=torch.long))
    u_tz = model.time_zone_embedding_u(torch.tensor([int(fixed_tz)], device=device, dtype=torch.long))
    u_e  = model.event_embedding_u(e_k)

    u = torch.cat([u_tok, u_h, u_tz, u_e], dim=-1)  # [1, u_dim]
    Bu = u @ model.B.T                               # [1, z_dim]

    z_auto = z_t @ model.A.T
    z_next = z_auto + 0.1 * Bu
    return z_next, z_auto, Bu

# =========================================================
#  Utility Functions
# =========================================================

def get_movable_tokens(current_node, adj_matrix, pad_token_id=38, end_token_id=39):
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
    """トークンIDからラベル文字列を取得"""
    if 0 <= token_id <= 18:
        return f"M{token_id}"
    elif 19 <= token_id <= 37:
        return f"S{token_id-19}"
    elif token_id in tokenizer.SPECIAL_TOKENS.values():
        for k, v in tokenizer.SPECIAL_TOKENS.items():
            if v == token_id:
                return k
        return f"?{token_id}"
    else:
        return f"T{token_id}"


# =========================================================
#  Eigen Analyzer Class
# =========================================================

class KoopmanEigenAnalyzer:
    """Koopman演算子Aの固有値分解と解析"""
    def __init__(self, model):
        self.model = model
        self.z_dim = model.z_dim
        
        A_np = model.A.detach().cpu().numpy()
        eigvals, eigvecs = scipy.linalg.eig(A_np)
        
        sort_idx = np.argsort(np.abs(eigvals))[::-1]
        self.eigvals = eigvals[sort_idx]
        self.V = eigvecs[:, sort_idx]
        self.V_inv = scipy.linalg.inv(self.V)
        
        self.W = model.to_logits.weight.detach().cpu().numpy()
        self.b = model.to_logits.bias.detach().cpu().numpy()
        self.W_modal = self.W @ self.V
        
        print(f"Eigenvalue decomposition complete: Max |λ| = {np.abs(self.eigvals).max():.4f}")
    
    def transform_to_eigenspace(self, z):
        if z.ndim == 1:
            return self.V_inv @ z
        else:
            return (self.V_inv @ z.T).T
    
    def plot_eigenvalues(self, save_path):
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
    tokens = torch.tensor([prefix], dtype=torch.long).to(device)
    stay_counts = tokenizer.calculate_stay_counts(tokens).to(device)
    agent_ids = torch.tensor([agent_id], dtype=torch.long).to(device)
    holidays = torch.tensor([[holiday] * seq_len], dtype=torch.long).to(device)
    time_zones = torch.tensor([[time_zone] * seq_len], dtype=torch.long).to(device)
    
    events = torch.zeros((1, seq_len), dtype=torch.long).to(device)
    if use_plaza:
        for pos, tok in enumerate(prefix):
            if tok in plaza_tokens:
                events[0, pos] = 1
    
    with torch.no_grad():
        z_0, _ = model.encode_prefix(
            tokens, stay_counts, agent_ids,
            holidays, time_zones, events
        )
    return z_0[0]


def greedy_rollout_with_analysis(model, analyzer, tokenizer, network, adj_matrix,
                                  scenario, num_steps, device, out_dir):
    """Greedyロールアウトしながら解析"""
    print(f"\n{'='*60}")
    print(f"Analyzing Scenario: {scenario['name']}")
    print(f"{'='*60}")
    
    prefix = scenario['prefix']
    agent_id = scenario['agent_id']
    holiday = scenario['holiday']
    time_zone = scenario['time_zone']
    plaza_tokens = scenario.get('plaza_node_tokens', [])
    
    # 初期エンコード
    z_with = encode_prefix_with_plaza(model, tokenizer, prefix, agent_id, holiday, time_zone, plaza_tokens, True, device).cpu().numpy()
    z_without = encode_prefix_with_plaza(model, tokenizer, prefix, agent_id, holiday, time_zone, plaza_tokens, False, device).cpu().numpy()
    
    z_with = np.asarray(z_with).reshape(-1)
    z_without = np.asarray(z_without).reshape(-1)
    
    step_data = []
    current_node = prefix[-1] % 19
    generated_with = []
    generated_without = []
    
    # =====================================================
    # prefix と同じ文脈（学習時と完全一致）
    # =====================================================
    device = next(model.parameters()).device

    # encode_prefix_with_plaza で使ったものと同一
    fixed_h = torch.tensor(scenario['holiday'], device=device)
    fixed_tz = torch.tensor(scenario['time_zone'], device=device)

    # plaza_tokens から event_node_mask を作る（学習時の events と同義）
    event_node_mask_1d = torch.zeros(model.base_N, device=device, dtype=torch.bool)
    for tok in plaza_tokens:
        if 0 <= tok < model.base_N:
            event_node_mask_1d[tok] = True
        elif model.base_N <= tok < model.base_N * 2:
            event_node_mask_1d[tok - model.base_N] = True

    # 追加：現在トークンを with / without で持つ（初期は prefix の最後）
    cur_tok_with = int(prefix[-1])
    cur_tok_without = int(prefix[-1])

    for step in range(num_steps):
        print(f"\n--- Step {step+1}/{num_steps} ---")
        
        movable_tokens = get_movable_tokens(current_node, adj_matrix)
        
        # ===== Dynamics (Bu込み) =====
        z_t_with = torch.tensor(z_with, device=device, dtype=torch.float32).view(1, -1)
        z_t_wo   = torch.tensor(z_without, device=device, dtype=torch.float32).view(1, -1)

        # with: イベント埋め込みあり
        z_next_with_t, z_auto_with_t, Bu_with_t = _step_dynamics_with_Bu(
            model, z_t_with, cur_tok_with, fixed_h, fixed_tz, event_node_mask_1d, force_event_zero=False
        )

        # without: イベント埋め込みを強制オフ（＝広場埋め込み無し）
        z_next_wo_t, z_auto_wo_t, Bu_wo_t = _step_dynamics_with_Bu(
            model, z_t_wo, cur_tok_without, fixed_h, fixed_tz, event_node_mask_1d, force_event_zero=True
        )

        z_next_with = z_next_with_t.squeeze(0).cpu().numpy()
        z_next_without = z_next_wo_t.squeeze(0).cpu().numpy()

        Bu_with = Bu_with_t.squeeze(0).cpu().numpy()
        Bu_without = Bu_wo_t.squeeze(0).cpu().numpy()

        # 固有空間
        alpha_with = analyzer.V_inv @ z_next_with
        alpha_without = analyzer.V_inv @ z_next_without

        # ★ Bu の固有空間成分（総量）
        alpha_Bu_with = analyzer.V_inv @ (0.1 * Bu_with)
        alpha_Bu_without = analyzer.V_inv @ (0.1 * Bu_without)

        # ★ ②用：イベント(広場)埋め込みによる Bu 差分
        dBu_plaza = 0.1 * (Bu_with - Bu_without)
        alpha_dBu_plaza = analyzer.V_inv @ dBu_plaza

        # ===== logits/prob（現行通りの W@z + b でOK）=====
        logits_with = analyzer.W @ z_next_with + analyzer.b
        logits_without = analyzer.W @ z_next_without + analyzer.b

        probs_with = np.exp(logits_with) / np.exp(logits_with).sum()
        probs_without = np.exp(logits_without) / np.exp(logits_without).sum()

        movable_probs_with = probs_with[movable_tokens]
        movable_probs_without = probs_without[movable_tokens]
        movable_probs_diff = movable_probs_with - movable_probs_without

        next_token_with = movable_tokens[np.argmax(movable_probs_with)]
        next_token_without = movable_tokens[np.argmax(movable_probs_without)]

        generated_with.append(next_token_with)
        generated_without.append(next_token_without)

        step_data.append({
            'step': step,
            'z_with': z_next_with.copy(),
            'z_without': z_next_without.copy(),
            'alpha_with': alpha_with.copy(),
            'alpha_without': alpha_without.copy(),

            # ★追加
            'Bu_with': Bu_with.copy(),
            'Bu_without': Bu_without.copy(),
            'alpha_Bu_with': alpha_Bu_with.copy(),
            'alpha_Bu_without': alpha_Bu_without.copy(),
            'alpha_dBu_plaza': alpha_dBu_plaza.copy(),   # ★追加（②の本体）


            'movable_tokens': movable_tokens.copy(),
            'movable_probs_with': movable_probs_with.copy(),
            'movable_probs_without': movable_probs_without.copy(),
            'movable_probs_diff': movable_probs_diff.copy(),
            'next_token_with': next_token_with,
            'next_token_without': next_token_without,
        })

        # 次ステップの状態更新（既存ロジックを踏襲）
        z_with = z_next_with
        z_without = z_next_without

        cur_tok_with = int(next_token_with)
        cur_tok_without = int(next_token_without)
        
        if next_token_with < 19:
            current_node = next_token_with
        elif 19 <= next_token_with <= 37:
            current_node = next_token_with - 19
        else:
            print(f"  End token selected, stopping.")
            break
    
    # 既存のロールアウト可視化
    visualize_rollout_analysis(analyzer, step_data, scenario, tokenizer, out_dir)

    # ★更新: Koopman Biplot (全モードを2次元ペアでまとめて可視化)
    visualize_koopman_biplot_grid(analyzer, step_data, scenario, tokenizer, out_dir)
    
    print(f"\nGenerated sequence (with plaza): {prefix} -> {generated_with}")


def visualize_rollout_analysis(analyzer, step_data, scenario, tokenizer, out_dir):
    """
    Rollout 可視化（最終版）
    Row0: ① Bu 総量による token 寄与
    Row1: ② u の plaza(event) 埋め込みによる ΔBu 寄与
    Row2: ③ Mode×Token (with)
    Row3: ④ ΔContrib (with - without)  ※復活
    Row4: ⑤ Probabilities
    Row5: ⑥ Prob Diff
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    num_steps = len(step_data)
    z_dim = analyzer.z_dim

    fig = plt.figure(figsize=(num_steps * 6, 28))
    gs = gridspec.GridSpec(
        6, num_steps, figure=fig,
        hspace=0.45, wspace=0.25,
        top=0.95, bottom=0.04, left=0.05, right=0.98
    )

    fig.suptitle(
        f"Koopman Rollout Analysis (u-event ON/OFF): {scenario['name']}",
        fontsize=24, weight='bold'
    )

    # ========= スケール計算 =========
    all_W_vals = []
    all_Bu_vals = []
    all_dBu_vals = []
    all_diff_contrib_vals = []
    all_prob_vals = []
    all_prob_diff_vals = []

    for d in step_data:
        movable = d['movable_tokens']

        alpha_with = np.real(np.asarray(d['alpha_with']))
        alpha_wo = np.real(np.asarray(d['alpha_without']))
        delta_alpha = alpha_with - alpha_wo

        all_W_vals.append(analyzer.W_modal[movable, :] * alpha_with.reshape(1, -1))
        all_diff_contrib_vals.append(
            analyzer.W_modal[movable, :] * delta_alpha.reshape(1, -1)
        )

        all_Bu_vals.append(
            analyzer.W_modal[movable, :] *
            np.real(d['alpha_Bu_with']).reshape(1, -1)
        )

        all_dBu_vals.append(
            analyzer.W_modal[movable, :] *
            np.real(d['alpha_dBu_plaza']).reshape(1, -1)
        )

        all_prob_vals.extend([d['movable_probs_with'], d['movable_probs_without']])
        all_prob_diff_vals.append(d['movable_probs_diff'])

    W_vmax = max(np.abs(np.concatenate(all_W_vals)).max(), 1e-6)
    Bu_vmax = max(np.abs(np.concatenate(all_Bu_vals)).max(), 1e-6)
    dBu_vmax = max(np.abs(np.concatenate(all_dBu_vals)).max(), 1e-6)
    diff_contrib_vmax = max(np.abs(np.concatenate(all_diff_contrib_vals)).max(), 1e-6)
    prob_vmax = max(np.concatenate(all_prob_vals).max(), 1e-6)
    prob_diff_vmax = max(np.abs(np.concatenate(all_prob_diff_vals)).max(), 1e-6)

    eigenmode_labels = [
        f"{lam.real:.2f}{lam.imag:+.2f}j" for lam in analyzer.eigvals
    ]

    # colorbar 用
    row_axes = [[] for _ in range(6)]
    last_im = [None] * 6

    # ========= 各ステップ描画 =========
    for step_idx, d in enumerate(step_data):
        col = step_idx
        movable = d['movable_tokens']
        token_labels = [get_token_label(t, tokenizer) for t in movable]
        num_movable = len(movable)

        alpha_with = np.real(np.asarray(d['alpha_with']))
        alpha_wo = np.real(np.asarray(d['alpha_without']))
        delta_alpha = alpha_with - alpha_wo

        # ---------- Row0: Bu 総量 ----------
        ax = fig.add_subplot(gs[0, col])
        Bu_contrib = np.real(
            analyzer.W_modal[movable, :] * d['alpha_Bu_with'].reshape(1, -1)
        )
        im = ax.imshow(Bu_contrib.T, cmap='coolwarm', aspect='auto',
                       vmin=-Bu_vmax, vmax=Bu_vmax)
        ax.set_title(f"Step {step_idx+1}\n① Bu contrib", fontsize=11)
        ax.set_xticks(range(num_movable))
        ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(z_dim))
        ax.set_yticklabels(eigenmode_labels, fontsize=6)
        row_axes[0].append(ax)
        last_im[0] = im

        # ---------- Row1: ΔBu (plaza in u) ----------
        ax = fig.add_subplot(gs[1, col])
        dBu_contrib = np.real(
            analyzer.W_modal[movable, :] * d['alpha_dBu_plaza'].reshape(1, -1)
        )
        im = ax.imshow(dBu_contrib.T, cmap='coolwarm', aspect='auto',
                       vmin=-dBu_vmax, vmax=dBu_vmax)
        ax.set_title("② ΔBu from plaza-embedding in u", fontsize=11)
        ax.set_xticks(range(num_movable))
        ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticks([])
        row_axes[1].append(ax)
        last_im[1] = im

        # ---------- Row2: Mode×Token (with) ----------
        ax = fig.add_subplot(gs[2, col])
        W_contrib = np.real(
            analyzer.W_modal[movable, :] * alpha_with.reshape(1, -1)
        )
        im = ax.imshow(W_contrib.T, cmap='coolwarm', aspect='auto',
                       vmin=-W_vmax, vmax=W_vmax)
        ax.set_title("③ Mode × Token (with)", fontsize=11)
        ax.set_xticks(range(num_movable))
        ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(z_dim))
        ax.set_yticklabels(eigenmode_labels, fontsize=6)
        row_axes[2].append(ax)
        last_im[2] = im

        # ---------- Row3: ΔContrib (with - without) ----------
        ax = fig.add_subplot(gs[3, col])
        W_diff = np.real(
            analyzer.W_modal[movable, :] * delta_alpha.reshape(1, -1)
        )
        im = ax.imshow(W_diff.T, cmap='coolwarm', aspect='auto',
                       vmin=-diff_contrib_vmax, vmax=diff_contrib_vmax)
        ax.set_title("④ ΔContrib (with − without)", fontsize=11)
        ax.set_xticks(range(num_movable))
        ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticks([])
        row_axes[3].append(ax)
        last_im[3] = im

        # ---------- Row4: Probabilities ----------
        ax = fig.add_subplot(gs[4, col])
        x = np.arange(num_movable)
        ax.bar(x - 0.15, d['movable_probs_with'], 0.3, label='with', color='steelblue')
        ax.bar(x + 0.15, d['movable_probs_without'], 0.3, label='without', color='coral')
        ax.set_ylim(0, prob_vmax * 1.1)
        ax.set_xticks(x)
        ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=7)
        ax.set_title("⑤ Probabilities", fontsize=11)
        if col == 0:
            ax.legend(fontsize=8)

        # ---------- Row5: Prob Diff ----------
        ax = fig.add_subplot(gs[5, col])
        colors = ['green' if v >= 0 else 'red' for v in d['movable_probs_diff']]
        ax.bar(x, d['movable_probs_diff'], color=colors)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_ylim(-prob_diff_vmax * 1.1, prob_diff_vmax * 1.1)
        ax.set_xticks(x)
        ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=7)
        ax.set_title("⑥ Prob Diff", fontsize=11)

    # ========= colorbar =========
    for i in range(4):  # heatmap 行のみ
        if last_im[i] is not None and row_axes[i]:
            fig.colorbar(last_im[i], ax=row_axes[i], fraction=0.02, pad=0.01)

    save_path = os.path.join(out_dir, f"{scenario['name']}_rollout_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved rollout analysis: {save_path}")


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

    # step_data から Bu の alpha を集める（with 側）
    bu_alpha_with = np.array([d['alpha_Bu_with'] for d in step_data])  # [T, z_dim] (complex想定)
    mean_bu = bu_alpha_with.mean(axis=0)  # [z_dim]

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

        # 3) Bu 矢印（平均方向を1本）
        # if p['kind'] == 'complex':
        #     # (i,i+1) は複素ペア: 平面は (Re(α_i), Im(α_i)) として描いている
        #     bu_dx = mean_bu[i].real
        #     bu_dy = mean_bu[i].imag
        # else:
        #     bu_dx = mean_bu[i].real
        #     bu_dy = mean_bu[j].real

        # # スケールは軌跡スケールに合わせて控えめに
        # arrow_scale = 0.8
        # ax.arrow(0, 0, bu_dx * arrow_scale, bu_dy * arrow_scale,
        #         color='purple', alpha=0.7, head_width=traj_max * 0.03, length_includes_head=True)
        # ax.text(bu_dx * arrow_scale * 1.05, bu_dy * arrow_scale * 1.05, "Bu",
        #         color='purple', fontsize=8, fontweight='bold')


        # 凡例は左上の1つだけ（従来のスタイルを維持）
        if p_idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_path = os.path.join(out_dir, f"{scenario['name']}_biplot_grid.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved biplot grid: {save_path}")


# =========================================================
#  Main Pipeline
# =========================================================

def main():
    print("="*60)
    print("Koopman Analysis Pipeline (Original Model)")
    print("="*60)
    
    # データロード
    print("\n1. Loading Data...")
    adj_matrix = torch.load(ADJ_PATH, weights_only=True)
    if adj_matrix.shape[0] == 38:
        base_N = 19
    else:
        base_N = int(adj_matrix.shape[0])
    
    expanded_adj = expand_adjacency_matrix(adj_matrix)
    dummy_feat = torch.zeros((len(adj_matrix), 1))
    node_features = torch.cat([dummy_feat, dummy_feat], dim=0)
    network = Network(expanded_adj, node_features)
    tokenizer = Tokenization(network)
    
    # モデルロード
    print(f"\n2. Loading Model: {MODEL_PATH}")
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    state_dict = ckpt['model_state_dict']
    c = ckpt.get('config', {})
    
    # モデル初期化
    model = KoopmanRoutesFormer(
        vocab_size=c.get('vocab_size', 42),
        token_emb_dim=c.get('token_emb_dim', 64),
        d_model=c.get('d_model', 64),
        nhead=c.get('nhead', 4),
        num_layers=c.get('num_layers', 3),
        d_ff=c.get('d_ff', 128),
        z_dim=c.get('z_dim', 16),
        pad_token_id=38,
        base_N=base_N,
        num_agents=c.get('num_agents', 1),
        agent_emb_dim=c.get('agent_emb_dim', 16),
        max_stay_count=c.get('max_stay_count', 500),
        stay_emb_dim=c.get('stay_emb_dim', 16),
        holiday_emb_dim=c.get('holiday_emb_dim', 4),
        time_zone_emb_dim=c.get('time_zone_emb_dim', 4),
        event_emb_dim=c.get('event_emb_dim', 4),
    ).to(DEVICE)
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # 解析準備
    analyzer = KoopmanEigenAnalyzer(model)
    analyzer.plot_eigenvalues(os.path.join(OUT_DIR, "eigenvalues.png"))
    
    # シナリオ実行
    print("\n3. Running Scenarios...")
    for scenario in SCENARIOS:
        greedy_rollout_with_analysis(
            model, analyzer, tokenizer, network, expanded_adj,
            scenario, NUM_ROLLOUT_STEPS, DEVICE, OUT_DIR
        )
    
    print("\n" + "="*60)
    print("All done.")
    print(f"Output: {OUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()