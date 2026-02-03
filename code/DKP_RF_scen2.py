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
MODEL_PATH = "/home/mizutani/projects/RF/runs/20260127_014201/model_weights_20260127_014201.pth"
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
    
    A_np = analyzer.model.A.detach().cpu().numpy()
    
    for step in range(num_steps):
        print(f"\n--- Step {step+1}/{num_steps} ---")
        
        z_next_with = A_np @ z_with
        z_next_without = A_np @ z_without
        
        alpha_with = analyzer.transform_to_eigenspace(z_next_with)
        alpha_without = analyzer.transform_to_eigenspace(z_next_without)
        
        movable_tokens = get_movable_tokens(current_node, adj_matrix)
        
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
        
        print(f"  Current node: {current_node}")
        print(f"  Selected (with): {get_token_label(next_token_with, tokenizer)} (p={movable_probs_with.max():.4f})")
        
        step_data.append({
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
            'next_token_without': next_token_without,
        })
        
        z_with = z_next_with
        z_without = z_next_without
        
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
    """ロールアウト解析結果を可視化（5ステップ横並び）"""
    num_steps = len(step_data)
    z_dim = analyzer.z_dim
    
    fig = plt.figure(figsize=(num_steps * 6, 28))
    gs = gridspec.GridSpec(6, num_steps, figure=fig, hspace=0.35, wspace=0.25,
                          top=0.96, bottom=0.04, left=0.05, right=0.98)
    fig.suptitle(f"Koopman Mode Decomposition Analysis: {scenario['name']}", fontsize=24, weight='bold')
    
    # スケール計算用の収集
    all_alpha_vals = []
    all_W_vals = []
    all_prob_vals = []
    all_diff_vals = []
    for data in step_data:
        all_alpha_vals.extend([data['alpha_with'], data['alpha_without']])
        movable = data['movable_tokens']
        all_W_vals.append(analyzer.W_modal[movable, :])
        all_prob_vals.extend([data['movable_probs_with'], data['movable_probs_without']])
        all_diff_vals.append(data['movable_probs_diff'])
    
    alpha_vmax = max(np.abs(np.concatenate(all_alpha_vals)).max(), 1e-6)
    W_vmax = max(np.abs(np.concatenate(all_W_vals, axis=0)).max(), 1e-6)
    prob_vmax = max(np.concatenate(all_prob_vals).max(), 1e-6)
    diff_vmax = max(np.abs(np.concatenate(all_diff_vals)).max(), 1e-6)
    
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

    for step_idx, data in enumerate(step_data):
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
        ax3.set_title("③ Mode×Token (with)", fontsize=11)
        ax3.set_xticks(range(num_movable)); ax3.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=7)
        ax3.set_yticks(range(z_dim)); ax3.set_yticklabels(eigenmode_labels, fontsize=6)
        row_W_axes.append(ax3)
        last_im_W = im3

        # Row 3: W diff
        ax4 = fig.add_subplot(gs[3, col])
        W_diff = np.real(analyzer.W_modal[movable, :] * delta_alpha.reshape(1, -1))
        im4 = ax4.imshow(W_diff.T, cmap='coolwarm', aspect='auto', vmin=-diff_vmax, vmax=diff_vmax)
        ax4.set_title("④ ΔContrib (with-w/o)", fontsize=11)
        ax4.set_xticks(range(num_movable)); ax4.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=7)
        ax4.set_yticks([])
        row_Wdiff_axes.append(ax4)
        last_im_Wdiff = im4

        # Row 4: Prob
        ax5 = fig.add_subplot(gs[4, col])
        x = np.arange(num_movable)
        ax5.bar(x - 0.15, data['movable_probs_with'], 0.3, label='With', color='coral')
        ax5.bar(x + 0.15, data['movable_probs_without'], 0.3, label='W/o', color='steelblue')
        ax5.set_xticks(x); ax5.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=7)
        ax5.set_title("⑤ Probabilities", fontsize=11); ax5.set_ylim(0, prob_vmax * 1.1)
        if col == 0: ax5.legend()

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

    save_path = os.path.join(out_dir, f"{scenario['name']}_rollout_analysis.png")
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