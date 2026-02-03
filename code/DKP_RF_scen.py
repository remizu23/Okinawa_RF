"""
Koopman Mode Decomposition Analysis for DKP_RF (Combined Version)

統合された解析コード:
1. A行列の固有値分解と単位円上プロット
2. 各ステップでの潜在状態z_tを固有空間射影（16次元分8枚）
3. 重みベクトルの固有空間寄与分析
4. トークン選択確率と広場有無の差分可視化
5. Greedy生成による5ステップ分の横並び可視化
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

SCENARIOS = [
    {
        "name": "0,1,2,21",
        "prefix": [0, 1, 2, 21],
        "time": 20240101,
        "agent_id": 0,
        "holiday": 1,
        "time_zone": 0,
        "plaza_node_tokens": [2, 21],
    },
    {
        "name": "6,5,4,11",
        "prefix": [6, 5, 4, 11, 30],
        "time": 20240101,
        "agent_id": 0,
        "holiday": 1,
        "time_zone": 0,
        "plaza_node_tokens": [11, 30],
    },
    {
        "name": "1,5,6,14",
        "prefix": [1, 5, 6, 14, 33],
        "time": 20240101,
        "agent_id": 0,
        "holiday": 1,
        "time_zone": 0,
        "plaza_node_tokens": [14, 33]
    },
    {
        "name": "18,16,14",
        "prefix": [18, 16, 14, 33],
        "time": 20240101,
        "holiday": 1,
        "time_zone": 0,
        "agent_id": 0,
        "plaza_node_tokens": [14, 33]
    },
]

# パス設定
MODEL_PATH = "/home/mizutani/projects/RF/runs/20260127_014201/model_weights_20260127_014201.pth"
ADJ_PATH = "/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt"
DATA_PATH = "/home/mizutani/projects/RF/data/input_real_m5.npz"

# 出力先
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"/home/mizutani/projects/RF/runs/20260127_014201/scen_{RUN_ID}"
os.makedirs(OUT_DIR, exist_ok=True)

# デバイス
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 解析パラメータ
NUM_ROLLOUT_STEPS = 15  # 生成ステップ数（変更可能）


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
    
    for neighbor in neighbors:
        if 0 <= neighbor <= 18:
            movable_tokens.append(int(neighbor))
    
    if 0 <= current_node <= 18:
        stay_token = current_node + 19
        movable_tokens.append(stay_token)
    
    movable_tokens.append(end_token_id)
    return sorted(movable_tokens)


def get_token_label(token_id, tokenizer):
    """トークンIDをラベルに変換"""
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
    """Koopman行列の固有値分解とモード解析"""
    
    def __init__(self, model):
        self.model = model
        self.z_dim = model.z_dim
        
        # A行列の固有値分解
        A_np = model.A.detach().cpu().numpy()
        eigvals, eigvecs = scipy.linalg.eig(A_np)
        
        # 固有値の絶対値が大きい順にソート
        sort_idx = np.argsort(np.abs(eigvals))[::-1]
        self.eigvals = eigvals[sort_idx]
        self.V = eigvecs[:, sort_idx]
        self.V_inv = scipy.linalg.inv(self.V)
        
        # 重みベクトルを固有空間に射影
        self.W = model.to_logits.weight.detach().cpu().numpy()
        self.b = model.to_logits.bias.detach().cpu().numpy()
        self.W_modal = self.W @ self.V
        
        print(f"Eigenvalue decomposition complete: Max |λ| = {np.abs(self.eigvals).max():.4f}")
    
    def transform_to_eigenspace(self, z):
        """潜在状態を固有空間に変換"""
        if z.ndim == 1:
            return self.V_inv @ z
        else:
            return (self.V_inv @ z.T).T
    
    def plot_eigenvalues(self, save_path):
        """固有値を単位円上にプロット"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 単位円
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1, alpha=0.3)
        
        # 固有値
        ax.scatter(self.eigvals.real, self.eigvals.imag,
                  c=np.arange(len(self.eigvals)), cmap='coolwarm',
                  s=100, edgecolors='black', linewidth=1.5, zorder=5)
        
        # ラベル
        for i, ev in enumerate(self.eigvals):
            ax.annotate(f'λ{i}', (ev.real, ev.imag),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        ax.set_title('Eigenvalues of Koopman Matrix A')
        ax.axhline(0, color='black', alpha=0.3)
        ax.axvline(0, color='black', alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved eigenvalue plot: {save_path}")


# =========================================================
#  Visualization Functions
# =========================================================

def plot_tokens_on_ax(ax, token_vecs, token_indices, tokenizer, scale_factor,
                     selected_tokens=None, color='gray', alpha=0.6):
    """
    トークンを固有空間にプロット
    
    Args:
        selected_tokens: Greedyで選ばれたトークンのリスト（強調表示）
    """
    if selected_tokens is None:
        selected_tokens = []
    
    texts = []
    for idx in token_indices:
        if idx >= len(token_vecs):
            continue
        
        vec = token_vecs[idx]
        x, y = vec[0] * scale_factor, vec[1] * scale_factor
        
        # Greedyで選ばれたトークンは強調表示
        if idx in selected_tokens:
            # 大きな赤い星マーカー
            ax.scatter(x, y, s=200, marker='*', color='red',
                      edgecolors='darkred', linewidths=2, zorder=10,
                      label='Selected' if idx == selected_tokens[0] else '')
        else:
            # 通常のトークン
            ax.scatter(x, y, s=30, color=color, alpha=alpha, zorder=3)
        
        # ラベル
        label = get_token_label(idx, tokenizer)
        fontweight = 'bold' if idx in selected_tokens else 'normal'
        fontsize = 10 if idx in selected_tokens else 7
        text_color = 'red' if idx in selected_tokens else 'black'
        
        txt = ax.text(x, y, label, fontsize=fontsize, ha='center',
                     fontweight=fontweight, color=text_color, zorder=4)
        texts.append(txt)
    
    return texts


def visualize_eigen_projection_per_step(analyzer, step_data, scenario, tokenizer, out_dir):
    """
    各ステップごとに固有空間射影を8枚のプロットで表示
    
    16個の固有モードを8ペア（2モードずつ）でプロット
    - 8枚すべてで同じ軸範囲
    - 原点が中心
    - Greedyで選ばれたトークンを強調表示
    """
    num_steps = len(step_data)
    z_dim = analyzer.z_dim
    
    # 固有モードのペアリング
    pairs = []
    used_indices = set()
    
    for i in range(z_dim):
        if i in used_indices:
            continue
        
        eig = analyzer.eigvals[i]
        
        # 複素数の場合
        if abs(eig.imag) > 1e-5:
            pairs.append({
                'x_idx': i, 'y_idx': i,
                'type': 'complex',
                'eig': eig
            })
            used_indices.add(i)
            # 共役ペア
            if i + 1 < z_dim and abs(analyzer.eigvals[i+1].imag) > 1e-5:
                used_indices.add(i+1)
        else:
            # 実数の場合、次の実数ペアを探す
            pair_found = False
            for j in range(i + 1, z_dim):
                if j not in used_indices and abs(analyzer.eigvals[j].imag) < 1e-5:
                    pairs.append({
                        'x_idx': i, 'y_idx': j,
                        'type': 'real',
                        'eig_x': eig, 'eig_y': analyzer.eigvals[j]
                    })
                    used_indices.add(i)
                    used_indices.add(j)
                    pair_found = True
                    break
            
            if not pair_found:
                # 単独実数モード
                pairs.append({
                    'x_idx': i, 'y_idx': i,
                    'type': 'real_single',
                    'eig': eig
                })
                used_indices.add(i)
    
    # 最大8ペア（16次元分）
    pairs = pairs[:8]
    
    # 各ステップごとに8枚のプロット
    for step_idx, data in enumerate(step_data):
        step = data['step']
        
        # 軌跡データ（このステップまで）
        traj_alpha_with = np.array([step_data[s]['alpha_with'] for s in range(step_idx + 1)])
        traj_alpha_without = np.array([step_data[s]['alpha_without'] for s in range(step_idx + 1)])
        
        # Greedyで選ばれたトークン
        selected_token = data['next_token_with']
        
        # 全体のスケール計算（8枚すべてで統一）
        all_x_vals = []
        all_y_vals = []
        
        for pair_info in pairs:
            if pair_info['type'] == 'complex':
                idx = pair_info['x_idx']
                all_x_vals.extend(traj_alpha_with[:, idx].real)
                all_y_vals.extend(traj_alpha_with[:, idx].imag)
            else:
                idx_x = pair_info['x_idx']
                idx_y = pair_info['y_idx']
                all_x_vals.extend(traj_alpha_with[:, idx_x].real)
                if pair_info['type'] == 'real':
                    all_y_vals.extend(traj_alpha_with[:, idx_y].real)
        
        # 軸範囲を統一（原点中心）
        max_val = max(np.abs(all_x_vals + all_y_vals)) if len(all_x_vals) > 0 else 1.0
        max_val = max(max_val, 0.1)  # 最小値
        axis_lim = [-max_val * 1.1, max_val * 1.1]
        
        # プロット作成（4×2グリッド）
        num_plots = len(pairs)
        cols = 4
        rows = 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
        axes = np.array(axes).flatten()
        
        fig.suptitle(f"Eigen-space Projection: {scenario['name']} | Step {step}",
                    fontsize=16, weight='bold')
        
        # 全トークン
        all_tokens = list(range(40))
        
        for plot_idx, pair_info in enumerate(pairs):
            ax = axes[plot_idx]
            
            # X, Yデータの抽出
            if pair_info['type'] == 'complex':
                idx = pair_info['x_idx']
                x_vals_with = traj_alpha_with[:, idx].real
                y_vals_with = traj_alpha_with[:, idx].imag
                x_vals_wo = traj_alpha_without[:, idx].real
                y_vals_wo = traj_alpha_without[:, idx].imag
                
                # トークン重みベクトル
                W_vecs = np.column_stack([
                    analyzer.W_modal[:, idx].real,
                    analyzer.W_modal[:, idx].imag
                ])
                
                title = f"Mode {idx} (Complex)\nλ={pair_info['eig'].real:.2f}±{abs(pair_info['eig'].imag):.2f}j"
                xlabel, ylabel = "Real", "Imag"
                
            else:  # real or real_single
                idx_x = pair_info['x_idx']
                idx_y = pair_info['y_idx']
                
                x_vals_with = traj_alpha_with[:, idx_x].real
                y_vals_with = traj_alpha_with[:, idx_y].real if pair_info['type'] == 'real' else np.zeros_like(x_vals_with)
                
                x_vals_wo = traj_alpha_without[:, idx_x].real
                y_vals_wo = traj_alpha_without[:, idx_y].real if pair_info['type'] == 'real' else np.zeros_like(x_vals_wo)
                
                W_vecs = np.column_stack([
                    analyzer.W_modal[:, idx_x].real,
                    analyzer.W_modal[:, idx_y].real if pair_info['type'] == 'real' else np.zeros_like(analyzer.W_modal[:, idx_x].real)
                ])
                
                if pair_info['type'] == 'real':
                    title = f"Mode {idx_x} vs {idx_y} (Real)\nλx={pair_info['eig_x'].real:.2f}, λy={pair_info['eig_y'].real:.2f}"
                    xlabel, ylabel = f"Re(α_{idx_x})", f"Re(α_{idx_y})"
                else:
                    title = f"Mode {idx_x} (Single Real)\nλ={pair_info['eig'].real:.2f}"
                    xlabel, ylabel = f"Re(α_{idx_x})", "0"
            
            # スケーリング計算（トークン表示用）
            traj_max = max(np.max(np.abs(x_vals_with)), np.max(np.abs(y_vals_with)))
            W_max = np.max(np.abs(W_vecs)) if W_vecs.size > 0 else 1.0
            scale_factor = (traj_max / W_max) * 0.8 if W_max > 0 else 1.0
            
            # プロット: 軌跡
            ax.plot(x_vals_with, y_vals_with, 'o-', color='steelblue',
                   label='With', markersize=4, alpha=0.7, linewidth=1.5)
            ax.plot(x_vals_with[0], y_vals_with[0], 'D', color='green',
                   markersize=6, label='Start')
            ax.plot(x_vals_wo, y_vals_wo, '--', color='gray',
                   alpha=0.4, label='W/o', linewidth=1)
            
            # プロット: トークン（Greedy選択を強調）
            plot_tokens_on_ax(ax, W_vecs, all_tokens, tokenizer, scale_factor,
                            selected_tokens=[selected_token])
            
            # 軸設定（統一範囲）
            ax.set_xlim(axis_lim)
            ax.set_ylim(axis_lim)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.axhline(0, color='k', ls=':', lw=0.5)
            ax.axvline(0, color='k', ls=':', lw=0.5)
            ax.grid(True, alpha=0.3)
            if plot_idx == 0:
                ax.legend(fontsize=7)
        
        # 余ったサブプロットを消す
        for k in range(num_plots, len(axes)):
            axes[k].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(out_dir, f"{scenario['name']}_eigen_projection_step{step:02d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved eigen projection (step {step}): {save_path}")


def visualize_rollout_analysis(analyzer, step_data, scenario, tokenizer, out_dir):
    """
    ロールアウト解析結果を可視化（5ステップ横並び）
    
    レイアウト:
      各列 = 1ステップ
      Row 0: z_with固有射影（ヒートマップ）
      Row 1: z_without固有射影（ヒートマップ）
      Row 2: W寄与(with)（縦長ヒートマップ）
      Row 3: W寄与(without)（縦長ヒートマップ）
      Row 4: 確率棒グラフ（with/without重ねて表示）
      Row 5: 差分棒グラフ
    """
    num_steps = min(len(step_data), 5)  # 最大5ステップ
    z_dim = analyzer.z_dim
    
    # Figure作成
    fig = plt.figure(figsize=(num_steps * 6, 28))
    
    # GridSpec: 6行×num_steps列
    gs = gridspec.GridSpec(6, num_steps, figure=fig,
                          hspace=0.35, wspace=0.25,
                          top=0.96, bottom=0.04, left=0.05, right=0.98)
    
    fig.suptitle(f"Koopman Mode Decomposition Analysis: {scenario['name']}",
                fontsize=24, weight='bold')
    
    # カラーマップの範囲を事前計算（全ステップで統一）
    all_alpha_vals = []
    all_W_vals = []
    all_prob_vals = []
    all_diff_vals = []
    
    for data in step_data[:num_steps]:
        all_alpha_vals.extend([data['alpha_with'], data['alpha_without']])
        
        movable = data['movable_tokens']
        W_modal_movable = analyzer.W_modal[movable, :]
        all_W_vals.append(W_modal_movable)
        
        all_prob_vals.extend([data['movable_probs_with'], data['movable_probs_without']])
        all_diff_vals.append(data['movable_probs_diff'])
    
    alpha_vmax = max(np.abs(np.concatenate(all_alpha_vals)).max(), 1e-6)
    W_vmax = max(np.abs(np.concatenate(all_W_vals, axis=0)).max(), 1e-6)
    prob_vmax = max(np.concatenate(all_prob_vals).max(), 1e-6)
    diff_vmax = max(np.abs(np.concatenate(all_diff_vals)).max(), 1e-6)
    
    # 各ステップを描画
    for step_idx in range(num_steps):
        data = step_data[step_idx]
        col = step_idx
        movable = data['movable_tokens']
        num_movable = len(movable)
        
        token_labels = [get_token_label(t, tokenizer) for t in movable]
        
        # x軸ラベル: 固有値
        evals = analyzer.eigvals
        mode_labels = []
        for i, ev in enumerate(evals):
            if abs(ev.imag) > 1e-5:
                mode_labels.append(f"λ{i}\n{ev.real:.1f}±{abs(ev.imag):.1f}j")
            else:
                mode_labels.append(f"λ{i}\n{ev.real:.2f}")
        
        # === Row 0: z_with固有射影 ===
        ax0 = fig.add_subplot(gs[0, col])
        alpha_with_2d = data['alpha_with'].reshape(1, -1)
        im0 = ax0.imshow(alpha_with_2d.real, aspect='auto', cmap='RdBu_r',
                        vmin=-alpha_vmax, vmax=alpha_vmax)
        ax0.set_title(f"Step {data['step']}: α (with plaza)", fontsize=11, weight='bold')
        ax0.set_yticks([0])
        ax0.set_yticklabels(['α'])
        ax0.set_xticks(range(z_dim))
        ax0.set_xticklabels(mode_labels, rotation=45, ha='right', fontsize=7)
        if col == num_steps - 1:
            plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
        
        # === Row 1: z_without固有射影 ===
        ax1 = fig.add_subplot(gs[1, col])
        alpha_without_2d = data['alpha_without'].reshape(1, -1)
        im1 = ax1.imshow(alpha_without_2d.real, aspect='auto', cmap='RdBu_r',
                        vmin=-alpha_vmax, vmax=alpha_vmax)
        ax1.set_title(f"α (without plaza)", fontsize=11, weight='bold')
        ax1.set_yticks([0])
        ax1.set_yticklabels(['α'])
        ax1.set_xticks(range(z_dim))
        ax1.set_xticklabels(mode_labels, rotation=45, ha='right', fontsize=7)
        if col == num_steps - 1:
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # === Row 2: W寄与(with) ===
        ax2 = fig.add_subplot(gs[2, col])
        W_modal_movable = analyzer.W_modal[movable, :]
        im2 = ax2.imshow(W_modal_movable.real, aspect='auto', cmap='RdBu_r',
                        vmin=-W_vmax, vmax=W_vmax)
        ax2.set_title(f"W_modal (movable)", fontsize=11, weight='bold')
        ax2.set_yticks(range(num_movable))
        ax2.set_yticklabels(token_labels, fontsize=8)
        ax2.set_xticks(range(z_dim))
        ax2.set_xticklabels(mode_labels, rotation=45, ha='right', fontsize=7)
        if col == num_steps - 1:
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # === Row 3: W寄与(without)+bias ===
        ax3 = fig.add_subplot(gs[3, col])
        logits_contribution = W_modal_movable.real @ data['alpha_without'].real
        logits_with_bias = logits_contribution + analyzer.b[movable]
        logits_2d = logits_with_bias.reshape(-1, 1)
        im3 = ax3.imshow(logits_2d, aspect='auto', cmap='RdBu_r',
                        vmin=-W_vmax*alpha_vmax, vmax=W_vmax*alpha_vmax)
        ax3.set_title(f"Logits (W@α + b)", fontsize=11, weight='bold')
        ax3.set_yticks(range(num_movable))
        ax3.set_yticklabels(token_labels, fontsize=8)
        ax3.set_xticks([0])
        ax3.set_xticklabels(['Logit'], fontsize=8)
        if col == num_steps - 1:
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # === Row 4: 確率棒グラフ ===
        ax4 = fig.add_subplot(gs[4, col])
        x_pos = np.arange(num_movable)
        width = 0.35
        ax4.bar(x_pos - width/2, data['movable_probs_with'], width,
               label='With plaza', color='steelblue', alpha=0.7)
        ax4.bar(x_pos + width/2, data['movable_probs_without'], width,
               label='Without plaza', color='gray', alpha=0.7)
        ax4.set_title(f"Token Probabilities", fontsize=11, weight='bold')
        ax4.set_ylabel('Probability', fontsize=9)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=7)
        ax4.set_ylim([0, prob_vmax * 1.1])
        ax4.legend(fontsize=7, loc='upper right')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # === Row 5: 差分棒グラフ ===
        ax5 = fig.add_subplot(gs[5, col])
        colors = ['red' if d > 0 else 'blue' for d in data['movable_probs_diff']]
        ax5.bar(x_pos, data['movable_probs_diff'], color=colors, alpha=0.7)
        ax5.set_title(f"Probability Diff (With - Without)", fontsize=11, weight='bold')
        ax5.set_ylabel('Diff', fontsize=9)
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=7)
        ax5.set_ylim([-diff_vmax * 1.1, diff_vmax * 1.1])
        ax5.axhline(0, color='black', linewidth=0.8)
        ax5.grid(True, alpha=0.3, axis='y')
    
    save_path = os.path.join(out_dir, f"{scenario['name']}_rollout_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved rollout analysis: {save_path}")


# =========================================================
#  Main Analysis Loop
# =========================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def visualize_rollout_summary_8plots(
    analyzer,
    step_data,
    scenario,
    tokenizer,
    out_dir,
    end_token_id=39,
    topk_modes=8,
):
    """
    1シナリオにつき1枚、8サブプロットで
    - 全ステップの固有空間(α(t))推移
    - 選択トークンの履歴
    - (可能なら) <end>確率推移
    - (差分) with-without の寄与
    をまとめて可視化する
    """
    z_dim = analyzer.z_dim
    T = len(step_data)

    # ----- 時系列テンソル化 -----
    alpha_with = np.stack([d["alpha_with"] for d in step_data], axis=0)      # [T, z]
    alpha_wo   = np.stack([d["alpha_without"] for d in step_data], axis=0)   # [T, z]
    delta_alpha = alpha_with - alpha_wo                                      # [T, z]

    # complex -> 実部/虚部/絶対値
    aw_re = np.real(alpha_with); aw_im = np.imag(alpha_with); aw_abs = np.abs(alpha_with)
    dw_re = np.real(delta_alpha); dw_im = np.imag(delta_alpha); dw_abs = np.abs(delta_alpha)

    sel_with = [int(d["selected_with"]) for d in step_data]
    sel_wo   = [int(d["selected_without"]) for d in step_data]
    p_with   = [float(d.get("prob_with", np.nan)) for d in step_data]
    p_wo     = [float(d.get("prob_without", np.nan)) for d in step_data]

    has_end = ("end_prob_with" in step_data[0]) and ("end_prob_without" in step_data[0])
    if has_end:
        endp_with = [float(d["end_prob_with"]) for d in step_data]
        endp_wo   = [float(d["end_prob_without"]) for d in step_data]

    # ----- ラベル（λをそのまま表示） -----
    # analyzer.eigvals: (z_dim,) complex を想定
    eigvals = getattr(analyzer, "eigvals", None)
    if eigvals is None:
        mode_labels = [f"mode{i}" for i in range(z_dim)]
    else:
        mode_labels = [str(eigvals[i]) for i in range(z_dim)]  # 複素数OK

    # ----- 表示対象のモードを選ぶ（長期寄与っぽい順） -----
    # |λ|が大きい順で上位 topk_modes
    if eigvals is not None:
        order = np.argsort(-np.abs(eigvals))
    else:
        order = np.arange(z_dim)
    modes = order[:min(topk_modes, z_dim)]

    # ----- 図全体 -----
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1.1, 1.1, 1.0, 1.0], hspace=0.35, wspace=0.25)

    title = f"Scenario: {scenario} | T={T} | Modes shown={len(modes)}"
    fig.suptitle(title, fontsize=16, weight="bold")

    # (1) Re(alpha_with): time x mode heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(aw_re[:, modes], aspect="auto", cmap="coolwarm")
    ax1.set_title("① Re(α_with(t))  (time × mode)")
    ax1.set_xlabel("mode (λ)")
    ax1.set_ylabel("time step")
    ax1.set_xticks(np.arange(len(modes)))
    ax1.set_xticklabels([mode_labels[i] for i in modes], fontsize=8, rotation=45, ha="right")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)

    # (2) Re(alpha_without): time x mode heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(np.real(alpha_wo)[:, modes], aspect="auto", cmap="coolwarm")
    ax2.set_title("② Re(α_without(t))  (time × mode)")
    ax2.set_xlabel("mode (λ)")
    ax2.set_ylabel("time step")
    ax2.set_xticks(np.arange(len(modes)))
    ax2.set_xticklabels([mode_labels[i] for i in modes], fontsize=8, rotation=45, ha="right")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)

    # (3) Re(delta_alpha): time x mode heatmap
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(dw_re[:, modes], aspect="auto", cmap="coolwarm")
    ax3.set_title("③ Re(Δα(t)=α_with−α_wo)  (time × mode)")
    ax3.set_xlabel("mode (λ)")
    ax3.set_ylabel("time step")
    ax3.set_xticks(np.arange(len(modes)))
    ax3.set_xticklabels([mode_labels[i] for i in modes], fontsize=8, rotation=45, ha="right")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.02)

    # (4) |Δα|: time x mode heatmap（大きさ）
    ax4 = fig.add_subplot(gs[1, 1])
    im4 = ax4.imshow(dw_abs[:, modes], aspect="auto", cmap="viridis")
    ax4.set_title("④ |Δα(t)|  (time × mode)")
    ax4.set_xlabel("mode (λ)")
    ax4.set_ylabel("time step")
    ax4.set_xticks(np.arange(len(modes)))
    ax4.set_xticklabels([mode_labels[i] for i in modes], fontsize=8, rotation=45, ha="right")
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.02)

    # (5) 選択トークン履歴（with/wo） + その確率
    ax5 = fig.add_subplot(gs[2, 0])
    t = np.arange(1, T + 1)
    ax5.plot(t, sel_with, marker="o", linewidth=1.5, label="selected (with)")
    ax5.plot(t, sel_wo, marker="x", linewidth=1.5, label="selected (w/o)")
    ax5.set_title("⑤ Selected token over time")
    ax5.set_xlabel("time step")
    ax5.set_ylabel("token id")
    ax5.grid(alpha=0.3)
    ax5.legend(fontsize=9, loc="best")

    # 右軸に確率（見える範囲の確認用）
    ax5b = ax5.twinx()
    ax5b.plot(t, p_with, linestyle="--", alpha=0.6, label="prob(with)")
    ax5b.plot(t, p_wo, linestyle="--", alpha=0.6, label="prob(w/o)")
    ax5b.set_ylabel("selected prob")
    # 凡例統合
    h1, l1 = ax5.get_legend_handles_labels()
    h2, l2 = ax5b.get_legend_handles_labels()
    ax5b.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")

    # (6) <end> 確率推移（あれば）
    ax6 = fig.add_subplot(gs[2, 1])
    if has_end:
        ax6.plot(t, endp_with, marker="o", label=f"<end> prob (with)")
        ax6.plot(t, endp_wo, marker="x", label=f"<end> prob (w/o)")
        ax6.set_ylim(0, min(1.0, 1.05 * max(max(endp_with), max(endp_wo), 1e-6)))
        ax6.set_title("⑥ <end> probability over time")
        ax6.set_xlabel("time step")
        ax6.set_ylabel("P(<end>)")
        ax6.grid(alpha=0.3)
        ax6.legend(fontsize=9, loc="best")
    else:
        ax6.text(0.5, 0.5, "⑥ <end> prob not available\n(add end_prob_* to step_data)",
                 ha="center", va="center", fontsize=12)
        ax6.axis("off")

    # (7) 2次元軌跡（上位の複素ペア or 上位2モードを平面に）
    ax7 = fig.add_subplot(gs[3, 0])
    if len(modes) >= 2:
        i, j = modes[0], modes[1]
        x = aw_re[:, i]
        y = aw_re[:, j]
        ax7.plot(x, y, marker="o", linewidth=1.5)
        for k in range(T):
            ax7.text(x[k], y[k], str(k+1), fontsize=8)
        ax7.set_title(f"⑦ Trajectory in (Re α_mode0, Re α_mode1)\nmode0 λ={mode_labels[i]} | mode1 λ={mode_labels[j]}")
        ax7.set_xlabel(f"Re α[{i}]")
        ax7.set_ylabel(f"Re α[{j}]")
        ax7.grid(alpha=0.3)
    else:
        ax7.text(0.5, 0.5, "⑦ Need ≥2 modes to plot trajectory", ha="center", va="center")
        ax7.axis("off")

    # (8) “差分成分がどのモードに乗っていたか”を要約（時間方向に集約）
    # 例：Δα(t) のエネルギーをモードごとに積分して棒グラフ
    ax8 = fig.add_subplot(gs[3, 1])
    mode_energy = np.sum(dw_abs**2, axis=0)  # [z]
    show = modes
    ax8.bar(np.arange(len(show)), mode_energy[show])
    ax8.set_title("⑧ Integrated Δα energy per mode  (Σ_t |Δα_i(t)|^2)")
    ax8.set_xlabel("mode (λ)")
    ax8.set_ylabel("energy")
    ax8.set_xticks(np.arange(len(show)))
    ax8.set_xticklabels([mode_labels[i] for i in show], fontsize=8, rotation=45, ha="right")
    ax8.grid(axis="y", alpha=0.3)

    # 保存
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"scenario_{'_'.join(map(str, scenario))}_summary8.png")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[Saved] summary figure: {out_path}")



def run_scenario_analysis(scenario, model, tokenizer, analyzer, adj_matrix, out_dir):
    """シナリオごとの解析を実行"""
    print(f"\n{'='*60}")
    print(f"Analyzing scenario: {scenario['name']}")
    print(f"{'='*60}")
    
    prefix = scenario['prefix']
    plaza_tokens = set(scenario['plaza_node_tokens'])
    
    # Prefix encoding
    with torch.no_grad():
        prefix_tensor = torch.tensor([prefix], dtype=torch.long).to(DEVICE)
        stay_counts = tokenizer.calculate_stay_counts(prefix_tensor).to(DEVICE)
        
        seq_len = len(prefix)
        holidays = torch.full((1, seq_len), scenario['holiday'], dtype=torch.long).to(DEVICE)
        time_zones = torch.full((1, seq_len), scenario['time_zone'], dtype=torch.long).to(DEVICE)
        events = torch.zeros((1, seq_len), dtype=torch.long).to(DEVICE)
        agent_ids = torch.tensor([scenario['agent_id']], dtype=torch.long).to(DEVICE)
        
        z0, h_last = model.encode_prefix(
            prefix_tensor, stay_counts, agent_ids,
            holidays, time_zones, events
        )
        z_0 = z0[0].detach().cpu().numpy()   # [z_dim]

    
    # ロールアウト
    z_with = z_0.copy()
    z_without = z_0.copy()
    
    current_node = prefix[-1] if prefix[-1] < 19 else (prefix[-1] - 19)
    
    generated_with = []
    generated_without = []
    step_data = []
    
    for step in range(NUM_ROLLOUT_STEPS):
        print(f"\n--- Step {step} ---")
        
        # 移動可能トークン
        movable_tokens = get_movable_tokens(current_node, adj_matrix)
        
        # 固有空間に射影
        alpha_with = analyzer.transform_to_eigenspace(z_with)
        alpha_without = analyzer.transform_to_eigenspace(z_without)
        
        # 確率計算（with plaza）
        logits_with = analyzer.W[movable_tokens, :] @ z_with + analyzer.b[movable_tokens]
        # event_mask_with = np.array([1.0 if t in plaza_tokens else 0.0 for t in movable_tokens])
        # event_bias_with = model.event_bias.weight[1].detach().cpu().numpy()[movable_tokens]
        # logits_with = logits_with + event_mask_with * event_bias_with
        
        movable_probs_with = np.exp(logits_with - scipy.special.logsumexp(logits_with))
        next_token_with = movable_tokens[np.argmax(movable_probs_with)]
        
        # 確率計算（without plaza）
        logits_without = analyzer.W[movable_tokens, :] @ z_without + analyzer.b[movable_tokens]
        movable_probs_without = np.exp(logits_without - scipy.special.logsumexp(logits_without))
        next_token_without = movable_tokens[np.argmax(movable_probs_without)]
        
        # 差分
        movable_probs_diff = movable_probs_with - movable_probs_without
        
        # 次状態
        A_np = model.A.detach().cpu().numpy()
        z_next_with = A_np @ z_with
        z_next_without = A_np @ z_without
        
        generated_with.append(next_token_with)
        generated_without.append(next_token_without)
        
        print(f"  Current node: {current_node}")
        print(f"  Movable tokens: {movable_tokens}")
        print(f"  Selected (with): {next_token_with} (prob={movable_probs_with[np.argmax(movable_probs_with)]:.4f})")
        print(f"  Selected (without): {next_token_without} (prob={movable_probs_without[np.argmax(movable_probs_without)]:.4f})")
        
        # データ保存
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
        
        # 次ステップ準備
        z_with = z_next_with
        z_without = z_next_without
        
        if next_token_with < 19:
            current_node = next_token_with
        elif 19 <= next_token_with <= 37:
            current_node = next_token_with - 19
        else:
            print(f"  End token selected, stopping.")
            break
    
    # 可視化
    # 2. 各ステップでの固有空間射影（16次元分8枚）
    visualize_eigen_projection_per_step(analyzer, step_data, scenario, tokenizer, out_dir)
    
    # 3, 4, 5. ロールアウト解析（横並び）
    visualize_rollout_analysis(analyzer, step_data, scenario, tokenizer, out_dir)
    
    print(f"\nGenerated sequence (with plaza): {prefix} -> {generated_with}")
    print(f"Generated sequence (without plaza): {prefix} -> {generated_without}")


def main():
    """メイン処理"""
    print("="*60)
    print("Koopman Mode Decomposition Analysis - Combined Version")
    print("="*60)
    
    # モデル読み込み
    print(f"\nLoading model from: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    config = checkpoint.get('config', {})
    
    model = KoopmanRoutesFormer(
        vocab_size=config.get('vocab_size', 39),
        token_emb_dim=config.get('token_emb_dim', 64),
        d_model=config.get('d_model', 64),
        nhead=config.get('nhead', 4),
        num_layers=config.get('num_layers', 3),
        d_ff=config.get('d_ff', 128),
        z_dim=config.get('z_dim', 16),
        pad_token_id=38,
        base_N=19,
        num_agents=1,
        agent_emb_dim=16,
        max_stay_count=500,
        stay_emb_dim=16,
        holiday_emb_dim=4,
        time_zone_emb_dim=4,
        event_emb_dim=4,
        use_aux_loss=True,
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded. z_dim={model.z_dim}")
    
    # ネットワーク読み込み
    adj_matrix = torch.load(ADJ_PATH, weights_only=True)
    expanded_adj = expand_adjacency_matrix(adj_matrix)
    dummy_node_features = torch.zeros((len(adj_matrix), 1))
    expanded_features = torch.cat([dummy_node_features, dummy_node_features], dim=0)
    network = Network(expanded_adj, expanded_features)
    tokenizer = Tokenization(network)
    
    # 固有値分解
    print("\nPerforming eigenvalue decomposition...")
    analyzer = KoopmanEigenAnalyzer(model)
    
    # 1. 固有値プロット
    eigen_plot_path = os.path.join(OUT_DIR, "eigenvalues_unit_circle.png")
    analyzer.plot_eigenvalues(eigen_plot_path)
    
    # 各シナリオを解析
    for scenario in SCENARIOS:
        run_scenario_analysis(scenario, model, tokenizer, analyzer, adj_matrix, OUT_DIR)
    
    visualize_rollout_summary_8plots(analyzer, step_data, scenario, tokenizer, out_dir, end_token_id=39)

    print(f"\n{'='*60}")
    print(f"Analysis complete! Results saved to: {OUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()