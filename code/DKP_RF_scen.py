"""
Koopman Mode Decomposition Analysis for DKP_RF (New Model)

新モデル（Prefix-only encoding + Autonomous Koopman rollout）用の
固有モード分解による解釈性検証コード

主な機能：
1. A行列の固有値分解と単位円上プロット
2. 各ステップでの潜在状態z_tを固有空間射影
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

# シナリオ定義（以前のコードに倣う）
SCENARIOS = [
    {
        "name": "0,1,2,21",
        "prefix": [0, 1, 2, 21],  # 初期prefix
        "time": 20240101,
        "agent_id": 0,
        "holiday": 1,
        "time_zone": 0,
        "plaza_node_tokens": [2, 21],  # 広場として扱うノード
    },
    {
        "name": "6,5,4,11",
        "prefix": [6,5,4,11, 30],
        "time": 20240101,
        "agent_id": 0,
        "holiday": 1,
        "time_zone": 0,
        "plaza_node_tokens": [11, 30],
    },
    {
        "name": "1,5,6,14",  # ベース名（自動展開時に使用）
        "prefix": [1,5,6,14,33],  # 最長のprefix
        "time": 20240101,
        "agent_id": 0,
        "holiday": 1,              
        "time_zone": 0,            
        "plaza_node_tokens": [14, 33]  # 全展開パターンで共通
    },
    {
        "name": "18,16,14",  # ベース名（自動展開時に使用）
        "prefix": [18,16,14,33],  # 最長のprefix
        "time": 20240101,
        "holiday": 1,              
        "time_zone": 0,            
        "agent_id": 0,
        "plaza_node_tokens": [14, 33]  # 全展開パターンで共通
    },
]

# パス設定
MODEL_PATH = "/home/mizutani/projects/RF/runs/20260127_014201/model_weights_20260127_014201.pth"  # ★要変更
ADJ_PATH = "/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt"
DATA_PATH = "/home/mizutani/projects/RF/data/input_real_m5.npz"

# 出力先
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"/home/mizutani/projects/RF/runs/20260127_014201/scen_{RUN_ID}"
os.makedirs(OUT_DIR, exist_ok=True)

# デバイス
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 解析パラメータ
NUM_ROLLOUT_STEPS = 30  # 何ステップ生成するか


# =========================================================
#  Utility Functions
# =========================================================

def get_movable_tokens(current_node, adj_matrix, pad_token_id=38, end_token_id=39):
    """
    現在位置から移動可能なトークンを取得
    
    Args:
        current_node: 現在のノードID（0-18）
        adj_matrix: 隣接行列 [N, N] (Tensor or ndarray)
        pad_token_id: パディングトークン（除外）
        end_token_id: 終了トークン（常に含める）
    
    Returns:
        movable_tokens: 移動可能トークンのリスト
    """
    # Tensor → numpy変換
    if isinstance(adj_matrix, torch.Tensor):
        adj_np = adj_matrix.cpu().numpy()
    else:
        adj_np = adj_matrix
    
    # 隣接するノードを取得
    neighbors = np.where(adj_np[current_node] > 0)[0]
    
    movable_tokens = []
    
    # Move tokens (0-18)
    for neighbor in neighbors:
        if 0 <= neighbor <= 18:
            movable_tokens.append(int(neighbor))
    
    # Stay tokens (19-37): 現在位置に対応するStay
    if 0 <= current_node <= 18:
        stay_token = current_node + 19
        movable_tokens.append(stay_token)
    
    # End token（常に含める）
    movable_tokens.append(end_token_id)
    
    return sorted(movable_tokens)


def get_token_label(token_id, tokenizer):
    """トークンIDからラベル文字列を取得"""
    # Move tokens (0-18)
    if 0 <= token_id <= 18:
        return f"M{token_id}"
    # Stay tokens (19-37)
    elif 19 <= token_id <= 37:
        return f"S{token_id-19}"
    # Special tokens
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
    """
    Koopman演算子Aの固有値分解と解析
    """
    
    def __init__(self, model):
        """
        Args:
            model: KoopmanRoutesFormer model
        """
        self.model = model
        self.z_dim = model.z_dim
        
        # A行列を取得 [z_dim, z_dim]
        # A は nn.Parameter として定義されている
        A_np = model.A.detach().cpu().numpy()  # [z_dim, z_dim]
        
        # 固有値分解 A = V @ Lambda @ V^{-1}
        eigvals, eigvecs = scipy.linalg.eig(A_np)
        
        # ソート: 絶対値の大きい順
        sort_idx = np.argsort(np.abs(eigvals))[::-1]
        self.eigvals = eigvals[sort_idx]  # [z_dim]
        self.V = eigvecs[:, sort_idx]      # [z_dim, z_dim]
        
        # 逆行列
        self.V_inv = scipy.linalg.inv(self.V)
        
        # 出力重み W [vocab_size, z_dim]
        self.W = model.to_logits.weight.detach().cpu().numpy()
        
        # バイアス b [vocab_size]
        self.b = model.to_logits.bias.detach().cpu().numpy()
        
        # 固有空間での重み W @ V [vocab_size, z_dim]
        self.W_modal = self.W @ self.V
        
        print(f"Eigenvalue decomposition complete:")
        print(f"  z_dim: {self.z_dim}")
        print(f"  A shape: {A_np.shape}")
        print(f"  Eigenvalues (top 5): {self.eigvals[:5]}")
        print(f"  Max eigenvalue magnitude: {np.abs(self.eigvals).max():.4f}")
    
    def transform_to_eigenspace(self, z):
        """
        z を固有空間に射影
        
        Args:
            z: [z_dim] or [batch, z_dim]
        
        Returns:
            alpha: [z_dim] or [batch, z_dim] - 固有空間係数
        """
        if z.ndim == 1:
            return self.V_inv @ z
        else:
            return (self.V_inv @ z.T).T
    
    def plot_eigenvalues(self, save_path):
        """固有値を単位円上にプロット"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 単位円
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1, alpha=0.3, label='Unit Circle')
        
        # 固有値プロット
        ax.scatter(self.eigvals.real, self.eigvals.imag, 
                  c=np.arange(len(self.eigvals)), cmap='coolwarm',
                  s=100, edgecolors='black', linewidth=1.5, zorder=5)
        
        # ラベル
        for i, ev in enumerate(self.eigvals):
            ax.annotate(f'λ{i}', (ev.real, ev.imag), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
        
        ax.set_xlabel('Real', fontsize=14)
        ax.set_ylabel('Imaginary', fontsize=14)
        ax.set_title('Eigenvalues of Koopman Matrix A', fontsize=16, weight='bold')
        ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)
        ax.axvline(0, color='black', linewidth=0.5, alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        ax.axis('equal')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved eigenvalue plot: {save_path}")


# =========================================================
#  Scenario Analysis
# =========================================================

def encode_prefix_with_plaza(model, tokenizer, prefix, agent_id, holiday, time_zone, 
                              plaza_tokens, use_plaza, device):
    """
    Prefixをエンコードして初期潜在状態z_0を取得
    
    Args:
        model: KoopmanRoutesFormer
        tokenizer: Tokenization
        prefix: List[int] - ノード系列
        agent_id: int
        holiday: int
        time_zone: int
        plaza_tokens: List[int] - 広場として扱うノード
        use_plaza: bool - 広場情報を使うか
        device: torch.device
    
    Returns:
        z_0: torch.Tensor [z_dim] - 初期潜在状態
    """
    seq_len = len(prefix)
    
    # Tokenize
    tokens = torch.tensor([prefix], dtype=torch.long).to(device)
    stay_counts = tokenizer.calculate_stay_counts(tokens).to(device)
    agent_ids = torch.tensor([agent_id], dtype=torch.long).to(device)
    holidays = torch.tensor([[holiday] * seq_len], dtype=torch.long).to(device)
    time_zones = torch.tensor([[time_zone] * seq_len], dtype=torch.long).to(device)
    
    # Events（広場情報）
    events = torch.zeros((1, seq_len), dtype=torch.long).to(device)
    if use_plaza:
        for pos, tok in enumerate(prefix):
            if tok in plaza_tokens:
                events[0, pos] = 1
    
    # Encode
    with torch.no_grad():
        # Prefixのみエンコード
        z_0 = model.encode_prefix(
            tokens, stay_counts, agent_ids,
            holidays, time_zones, events
        )  # [1, z_dim]
    
    return z_0[0]  # [z_dim]


def greedy_rollout_with_analysis(model, analyzer, tokenizer, network, adj_matrix,
                                  scenario, num_steps, device, out_dir):
    """
    Greedyロールアウトしながら各ステップでモード分解解析
    
    主な処理：
    1. 広場あり・なしでz_0を取得
    2. 各ステップでA行列をかけてz_t+1を計算
    3. z_t+1を固有空間射影
    4. 重みベクトルの寄与を分析
    5. トークン確率を計算
    6. 最大確率のトークンを選択して次へ
    
    Args:
        model: KoopmanRoutesFormer
        analyzer: KoopmanEigenAnalyzer
        tokenizer: Tokenization
        network: Network
        adj_matrix: 隣接行列
        scenario: dict - シナリオ情報
        num_steps: int - 何ステップ生成するか
        device: torch.device
        out_dir: 出力ディレクトリ
    """
    print(f"\n{'='*60}")
    print(f"Analyzing Scenario: {scenario['name']}")
    print(f"{'='*60}")
    
    prefix = scenario['prefix']
    agent_id = scenario['agent_id']
    holiday = scenario['holiday']
    time_zone = scenario['time_zone']
    plaza_tokens = scenario.get('plaza_node_tokens', [])
    
    # 初期エンコード
    print("Encoding prefix...")
    z_with = encode_prefix_with_plaza(
        model, tokenizer, prefix, agent_id, holiday, time_zone,
        plaza_tokens, use_plaza=True, device=device
    ).cpu().numpy()  # [z_dim]
    
    z_without = encode_prefix_with_plaza(
        model, tokenizer, prefix, agent_id, holiday, time_zone,
        plaza_tokens, use_plaza=False, device=device
    ).cpu().numpy()  # [z_dim]
    
    print(f"  z_with norm: {np.linalg.norm(z_with):.4f}")
    print(f"  z_without norm: {np.linalg.norm(z_without):.4f}")
    print(f"  z difference norm: {np.linalg.norm(z_with - z_without):.4f}")
    
    # z_with, z_without が torch の場合も numpy の場合も対応
    z_with = np.asarray(z_with).reshape(-1)        # (16,)
    z_without = np.asarray(z_without).reshape(-1)  # (16,)

    # 各ステップのデータを収集
    step_data = []
    
    current_node = prefix[-1] % 19  # 最後のノード（Moveトークンに変換）
    generated_with = []
    generated_without = []
    
    # A行列（numpy）
    # モデルでは z @ A.T を使うので、同じように
    A_param = analyzer.model.A.detach().cpu().numpy()  # [z_dim, z_dim]
    A_np = A_param.T  # 転置して使用
    
    for step in range(num_steps):
        print(f"\n--- Step {step+1}/{num_steps} ---")
        
        # z_t+1 = A @ z_t
        z_next_with = A_np @ z_with          # (16,)
        z_next_without = A_np @ z_without    # (16,)
        
        # 固有空間射影
        alpha_with = analyzer.transform_to_eigenspace(z_next_with)  # [z_dim]
        alpha_without = analyzer.transform_to_eigenspace(z_next_without)  # [z_dim]
        
        # 移動可能トークン取得
        movable_tokens = get_movable_tokens(current_node, adj_matrix)
        
        # logits計算: W @ z + b
        logits_with = analyzer.W @ z_next_with + analyzer.b  # [vocab_size]
        logits_without = analyzer.W @ z_next_without + analyzer.b  # [vocab_size]
        
        # 確率計算
        probs_with = np.exp(logits_with) / np.exp(logits_with).sum()
        probs_without = np.exp(logits_without) / np.exp(logits_without).sum()
        
        # 移動可能トークンのみ抽出
        movable_probs_with = probs_with[movable_tokens]
        movable_probs_without = probs_without[movable_tokens]
        movable_probs_diff = movable_probs_with - movable_probs_without
        
        # Greedyに次トークン選択
        next_token_with = movable_tokens[np.argmax(movable_probs_with)]
        next_token_without = movable_tokens[np.argmax(movable_probs_without)]
        
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
        
        # 次ステップ準備（広場ありの方を採用）
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
    visualize_rollout_analysis(
        analyzer, step_data, scenario, tokenizer, out_dir
    )
    
    print(f"\nGenerated sequence (with plaza): {prefix} -> {generated_with}")
    print(f"Generated sequence (without plaza): {prefix} -> {generated_without}")


def visualize_rollout_analysis(analyzer, step_data, scenario, tokenizer, out_dir):
    """
    ロールアウト解析結果を可視化（5ステップ横並び）
    
    レイアウト:
      各列 = 1ステップ
      Row 0: ①z_with固有射影（横長ヒートマップ）
      Row 1: ②z_without固有射影（横長ヒートマップ）
      Row 2: ③W寄与(with)（縦長ヒートマップ: 縦=移動可能トークン, 横=固有モード）
      Row 3: ④W寄与(without)+bias（縦長ヒートマップ）
      Row 4: ⑤確率棒グラフ（with/without重ねて表示）
      Row 5: ⑥差分棒グラフ
    """
    num_steps = len(step_data)
    z_dim = analyzer.z_dim
    
    # Figure作成（横幅=ステップ数×6, 縦幅=固定）
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
    
    for data in step_data:
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
    for step_idx, data in enumerate(step_data):
        col = step_idx
        movable = data['movable_tokens']
        num_movable = len(movable)

        token_labels = [get_token_label(t, tokenizer) for t in movable]

        # ===== x軸ラベル：固有値 λ_i をそのまま表示（複素OK） =====
        # analyzer.evals: shape (z_dim,) complex を想定
        evals = analyzer.eigvals

        def fmt_lambda(lam):
            # 例: 0.79+0.05j の短縮表示
            return f"{lam.real:.2f}{lam.imag:+.2f}j"

        eigenmode_labels = [fmt_lambda(evals[i]) for i in range(z_dim)]

        # 先に各種ベクトルを 1D (z_dim,) に統一
        alpha_with = np.asarray(data['alpha_with']).reshape(-1)       # (z_dim,)
        alpha_without = np.asarray(data['alpha_without']).reshape(-1) # (z_dim,)
        delta_alpha = alpha_with - alpha_without                       # (z_dim,)

        # 可視化用は実部（complexが出るため）
        alpha_with_plot = np.real(alpha_with).reshape(1, -1)           # (1,z)
        alpha_without_plot = np.real(alpha_without).reshape(1, -1)     # (1,z)

        # スケール（α）は with/without 両方で決める
        alpha_vmax_step = float(
            max(np.max(np.abs(np.real(alpha_with))), np.max(np.abs(np.real(alpha_without))))
        )

        # ===== Row 0: ① z_with の固有射影 =====
        ax1 = fig.add_subplot(gs[0, col])
        im1 = ax1.imshow(alpha_with_plot, cmap='coolwarm', aspect='auto',
                        vmin=-alpha_vmax_step, vmax=alpha_vmax_step)
        ax1.set_xticks(np.arange(z_dim))
        ax1.set_xticklabels(eigenmode_labels, fontsize=7, rotation=90, ha='center')
        ax1.set_yticks([0])
        ax1.set_yticklabels(['α'], fontsize=8)
        ax1.set_title(f"Step {step_idx+1}\n① α (with)", fontsize=11, weight='bold')
        if col == num_steps - 1:
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # ===== Row 1: ② z_without の固有射影 =====
        ax2 = fig.add_subplot(gs[1, col])
        im2 = ax2.imshow(alpha_without_plot, cmap='coolwarm', aspect='auto',
                        vmin=-alpha_vmax_step, vmax=alpha_vmax_step)
        ax2.set_xticks(np.arange(z_dim))
        ax2.set_xticklabels(eigenmode_labels, fontsize=7, rotation=90, ha='center')
        ax2.set_yticks([0])
        ax2.set_yticklabels(['α'], fontsize=8)
        ax2.set_title("② α (w/o)", fontsize=11, weight='bold')
        if col == num_steps - 1:
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # ===== Row 2: ③ W寄与（with）: token×mode =====
        ax3 = fig.add_subplot(gs[2, col])
        # analyzer.W_modal: (vocab, z_dim) を想定（= W @ V など）
        W_contrib_with = analyzer.W_modal[movable, :] * alpha_with.reshape(1, -1)  # (num_movable,z)
        W_contrib_with = np.real(W_contrib_with)

        W_vmax_with = float(np.max(np.abs(W_contrib_with))) + 1e-12
        W_contrib_with_plot = W_contrib_with.T

        im3 = ax3.imshow(W_contrib_with_plot, cmap='coolwarm', aspect='auto',
                        vmin=-W_vmax_with, vmax=W_vmax_with)

        # x軸 = token
        ax3.set_xticks(np.arange(num_movable))
        ax3.set_xticklabels(token_labels, fontsize=7, rotation=45, ha='right')

        # y軸 = mode (λ)
        ax3.set_yticks(np.arange(z_dim))
        ax3.set_yticklabels(eigenmode_labels, fontsize=6)

        ax3.set_title("③ Mode×Token contrib (with)", fontsize=11, weight='bold')
        ax3.set_xlabel("Token", fontsize=9)
        ax3.set_ylabel(r"Eigenvalue $\lambda_i$", fontsize=9)


        # ===== Row 3: ④ 差分寄与（with - without）: token×mode =====
        ax4 = fig.add_subplot(gs[3, col])

        # 差分寄与： (W_modal[token,mode] * Δα[mode])
        W_contrib_delta = analyzer.W_modal[movable, :] * delta_alpha.reshape(1, -1)  # (num_movable,z)
        W_contrib_delta = np.real(W_contrib_delta)

        W_vmax_delta = float(np.max(np.abs(W_contrib_delta))) + 1e-12

        W_contrib_delta_plot = W_contrib_delta.T

        im4 = ax4.imshow(W_contrib_delta_plot, cmap='coolwarm', aspect='auto',
                        vmin=-W_vmax_delta, vmax=W_vmax_delta)

        # x軸 = token
        ax4.set_xticks(np.arange(num_movable))
        ax4.set_xticklabels(token_labels, fontsize=7, rotation=45, ha='right')

        # y軸 = mode (λ)
        ax4.set_yticks(np.arange(z_dim))
        ax4.set_yticklabels(eigenmode_labels, fontsize=6)

        ax4.set_title("④ Δ(Mode×Token contrib)\n(with - w/o)", fontsize=11, weight='bold')
        ax4.set_xlabel("Token", fontsize=9)
        ax4.set_ylabel(r"Eigenvalue $\lambda_i$", fontsize=9)


        # ===== Row 4: ⑤ 確率棒グラフ =====
        ax5 = fig.add_subplot(gs[4, col])
        x_pos = np.arange(num_movable)
        width = 0.35
        ax5.bar(x_pos - width/2, data['movable_probs_with'], width,
                alpha=0.8, label='With', color='steelblue')
        ax5.bar(x_pos + width/2, data['movable_probs_without'], width,
                alpha=0.8, label='W/o', color='coral')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(token_labels, fontsize=7, rotation=45, ha='right')
        ax5.set_ylabel('Prob', fontsize=9)
        ax5.set_title('⑤ Token prob', fontsize=11, weight='bold')
        ax5.set_ylim([0, prob_vmax * 1.1])
        if col == 0:
            ax5.legend(fontsize=8, loc='upper left')
        ax5.grid(axis='y', alpha=0.3)

        # ===== Row 5: ⑥ 差分棒グラフ =====
        ax6 = fig.add_subplot(gs[5, col])
        colors = ['green' if v >= 0 else 'red' for v in data['movable_probs_diff']]
        ax6.bar(x_pos, data['movable_probs_diff'], color=colors, alpha=0.7)
        ax6.axhline(0, color='black', linewidth=1.0, linestyle='--')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(token_labels, fontsize=7, rotation=45, ha='right')
        ax6.set_ylabel('Diff', fontsize=9)
        ax6.set_title('⑥ Prob diff', fontsize=11, weight='bold')
        ax6.set_ylim([-diff_vmax * 1.1, diff_vmax * 1.1])
        ax6.grid(axis='y', alpha=0.3)

    
    # 保存
    save_path = os.path.join(out_dir, f"{scenario['name']}_rollout_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved rollout analysis: {save_path}")



# =========================================================
#  Main Analysis Pipeline
# =========================================================

def main():
    """
    メイン解析パイプライン
    """
    print("="*60)
    print("Koopman Mode Decomposition Analysis")
    print("="*60)
    
    # モデルのロード
    print("\n1. Loading model...")
    
    # ネットワーク・隣接行列
    print("  Loading adjacency matrix...")
    adj_matrix = torch.load(ADJ_PATH, weights_only=True)
    
    # 隣接行列の拡張（以前のコードに倣う）
    if adj_matrix.shape[0] == 38:
        base_N = 19
    else:
        base_N = int(adj_matrix.shape[0])
    
    expanded_adj = expand_adjacency_matrix(adj_matrix)
    
    # ダミーのノード特徴量（以前のコードに倣う）
    dummy_feat = torch.zeros((len(adj_matrix), 1))
    node_features = torch.cat([dummy_feat, dummy_feat], dim=0)
    
    # Network初期化
    network = Network(expanded_adj, node_features)
    
    # 隣接行列はexpanded_adjを使用
    # Networkオブジェクトにはadjacency_matrix属性がないため、expanded_adjを直接使う
    final_adj_matrix = expanded_adj
    
    # Tokenizer
    tokenizer = Tokenization(network)
    vocab_size = network.N + 4
    
    # モデルのロード
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found: {MODEL_PATH}")
        print("Please set MODEL_PATH to the correct checkpoint file.")
        return
    
    print(f"  Loading checkpoint from: {MODEL_PATH}")
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    
    # チェックポイントから設定を取得
    if isinstance(ckpt, dict) and 'config' in ckpt:
        c = ckpt['config']
        state_dict = ckpt['model_state_dict']
        print("  Checkpoint contains config")
    else:
        # 古い形式（state_dictのみ）
        state_dict = ckpt
        c = {}
        print("  WARNING: Old checkpoint format, using default config")
    
    # エージェント数と滞在カウント数を自動検出
    if 'agent_embedding.weight' in state_dict:
        det_num_agents = state_dict['agent_embedding.weight'].shape[0]
    else:
        det_num_agents = c.get('num_agents', 1)
    
    if 'stay_embedding.weight' in state_dict:
        det_max_stay = state_dict['stay_embedding.weight'].shape[0] - 1
    else:
        det_max_stay = c.get('max_stay_count', 500)
    
    # 各種埋め込み次元
    h_dim = c.get("holiday_emb_dim", 4)
    tz_dim = c.get("time_zone_emb_dim", 4)
    e_dim = c.get("event_emb_dim", 4)
    
    # base_N の決定
    if adj_matrix.shape[0] == 38:
        base_N = 19
    else:
        base_N = int(adj_matrix.shape[0])
    
    # モデル初期化
    model = KoopmanRoutesFormer(
        vocab_size=c.get('vocab_size', vocab_size),
        token_emb_dim=c.get('token_emb_dim', c.get('d_ie', 64)),
        d_model=c.get('d_model', c.get('d_ie', 64)),
        nhead=c.get('nhead', 4),
        num_layers=c.get('num_layers', 3),
        d_ff=c.get('d_ff', 128),
        z_dim=c.get('z_dim', 16),  # デフォルト16
        pad_token_id=38,
        dist_mat_base=None,
        base_N=base_N,
        num_agents=det_num_agents,
        agent_emb_dim=c.get('agent_emb_dim', 16),
        max_stay_count=det_max_stay,
        stay_emb_dim=c.get('stay_emb_dim', 16),
        holiday_emb_dim=h_dim,
        time_zone_emb_dim=tz_dim,
        event_emb_dim=e_dim,
    ).to(DEVICE)
    
    # 重みロード
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print(f"  Model loaded from: {MODEL_PATH}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 固有値分解
    print("\n2. Eigenvalue decomposition...")
    analyzer = KoopmanEigenAnalyzer(model)
    
    # 固有値プロット
    eigenval_path = os.path.join(OUT_DIR, "eigenvalues.png")
    analyzer.plot_eigenvalues(eigenval_path)
    
    # 各シナリオ解析
    print("\n3. Analyzing scenarios...")
    for scenario in SCENARIOS:
        greedy_rollout_with_analysis(
            model, analyzer, tokenizer, network, final_adj_matrix,
            scenario, NUM_ROLLOUT_STEPS, DEVICE, OUT_DIR
        )
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Results saved to: {OUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()