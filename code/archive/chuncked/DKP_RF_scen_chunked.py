"""
Koopman Mode Decomposition Analysis for Chunked DKP_RF (Multi-phase Rollout)

Chunked Multi-phase モデル（Lステップごとの再エンコード + Koopman Rollout）用の
固有モード分解による解釈性検証コード

主な機能：
1. A行列の固有値分解と単位円上プロット
2. Multi-phase Rolloutによる生成（LステップごとにTransformerでzをリセット）
3. 各ステップでの潜在状態z_tを固有空間射影
4. 重みベクトルの固有空間寄与分析
5. トークン選択確率と広場有無の差分可視化（チャンク境界を明示）
"""

import torch
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from datetime import datetime

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

# パス設定（環境に合わせて変更してください）
MODEL_PATH = "/home/mizutani/projects/RF/runs/20260128_152944_K5_L5/model_weights_20260128_152944.pth"
ADJ_PATH = "/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt"
DATA_PATH = "/home/mizutani/projects/RF/data/input_real_m5.npz"

# 出力先
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
# モデルパスの親ディレクトリなどを参考に保存先を決定
OUT_DIR = os.path.join(os.path.dirname(MODEL_PATH), f"scen_{RUN_ID}")
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


def get_current_node(token, base_N=19):
    """トークンからノードIDを取得"""
    if 0 <= token < base_N:
        return token
    elif base_N <= token < base_N * 2:
        return token - base_N
    else:
        return -1  # Special token


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
#  Scenario Analysis (Chunked / Multi-phase)
# =========================================================

def encode_sequence_with_plaza(model, tokenizer, sequence, agent_id, holiday, time_zone, 
                               plaza_tokens, use_plaza, device):
    """
    系列全体をエンコードして最新の z_0 を取得
    (Chunked Re-encoding用)
    """
    seq_len = len(sequence)
    tokens = torch.tensor([sequence], dtype=torch.long).to(device)
    stay_counts = tokenizer.calculate_stay_counts(tokens).to(device)
    agent_ids = torch.tensor([agent_id], dtype=torch.long).to(device)
    holidays = torch.tensor([[holiday] * seq_len], dtype=torch.long).to(device)
    time_zones = torch.tensor([[time_zone] * seq_len], dtype=torch.long).to(device)
    
    events = torch.zeros((1, seq_len), dtype=torch.long).to(device)
    if use_plaza:
        for pos, tok in enumerate(sequence):
            # Move/StayトークンからノードIDを割り出し、それが広場リストにあるか判定
            node = get_current_node(tok, model.base_N)
            # plaza_tokens にはトークンIDそのもの（Move/Stay）が含まれている前提か、
            # あるいはノードIDが含まれているか。
            # SCENARIO定義を見る限り「plaza_node_tokens: [2, 21]」のようにトークンIDで指定されている。
            # 厳密には node_id で判定すべきだが、指定がトークンならそのまま比較
            if tok in plaza_tokens:
                events[0, pos] = 1
            # 念のためStayトークン対応（例: 2が広場なら、21(Stay at 2)も広場）
            if node != -1:
                # Moveトークン換算でチェック
                if node in plaza_tokens or (node + 19) in plaza_tokens: 
                     events[0, pos] = 1

    with torch.no_grad():
        z_0, _ = model.encode_prefix(
            tokens, stay_counts, agent_ids,
            holidays, time_zones, events
        )
    
    return z_0[0]  # [z_dim]


def greedy_rollout_chunked(model, analyzer, tokenizer, network, adj_matrix,
                           scenario, num_steps, L, device, out_dir):
    """
    Chunked Multi-phase Rollout 解析
    
    Logic:
    1. LステップごとにTransformerで再エンコード（zのリセット）
    2. それ以外のステップは A行列で発展（z_{t+1} = z_t A^T）
    3. 「広場あり」系列で生成を進め、「広場なし」はその系列を強制入力してzを比較
    """
    print(f"\n{'='*60}")
    print(f"Analyzing Scenario: {scenario['name']} (L={L})")
    print(f"{'='*60}")
    
    # 初期設定
    current_seq = list(scenario['prefix'])
    agent_id = scenario['agent_id']
    holiday = scenario['holiday']
    time_zone = scenario['time_zone']
    plaza_tokens = scenario.get('plaza_node_tokens', [])
    
    # A行列（転置して使用: z @ A.T）
    A_np = analyzer.model.A.detach().cpu().numpy().T
    
    # zの初期化（ループ前）
    z_with = encode_sequence_with_plaza(
        model, tokenizer, current_seq, agent_id, holiday, time_zone,
        plaza_tokens, True, device
    ).cpu().numpy()
    
    z_without = encode_sequence_with_plaza(
        model, tokenizer, current_seq, agent_id, holiday, time_zone,
        plaza_tokens, False, device
    ).cpu().numpy()
    
    step_data = []
    chunk_boundaries = []
    
    for step in range(num_steps):
        print(f"\n--- Step {step+1}/{num_steps} ---")
        
        # -------------------------------------------------
        # 1. 状態発展 (Evolve or Re-encode)
        # -------------------------------------------------
        # 最初のステップ(step=0)は初期エンコード済み。
        # Koopman発展は「現在のz」から「次の予測用z」を作る処理。
        # DKP_RF.pyのforward_rolloutでは、Encode直後のz_0から
        # ループ内で z = z @ A.T してから logits を計算している。
        # つまり、各ステップの予測には必ず1回 A を掛ける必要がある。
        
        # ただし、Lステップ目（chunkの切れ目）では再エンコードを行う。
        # step 0: Encode -> Evolve -> Predict (Prefixの次)
        # ...
        # step L: Re-Encode (using new seq) -> Evolve -> Predict
        
        is_reencode_step = (step > 0 and step % L == 0)
        
        if is_reencode_step:
            print("  [Re-encoding by Transformer]")
            chunk_boundaries.append(step)
            
            # 再エンコード (Reset z)
            z_with = encode_sequence_with_plaza(
                model, tokenizer, current_seq, agent_id, holiday, time_zone,
                plaza_tokens, True, device
            ).cpu().numpy()
            
            z_without = encode_sequence_with_plaza(
                model, tokenizer, current_seq, agent_id, holiday, time_zone,
                plaza_tokens, False, device
            ).cpu().numpy()
        
        # Koopman Evolution (z_{t+1} = z_t @ A^T)
        # これは再エンコード直後でも行う（forward_rolloutの仕様に合わせる）
        z_next_with = A_np @ z_with
        z_next_without = A_np @ z_without
        
        # -------------------------------------------------
        # 2. 解析 & 予測
        # -------------------------------------------------
        # 固有空間射影
        alpha_with = analyzer.transform_to_eigenspace(z_next_with)
        alpha_without = analyzer.transform_to_eigenspace(z_next_without)
        
        # 移動可能トークン
        last_node = get_current_node(current_seq[-1], model.base_N)
        # Special tokenならEndのみ許可するなど
        if last_node == -1:
            movable_tokens = [model.pad_token_id, 39]
        else:
            movable_tokens = get_movable_tokens(last_node, adj_matrix)
        
        # Logits: W @ z + b
        logits_with = analyzer.W @ z_next_with + analyzer.b
        logits_without = analyzer.W @ z_next_without + analyzer.b
        
        # Softmax
        probs_with = np.exp(logits_with) / np.exp(logits_with).sum()
        probs_without = np.exp(logits_without) / np.exp(logits_without).sum()
        
        movable_probs_with = probs_with[movable_tokens]
        movable_probs_without = probs_without[movable_tokens]
        
        # Greedy Selection (based on 'With Plaza')
        best_idx = np.argmax(movable_probs_with)
        next_token = movable_tokens[best_idx]
        prob_val = movable_probs_with[best_idx]
        
        print(f"  Current tail: {get_token_label(current_seq[-1], tokenizer)}")
        print(f"  Selected: {get_token_label(next_token, tokenizer)} (p={prob_val:.4f})")
        
        # データ保存
        step_data.append({
            'step': step,
            'is_reencode': is_reencode_step,
            'z_with': z_next_with.copy(),
            'z_without': z_next_without.copy(),
            'alpha_with': alpha_with.copy(),
            'alpha_without': alpha_without.copy(),
            'movable_tokens': movable_tokens.copy(),
            'movable_probs_with': movable_probs_with.copy(),
            'movable_probs_without': movable_probs_without.copy(),
            'movable_probs_diff': movable_probs_with - movable_probs_without,
            'next_token': next_token,
        })
        
        # -------------------------------------------------
        # 3. 更新
        # -------------------------------------------------
        # 系列更新
        current_seq.append(next_token)
        
        # z更新 (次のループのために保持)
        z_with = z_next_with
        z_without = z_next_without
        
        # 終了判定
        if next_token == 39: # End token
            print("  End token generated.")
            break

    # 可視化
    visualize_chunked_analysis(
        analyzer, step_data, scenario, tokenizer, chunk_boundaries, out_dir
    )
    # ★追加: Koopman Biplot (z軌跡とWベクトルの同時プロット)
    visualize_koopman_biplot(
        analyzer, step_data, scenario, tokenizer, chunk_boundaries, out_dir
    )
    
    print(f"\n" + "="*30)


def visualize_chunked_analysis(analyzer, step_data, scenario, tokenizer, chunk_boundaries, out_dir):
    """
    Chunked Rollout解析結果の可視化
    - チャンク境界（Re-encode地点）に縦線を入れる
    """
    num_steps = len(step_data)
    z_dim = analyzer.z_dim
    
    # Figure作成
    fig = plt.figure(figsize=(num_steps * 6, 28))
    gs = gridspec.GridSpec(6, num_steps, figure=fig, 
                          hspace=0.35, wspace=0.25,
                          top=0.96, bottom=0.04, left=0.05, right=0.98)
    
    fig.suptitle(f"Chunked Koopman Analysis: {scenario['name']}", fontsize=24, weight='bold')
    
    # スケール計算
    all_alpha = []
    all_W = []
    all_prob = []
    all_diff = []
    
    for data in step_data:
        all_alpha.extend([data['alpha_with'], data['alpha_without']])
        W_modal_movable = analyzer.W_modal[data['movable_tokens'], :]
        all_W.append(W_modal_movable)
        all_prob.extend([data['movable_probs_with'], data['movable_probs_without']])
        all_diff.append(data['movable_probs_diff'])
    
    def safe_max(arr): return np.max(np.abs(arr)) if len(arr) > 0 else 1.0
    
    # 固有値ラベル
    eigenmode_labels = [f"{lam.real:.2f}{lam.imag:+.2f}j" for lam in analyzer.eigvals]
    
    for step_idx, data in enumerate(step_data):
        col = step_idx
        movable = data['movable_tokens']
        token_labels = [get_token_label(t, tokenizer) for t in movable]
        
        # 共通描画設定：チャンク境界線
        # step_idx が chunk_boundaries に含まれていれば左端に線を引く
        # ただし step_idx=0 は引かない
        show_boundary = (step_idx in chunk_boundaries)
        
        # データ準備
        alpha_with = np.real(np.asarray(data['alpha_with']).reshape(1, -1))
        alpha_without = np.real(np.asarray(data['alpha_without']).reshape(1, -1))
        delta_alpha = alpha_with - alpha_without
        
        # Row 0: alpha (with)
        ax1 = fig.add_subplot(gs[0, col])
        im1 = ax1.imshow(alpha_with, cmap='coolwarm', aspect='auto') # normは省略(自動)
        ax1.set_title(f"Step {step_idx+1}" + (" [RESET]" if data['is_reencode'] else "") + "\n① α (with)", fontsize=11)
        ax1.set_xticks(range(z_dim)); ax1.set_xticklabels([])
        ax1.set_yticks([]); 
        if show_boundary:
            ax1.axvline(x=-0.5, color='black', linestyle='--', linewidth=2)
            
        # Row 1: alpha (w/o)
        ax2 = fig.add_subplot(gs[1, col])
        im2 = ax2.imshow(alpha_without, cmap='coolwarm', aspect='auto')
        ax2.set_title("② α (w/o)", fontsize=11)
        ax2.set_xticks(range(z_dim)); ax2.set_xticklabels(eigenmode_labels, rotation=90, fontsize=7)
        ax2.set_yticks([]);
        if show_boundary:
            ax2.axvline(x=-0.5, color='black', linestyle='--', linewidth=2)

        # Row 2: W contrib (with)
        ax3 = fig.add_subplot(gs[2, col])
        W_contrib = np.real(analyzer.W_modal[movable, :] * data['alpha_with'].reshape(1, -1))
        im3 = ax3.imshow(W_contrib.T, cmap='coolwarm', aspect='auto')
        ax3.set_title("③ Mode×Token (with)", fontsize=11)
        ax3.set_xticks(range(len(movable))); ax3.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=7)
        ax3.set_yticks(range(z_dim)); ax3.set_yticklabels(eigenmode_labels, fontsize=6)
        if show_boundary:
            ax3.axvline(x=-0.5, color='black', linestyle='--', linewidth=2)

        # Row 3: W diff
        ax4 = fig.add_subplot(gs[3, col])
        W_diff = np.real(analyzer.W_modal[movable, :] * delta_alpha.reshape(1, -1))
        im4 = ax4.imshow(W_diff.T, cmap='coolwarm', aspect='auto')
        ax4.set_title("④ ΔContrib (with-w/o)", fontsize=11)
        ax4.set_xticks(range(len(movable))); ax4.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=7)
        ax4.set_yticks([]);
        if show_boundary:
            ax4.axvline(x=-0.5, color='black', linestyle='--', linewidth=2)

        # Row 4: Prob
        ax5 = fig.add_subplot(gs[4, col])
        x = np.arange(len(movable))
        ax5.bar(x - 0.15, data['movable_probs_with'], 0.3, label='With', color='steelblue')
        ax5.bar(x + 0.15, data['movable_probs_without'], 0.3, label='W/o', color='coral')
        ax5.set_xticks(x); ax5.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=7)
        ax5.set_title("⑤ Probabilities", fontsize=11)
        ax5.set_ylim(0, 1.0)
        if col == 0: ax5.legend()
        if show_boundary: # bar plotの境界線はx軸基準ではなく全体枠基準で描くのが難しいので省略、あるいは:
            # ax5.axvline(x=-0.5, ...) # x軸の-0.5に引く
            pass

        # Row 5: Prob Diff
        ax6 = fig.add_subplot(gs[5, col])
        colors = ['green' if v >= 0 else 'red' for v in data['movable_probs_diff']]
        ax6.bar(x, data['movable_probs_diff'], color=colors)
        ax6.set_xticks(x); ax6.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=7)
        ax6.set_title("⑥ Prob Diff", fontsize=11)
        ax6.axhline(0, color='k', linewidth=0.5)
        ax6.set_ylim(-0.5, 0.5)

    save_path = os.path.join(out_dir, f"{scenario['name']}_chunked_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def visualize_koopman_biplot(analyzer, step_data, scenario, tokenizer, chunk_boundaries, out_dir):
    """
    Koopman Biplot: 固有空間上での z の軌跡と、W（トークン重み）の可視化
    
    X軸: 第1固有モード (Real)
    Y軸: 第2固有モード (Real)
    """
    # データ準備
    z_dim = analyzer.z_dim
    
    # 軌跡データ (alpha)
    traj_alpha_with = np.array([d['alpha_with'] for d in step_data]).reshape(len(step_data), z_dim)
    traj_alpha_without = np.array([d['alpha_without'] for d in step_data]).reshape(len(step_data), z_dim)
    
    # 表示する固有モードのインデックス (上位2つ)
    # ※ 固有値は絶対値順にソート済みと仮定
    idx_x, idx_y = 0, 1
    
    # 固有値情報の取得
    eig_x = analyzer.eigvals[idx_x]
    eig_y = analyzer.eigvals[idx_y]
    
    # -------------------------------------------------------
    # プロット作成
    # -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 1. zの軌跡 (With Plaza)
    x_vals = traj_alpha_with[:, idx_x].real
    y_vals = traj_alpha_with[:, idx_y].real
    
    # チャンクごとに色を変えるか、一本の線で引くか
    # ここでは一本の線 + ポイントで時間を表現
    ax.plot(x_vals, y_vals, 'o-', color='steelblue', label='Trajectory (With)', markersize=4, alpha=0.7)
    
    # 開始点と終了点
    ax.plot(x_vals[0], y_vals[0], 'D', color='green', markersize=8, label='Start')
    ax.plot(x_vals[-1], y_vals[-1], 'X', color='red', markersize=8, label='End')
    
    # チャンク境界（リセット地点）を強調
    for boundary_step in chunk_boundaries:
        if boundary_step < len(x_vals):
            ax.plot(x_vals[boundary_step], y_vals[boundary_step], '*', color='orange', markersize=12, label='Reset' if boundary_step==chunk_boundaries[0] else None)
    
    # 2. zの軌跡 (Without Plaza) - 薄く表示
    x_vals_wo = traj_alpha_without[:, idx_x].real
    y_vals_wo = traj_alpha_without[:, idx_y].real
    ax.plot(x_vals_wo, y_vals_wo, '--', color='gray', alpha=0.4, label='Trajectory (W/o)')

    # 3. W (トークン重みベクトル) のプロット
    # 全トークンを表示すると見づらいため、「生成されたトークン」と「移動可能トークン」に絞る
    
    # 表示対象のトークンID集合を作成
    generated_tokens = set([d['next_token'] for d in step_data])
    
    # 初期位置周辺の移動可能トークンも含める（最初のステップのmovable）
    initial_movables = set(step_data[0]['movable_tokens'])
    
    target_tokens = generated_tokens.union(initial_movables)
    
    # W_modal = W @ V (固有空間での重み)
    W_modal = analyzer.W_modal
    
    # スケーリング係数の計算 (zの軌跡の範囲に合わせて矢印の長さを調整)
    traj_max = np.max(np.abs(np.concatenate([x_vals, y_vals])))
    W_max = np.max(np.abs(W_modal[list(target_tokens)][:, [idx_x, idx_y]]))
    scale_factor = (traj_max / W_max) * 0.8  # 軌跡の8割くらいの長さになるように調整
    
    texts = []
    for token in target_tokens:
        # ベクトル成分
        vec_x = W_modal[token, idx_x].real
        vec_y = W_modal[token, idx_y].real
        
        # 原点から矢印を描画
        ax.arrow(0, 0, vec_x * scale_factor, vec_y * scale_factor, 
                 color='coral', alpha=0.5, head_width=traj_max*0.03, length_includes_head=True)
        
        # ラベル表示
        label_txt = get_token_label(token, tokenizer)
        t = ax.text(vec_x * scale_factor * 1.1, vec_y * scale_factor * 1.1, label_txt, 
                    color='darkred', fontsize=9, fontweight='bold')
        texts.append(t)

    # -------------------------------------------------------
    # 装飾
    # -------------------------------------------------------
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
    
    ax.set_xlabel(f"Mode {idx_x} (λ={eig_x.real:.2f}{eig_x.imag:+.2f}j)", fontsize=12)
    ax.set_ylabel(f"Mode {idx_y} (λ={eig_y.real:.2f}{eig_y.imag:+.2f}j)", fontsize=12)
    ax.set_title(f"Koopman Biplot: Trajectory & Token Weights\nScenario: {scenario['name']}", fontsize=14)
    
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # ラベルが重ならないように調整（optional, matplotlibのバージョンによってはエラーになるのでtry-except）
    try:
        from adjustText import adjust_text
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5))
    except ImportError:
        pass
    
    # 保存
    save_path = os.path.join(out_dir, f"{scenario['name']}_biplot.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved biplot: {save_path}")

# =========================================================
#  Main Pipeline
# =========================================================

def main():
    print("="*60)
    print("Chunked Koopman Analysis Pipeline")
    print("="*60)
    
    # 1. データロード
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
    
    # 2. モデルロード
    print(f"\n2. Loading Model: {MODEL_PATH}")
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    state_dict = ckpt['model_state_dict']
    c = ckpt.get('config', {})
    
    # パラメータ取得
    K = c.get('K', 5)
    L = c.get('L', 5)
    print(f"  Config found: K={K}, L={L}")
    
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
        num_agents=c.get('num_agents', 1), # 学習時の設定に合わせる
        agent_emb_dim=c.get('agent_emb_dim', 16),
        max_stay_count=c.get('max_stay_count', 500),
        stay_emb_dim=c.get('stay_emb_dim', 16),
        holiday_emb_dim=c.get('holiday_emb_dim', 4),
        time_zone_emb_dim=c.get('time_zone_emb_dim', 4),
        event_emb_dim=c.get('event_emb_dim', 4),
    ).to(DEVICE)
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # 3. 解析準備
    analyzer = KoopmanEigenAnalyzer(model)
    analyzer.plot_eigenvalues(os.path.join(OUT_DIR, "eigenvalues.png"))
    
    # 4. シナリオ実行
    print("\n3. Running Scenarios...")
    for scenario in SCENARIOS:
        greedy_rollout_chunked(
            model, analyzer, tokenizer, network, expanded_adj,
            scenario, NUM_ROLLOUT_STEPS, L, DEVICE, OUT_DIR
        )
    
    print("\n" + "="*60)
    print("All done.")
    print(f"Output: {OUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()