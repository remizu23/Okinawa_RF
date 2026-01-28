"""
Koopman Mode Decomposition Analysis for DKP_RF (Original / No-Jump)
With Biplot (Eigen-space & PCA), All-Token Display, and Auto-Scaling.
"""

import torch
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
import os
from datetime import datetime

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
MODEL_PATH = "/home/mizutani/projects/RF/runs/20260128_201544/model_weights_20260128_201544.pth"
ADJ_PATH = "/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt"
DATA_PATH = "/home/mizutani/projects/RF/data/input_real_m5.npz"

# 出力先
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(os.path.dirname(MODEL_PATH), f"scen_{RUN_ID}")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ROLLOUT_STEPS = 30


# =========================================================
#  Utility Functions
# =========================================================

def get_movable_tokens(current_node, adj_matrix, pad_token_id=38, end_token_id=39):
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
    def __init__(self, model):
        self.model = model
        self.z_dim = model.z_dim
        
        A_np = model.A.detach().cpu().numpy()
        eigvals, eigvecs = scipy.linalg.eig(A_np)
        
        # 固有値の絶対値が大きい順にソート
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
#  Visualization Functions (Biplot & PCA)
# =========================================================

def plot_tokens_on_ax(ax, token_vecs, token_indices, tokenizer, scale_factor, color='gray', alpha=0.6):
    """
    指定されたAxesにトークンをプロットするヘルパー関数
    token_vecs: [N_tokens, 2] (X, Y座標)
    """
    texts = []
    # 全トークンを表示 (0 ~ 39)
    for idx in token_indices:
        if idx >= len(token_vecs): continue
        
        vec = token_vecs[idx]
        x, y = vec[0] * scale_factor, vec[1] * scale_factor
        
        # 点としてプロット
        ax.plot(x, y, '.', color=color, markersize=3, alpha=alpha)
        
        # ラベル
        label_txt = get_token_label(idx, tokenizer)
        t = ax.text(x, y, label_txt, fontsize=6, color='black', alpha=0.8)
        texts.append(t)
    
    return texts


def visualize_all_modes_biplot(analyzer, step_data, scenario, tokenizer, out_dir):
    """
    全固有モードをペアリングしてBiplotを作成し、1枚の画像にまとめる
    
    ロジック:
    - 固有値を大きい順に走査
    - 複素共役ペア (Im > eps) -> [Re(α), Im(α)] としてプロット (1つのサブプロットで2次元分消費)
    - 実数 (Im approx 0) -> 次の実数とペアにして [Re(α_i), Re(α_j)] (2次元分消費)
    """
    z_dim = analyzer.z_dim
    # 軌跡データ取得
    traj_alpha_with = np.array([d['alpha_with'] for d in step_data]).reshape(len(step_data), z_dim)
    traj_alpha_without = np.array([d['alpha_without'] for d in step_data]).reshape(len(step_data), z_dim)
    
    # ペアリングロジック
    pairs = [] # List of (idx_x, idx_y, type='complex'|'real', lambda_info)
    used_indices = set()
    
    for i in range(z_dim):
        if i in used_indices:
            continue
        
        eig = analyzer.eigvals[i]
        
        # 複素数の場合 (虚部が一定以上)
        if abs(eig.imag) > 1e-5:
            # Re vs Im
            pairs.append({
                'x_idx': i, 'y_idx': i, # 同じインデックスだが Re vs Im
                'type': 'complex',
                'eig': eig
            })
            used_indices.add(i)
            # 共役ペアと思われる次のインデックスも使用済みにする（厳密には確認すべきだが簡略化）
            # 通常、scipy.linalg.eigは共役を近くに出すことが多いが、絶対値ソートしているので隣り合うはず
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
                # 相手がいない場合 (奇数個の実数モード)、単独でRe vs Stepにするか、とりあえず Re vs 0 にする
                # ここではスキップまたは Re vs Im(0) として表示
                pairs.append({
                    'x_idx': i, 'y_idx': i, # Re vs Im (Imは0だが)
                    'type': 'real_single',
                    'eig': eig
                })
                used_indices.add(i)

    # プロット作成
    num_plots = len(pairs)
    cols = 4
    rows = (num_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    if rows == 1 and cols == 1: axes = [axes]
    axes = np.array(axes).flatten()
    
    # トークンインデックス全取得
    all_tokens = list(range(40)) # 0..39 (End=39)
    
    for plot_idx, pair_info in enumerate(pairs):
        ax = axes[plot_idx]
        
        # X, Yデータの抽出
        if pair_info['type'] == 'complex':
            idx = pair_info['x_idx']
            x_vals_with = traj_alpha_with[:, idx].real
            y_vals_with = traj_alpha_with[:, idx].imag
            x_vals_wo = traj_alpha_without[:, idx].real
            y_vals_wo = traj_alpha_without[:, idx].imag
            
            # W (トークン) も複素平面に射影 (W_modal[k, idx] の Re vs Im)
            W_vecs = np.column_stack([
                analyzer.W_modal[:, idx].real,
                analyzer.W_modal[:, idx].imag
            ])
            
            title = f"Mode {idx} (Complex)\nλ={pair_info['eig'].real:.2f}±{abs(pair_info['eig'].imag):.2f}j"
            xlabel, ylabel = "Real", "Imag"
            
        else: # real or real_single
            idx_x = pair_info['x_idx']
            idx_y = pair_info['y_idx']
            
            x_vals_with = traj_alpha_with[:, idx_x].real
            y_vals_with = traj_alpha_with[:, idx_y].real if pair_info['type'] == 'real' else np.zeros_like(x_vals_with)
            
            x_vals_wo = traj_alpha_without[:, idx_x].real
            y_vals_wo = traj_alpha_without[:, idx_y].real if pair_info['type'] == 'real' else np.zeros_like(x_vals_wo)
            
            # W
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

        # スケーリング計算
        traj_max = max(np.max(np.abs(x_vals_with)), np.max(np.abs(y_vals_with)))
        W_max = np.max(np.abs(W_vecs)) if W_vecs.size > 0 else 1.0
        scale_factor = (traj_max / W_max) * 0.8 if W_max > 0 else 1.0
        
        # プロット: 軌跡
        ax.plot(x_vals_with, y_vals_with, 'o-', color='steelblue', label='With', markersize=3, alpha=0.7)
        ax.plot(x_vals_with[0], y_vals_with[0], 'D', color='green', markersize=5) # Start
        ax.plot(x_vals_wo, y_vals_wo, '--', color='gray', alpha=0.4, label='W/o')
        
        # プロット: トークン (全表示 + 自動スケーリング)
        plot_tokens_on_ax(ax, W_vecs, all_tokens, tokenizer, scale_factor)
        
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.axhline(0, color='k', ls=':', lw=0.5)
        ax.axvline(0, color='k', ls=':', lw=0.5)
        ax.grid(True, alpha=0.3)
        if plot_idx == 0: ax.legend(fontsize=6)

    # 余ったサブプロットを消す
    for k in range(num_plots, len(axes)):
        axes[k].axis('off')
        
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"{scenario['name']}_eigen_biplot_all.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Eigen Biplot: {save_path}")


def visualize_pca_biplot(step_data, scenario, tokenizer, model, out_dir):
    """
    PCAによるBiplot (2D & 3D) - サンプル数不足対応版
    zの軌跡と、Wの射影を表示
    """
    # データが空の場合はスキップ
    if not step_data:
        print(f"Skipping PCA: No step data for {scenario['name']}")
        return

    # zデータの収集
    z_with_list = [d['z_with'] for d in step_data]
    z_without_list = [d['z_without'] for d in step_data]
    
    # shape: [T, z_dim]
    Z_with = np.array(z_with_list)
    Z_without = np.array(z_without_list)
    
    # 結合してfit
    Z_all = np.concatenate([Z_with, Z_without], axis=0)
    
    # サンプル数チェック
    n_samples, n_features = Z_all.shape
    
    # PCAのコンポーネント数は、サンプル数と特徴量数の最小値を超えられない
    # 3次元プロットを目指すが、データ不足なら2次元以下に落とす
    n_components = min(3, n_samples, n_features)
    
    if n_components < 2:
        print(f"Skipping PCA for {scenario['name']}: Not enough samples (n={n_samples}) for 2D plot.")
        return

    # PCA実行
    pca = PCA(n_components=n_components)
    pca.fit(Z_all)
    
    # Transform Z
    Z_with_pca = pca.transform(Z_with)
    Z_without_pca = pca.transform(Z_without)
    
    # Transform W (トークン重み)
    # W_pca = W @ V_pca.T
    # components_ は (n_components, n_features) なので転置して掛ける
    W_weight = model.to_logits.weight.detach().cpu().numpy() # [vocab, z_dim]
    W_pca = W_weight @ pca.components_.T # [vocab, n_components]
    
    # 全トークンインデックス
    all_tokens = list(range(40))
    
    # スケーリング計算 (2次元ベース)
    traj_max = np.max(np.abs(Z_with_pca[:, :2]))
    W_max = np.max(np.abs(W_pca[:, :2]))
    scale_factor = (traj_max / W_max) * 0.8 if W_max > 0 else 1.0
    
    # --- 2D Plot (PC1 vs PC2) ---
    # n_components >= 2 なので必ず実行可能
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 軌跡
    ax.plot(Z_with_pca[:, 0], Z_with_pca[:, 1], 'o-', color='purple', label='With Plaza', markersize=4, alpha=0.8)
    ax.plot(Z_with_pca[0, 0], Z_with_pca[0, 1], 'D', color='green', markersize=6, label='Start')
    ax.plot(Z_without_pca[:, 0], Z_without_pca[:, 1], '--', color='gray', label='W/o Plaza', alpha=0.5)
    
    # トークン
    plot_tokens_on_ax(ax, W_pca[:, :2], all_tokens, tokenizer, scale_factor, color='red', alpha=0.5)
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    ax.set_title(f"PCA Biplot (2D): {scenario['name']}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    save_path_2d = os.path.join(out_dir, f"{scenario['name']}_pca_biplot_2d.png")
    plt.savefig(save_path_2d, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved PCA 2D: {save_path_2d}")
    
    # --- 3D Plot ---
    # n_components >= 3 の場合のみ実行
    if n_components >= 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 軌跡
        ax.plot(Z_with_pca[:, 0], Z_with_pca[:, 1], Z_with_pca[:, 2], 'o-', color='purple', label='With', markersize=3)
        ax.plot(Z_without_pca[:, 0], Z_without_pca[:, 1], Z_without_pca[:, 2], '--', color='gray', label='W/o', alpha=0.5)
        
        # トークン (3D) - ドットのみ
        W_pca_scaled = W_pca * scale_factor
        ax.scatter(W_pca_scaled[:, 0], W_pca_scaled[:, 1], W_pca_scaled[:, 2], color='red', s=5, alpha=0.5)
        
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(f"PCA Biplot (3D): {scenario['name']}")
        
        save_path_3d = os.path.join(out_dir, f"{scenario['name']}_pca_biplot_3d.png")
        plt.savefig(save_path_3d, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved PCA 3D: {save_path_3d}")
    else:
        print(f"Skipping PCA 3D for {scenario['name']}: Not enough dimensions (n_components={n_components})")

# =========================================================
#  Scenario Analysis
# =========================================================

def encode_prefix_with_plaza(model, tokenizer, prefix, agent_id, holiday, time_zone, 
                              plaza_tokens, use_plaza, device):
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
    
    A_np = analyzer.model.A.detach().cpu().numpy().T
    
    for step in range(num_steps):
        # z_t+1 = A @ z_t
        z_next_with = A_np @ z_with
        z_next_without = A_np @ z_without
        
        # 固有空間射影
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
        })
        
        z_with = z_next_with
        z_without = z_next_without
        
        if next_token_with < 19:
            current_node = next_token_with
        elif 19 <= next_token_with <= 37:
            current_node = next_token_with - 19
        else:
            break
            
    # --- Visualization Calls ---
    
    # 1. 固有空間 Biplot (全モード網羅)
    visualize_all_modes_biplot(analyzer, step_data, scenario, tokenizer, out_dir)
    
    # 2. PCA Biplot (2D & 3D)
    visualize_pca_biplot(step_data, scenario, tokenizer, model, out_dir)
    
    print(f"\nGenerated sequence (with plaza): {prefix} -> {generated_with}")


# =========================================================
#  Main Pipeline
# =========================================================

def main():
    print("="*60)
    print("Koopman Analysis Pipeline (Original Model) - Advanced Viz")
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
    
    # エージェント数等の自動取得 (Configにない場合)
    if 'agent_embedding.weight' in state_dict:
        num_agents = state_dict['agent_embedding.weight'].shape[0]
    else:
        num_agents = c.get('num_agents', 1)

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
        num_agents=num_agents,
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