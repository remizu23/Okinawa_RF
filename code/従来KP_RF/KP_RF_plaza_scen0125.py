'''
※見づらいので不使用で，各ノードの入力を見るコードに置き換え．0125
【改良版】prefix自動展開 + z_pred_next固有モード分解ヒートマップ追加
'''

import torch
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
from datetime import datetime
import torch.nn.functional as F
import matplotlib.gridspec as gridspec

# ユーザー定義モジュール
from network import Network, expand_adjacency_matrix
from tokenization import Tokenization
from KP_RF import KoopmanRoutesFormer

# =========================================================
#  Config & Settings
# =========================================================

# 【新仕様】1つのシナリオを定義するだけで、prefixが自動展開される
BASE_SCENARIOS = [
    {
        "name": "自己回帰ID70",  # ベース名（自動展開時に使用）
        "full_prefix": [11,8,9,28,12,9,29,29,29],  # 最長のprefix
        "holiday": 1,              
        "time_zone": 0,            
        "agent_id": 0,
        "plaza_node_tokens": [11]  # 全展開パターンで共通
    },
    # {
    #     "name": "0,4,11",  # ベース名（自動展開時に使用）
    #     "full_prefix": [0,4,11,30,30,30],  # 最長のprefix
    #     "holiday": 1,              
    #     "time_zone": 0,            
    #     "agent_id": 0,
    #     "plaza_node_tokens": [11, 30]  # 全展開パターンで共通
    # },
    # {
    #     "name": "6,5,4,11",  # ベース名（自動展開時に使用）
    #     "full_prefix": [6, 5, 4, 11, 30, 30, 30],  # 最長のprefix
    #     "holiday": 1,              
    #     "time_zone": 0,            
    #     "agent_id": 0,
    #     "plaza_node_tokens": [11, 30]  # 全展開パターンで共通
    # },
    # {
    #     "name": "9,8,11",  # ベース名（自動展開時に使用）
    #     "full_prefix": [9,8,11,30,30,30],  # 最長のprefix
    #     "holiday": 1,              
    #     "time_zone": 0,            
    #     "agent_id": 0,
    #     "plaza_node_tokens": [11, 30]  # 全展開パターンで共通
    # },
    # {
    #     "name": "12,10,9,8,11",  # ベース名（自動展開時に使用）
    #     "full_prefix": [12,10,9,8,11,30,30,30],  # 最長のprefix
    #     "holiday": 1,              
    #     "time_zone": 0,            
    #     "agent_id": 0,
    #     "plaza_node_tokens": [11, 30]  # 全展開パターンで共通
    # },
    # {
    #     "name": "10,9,8,11",  # ベース名（自動展開時に使用）
    #     "full_prefix": [0,9,8,11,30,30,30],  # 最長のprefix
    #     "holiday": 1,              
    #     "time_zone": 0,            
    #     "agent_id": 0,
    #     "plaza_node_tokens": [11, 30]  # 全展開パターンで共通
    # },
    # {
    #     "name": "18,16,14",  # ベース名（自動展開時に使用）
    #     "full_prefix": [18,16,14,33,33,33],  # 最長のprefix
    #     "holiday": 1,              
    #     "time_zone": 0,            
    #     "agent_id": 0,
    #     "plaza_node_tokens": [14, 33]  # 全展開パターンで共通
    # },
    # {
    #     "name": "1,5,6,14",  # ベース名（自動展開時に使用）
    #     "full_prefix": [1,5,6,14,33,33,33],  # 最長のprefix
    #     "holiday": 1,              
    #     "time_zone": 0,            
    #     "agent_id": 0,
    #     "plaza_node_tokens": [14, 33]  # 全展開パターンで共通
    # },
]

# 出力先
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"/home/mizutani/projects/RF/runs/20260124_214854/plaza_eigen_{run_id}"
os.makedirs(OUT_DIR, exist_ok=True)

# データパス
ADJ_PATH = '/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt'
MODEL_PATH = '/home/mizutani/projects/RF/runs/20260124_214854/model_weights_20260124_214854.pth'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
#  Helper Functions
# =========================================================

def expand_prefix_scenarios(base_scenario):
    """
    1つのベースシナリオから、prefixを段階的に展開した複数シナリオを生成
    
    例: full_prefix=[1,5,6,14,33,33] の場合
    → [1], [1,5], [1,5,6], [1,5,6,14], [1,5,6,14,33], [1,5,6,14,33,33]
    の6パターンを生成
    """
    full_prefix = base_scenario["full_prefix"]
    base_name = base_scenario["name"]
    
    expanded_scenarios = []
    for i in range(1, len(full_prefix) + 1):
        scenario = {
            "name": f"{base_name}_step{i}",
            "prefix": full_prefix[:i],
            "holiday": base_scenario["holiday"],
            "time_zone": base_scenario["time_zone"],
            "agent_id": base_scenario["agent_id"],
            "plaza_node_tokens": base_scenario["plaza_node_tokens"]
        }
        expanded_scenarios.append(scenario)
    
    return expanded_scenarios

def load_model_and_network(model_path, adj_path):
    print("Loading Resources...")
    adj_matrix = torch.load(adj_path, weights_only=True)
    if adj_matrix.shape[0] == 38:
        base_N = 19
        base_adj = adj_matrix[:base_N, :base_N]
    else:
        base_adj = adj_matrix
        base_N = int(base_adj.shape[0])

    expanded_adj = expand_adjacency_matrix(adj_matrix)
    dummy_feat = torch.zeros((len(adj_matrix), 1))
    network = Network(expanded_adj, torch.cat([dummy_feat, dummy_feat], dim=0))
    
    print(f"Loading Model from {model_path}...")
    ckpt = torch.load(model_path, map_location=DEVICE)
    c = ckpt['config']
    state_dict = ckpt['model_state_dict']

    if 'agent_embedding.weight' in state_dict:
        det_num_agents = state_dict['agent_embedding.weight'].shape[0]
    else:
        det_num_agents = c.get('num_agents', 1)

    if 'stay_embedding.weight' in state_dict:
        det_max_stay = state_dict['stay_embedding.weight'].shape[0] - 1
    else:
        det_max_stay = c.get('max_stay_count', 500)
    
    h_dim = c.get("holiday_emb_dim", 4)
    tz_dim = c.get("time_zone_emb_dim", 4)
    e_dim = c.get("event_emb_dim", 4)
    
    model = KoopmanRoutesFormer(
        vocab_size=c['vocab_size'],
        token_emb_dim=c.get('token_emb_dim', c.get('d_ie', 64)),
        d_model=c.get('d_model', c.get('d_ie', 64)),
        nhead=c.get('nhead', 4),
        num_layers=c.get('num_layers', 3),
        d_ff=c.get('d_ff', 128),
        z_dim=c['z_dim'],
        pad_token_id=network.N,
        dist_mat_base=None, 
        base_N=base_N,
        holiday_emb_dim=h_dim, time_zone_emb_dim=tz_dim, event_emb_dim=e_dim,
        num_agents=det_num_agents, agent_emb_dim=c.get('agent_emb_dim', 16),
        max_stay_count=det_max_stay, stay_emb_dim=c.get('stay_emb_dim', 16)
    )
    
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    
    return model, network, expanded_adj, c

def get_valid_next_tokens(adj_matrix, current_token_id, num_nodes, end_token_id):
    """隣接行列に基づく遷移先 + <e> トークン"""
    if current_token_id >= num_nodes: 
        return np.array([end_token_id]) 
        
    adj = adj_matrix
    if isinstance(adj, torch.Tensor):
        adj = adj.cpu().numpy()
        
    valid_indices = np.where(adj[current_token_id] > 0)[0]
    if end_token_id not in valid_indices:
        valid_indices = np.append(valid_indices, end_token_id)
        
    return valid_indices

def get_token_label(tok, tokenizer):
    if tok in tokenizer.SPECIAL_TOKENS.values(): 
        for k, v in tokenizer.SPECIAL_TOKENS.items():
            if v == tok: return k
        return "*"
    return str(tok)

# =========================================================
#  Eigen Analysis Class
# =========================================================

class EigenAnalyzer:
    def __init__(self, model):
        self.model = model
        self.z_dim = model.to_z.out_features
        
        # 1. 固有値分解 A = V Lambda V^{-1}
        A_np = model.A.detach().cpu().numpy()
        eigvals, eigvecs = scipy.linalg.eig(A_np)
        
        # ソート: 絶対値の大きい順 (降順)
        sort_perm = np.argsort(np.abs(eigvals))[::-1]
        self.eigvals = eigvals[sort_perm] # [16]
        self.V = eigvecs[:, sort_perm]    # [16, 16]
        
        # 逆行列 V^{-1}
        self.V_inv = scipy.linalg.inv(self.V)
        
        # 出力重み行列 W (Vocab x z_dim)
        self.W = model.to_logits.weight.detach().cpu().numpy()
        
        # モード出力行列 W_modal = W @ V
        self.W_modal = self.W @ self.V # [Vocab, 16]
        
        # 入力行列 B
        self.B = model.B.detach().cpu().numpy()

    def transform_to_mode(self, z):
        return self.V_inv @ z

    def get_input_mode_excitation(self, u):
        """入力 u (実数) が各モードをどれだけ励起するか beta"""
        # delta_z = B u
        delta_z = self.B @ u
        return self.V_inv @ delta_z

    def visualize_eigenvalues(self, out_dir):
        """単位円上の固有値プロット"""
        plt.figure(figsize=(6, 6))
        theta = np.linspace(0, 2*np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), linestyle='--', color='gray', label='Unit Circle')
        plt.scatter(self.eigvals.real, self.eigvals.imag, color='red', s=100, zorder=5, label='Eigenvalues')
        for i, ev in enumerate(self.eigvals):
            plt.text(ev.real, ev.imag, f" λ{i}", fontsize=9)
        plt.xlabel("Real")
        plt.ylabel("Imaginary")
        plt.title("Eigenvalues of Koopman Matrix A")
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.grid(True)
        plt.legend()
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "eigenvalues.png"))
        plt.close()
        print(f"Saved eigenvalue plot to {out_dir}")

# =========================================================
#  Analysis Functions
# =========================================================

def analyze_scenario_decomposition(model, analyzer, network, adj_matrix, scenarios, out_dir):
    """
    各シナリオに対して、固有モード分解を用いた詳細分析を実施
    【既存の8枚組可視化】
    """
    tokenizer = Tokenization(network)
    # <e>トークンは明示的に40→39
    end_token_id = 39
    if "<e>" in tokenizer.SPECIAL_TOKENS:
        end_token_id = tokenizer.SPECIAL_TOKENS["<e>"]
    print(f"End token ID: {end_token_id}")
    
    print(f"\n=== Analyzing {len(scenarios)} Scenarios ===")
    
    # ==========================================
    # Phase 1: Data Collection (Global Scale用)
    # ==========================================
    results = []
    all_left_heatmap_vals = []
    all_right_heatmap_vals = []
    all_prob_vals = []
    all_prob_diff_vals = []

    for sc in scenarios:
        print(f"Processing: {sc['name']}")
        prefix = sc['prefix']
        agent_id = sc['agent_id']
        holiday = sc['holiday']
        time_zone = sc['time_zone']
        plaza_tokens = sc.get('plaza_node_tokens', [])

        # Forward Pass for Plaza ON
        seq_len = len(prefix)
        tokens_on = torch.tensor([prefix], dtype=torch.long).to(DEVICE)
        stay_counts_on = tokenizer.calculate_stay_counts(tokens_on)
        agent_ids_on = torch.tensor([agent_id], dtype=torch.long).to(DEVICE)
        
        holidays_on = torch.tensor([[holiday] * seq_len], dtype=torch.long).to(DEVICE)
        time_zones_on = torch.tensor([[time_zone] * seq_len], dtype=torch.long).to(DEVICE)
        events_on = torch.zeros((1, seq_len), dtype=torch.long).to(DEVICE)
        
        for pos, tok in enumerate(prefix):
            if tok in plaza_tokens:
                events_on[0, pos] = 1

        with torch.no_grad():
            _, z_hat_on, z_pred_next_on, u_all_on = model(
                tokens_on, stay_counts_on, agent_ids_on, 
                holidays_on, time_zones_on, events_on
            )

        # Forward Pass for Plaza OFF
        events_off = torch.zeros_like(events_on)
        with torch.no_grad():
            _, z_hat_off, z_pred_next_off, u_all_off = model(
                tokens_on, stay_counts_on, agent_ids_on, 
                holidays_on, time_zones_on, events_off
            )

        # Extract Last Step
        last_z_curr_on = z_hat_on[0, -1]
        last_z_curr_off = z_hat_off[0, -1]
        last_u_on = u_all_on[0, -1]
        last_u_off = u_all_off[0, -1]
        last_z_next_on = z_pred_next_on[0, -1]
        last_z_next_off = z_pred_next_off[0, -1]

        # Manual Rollout (既存コード維持)
        if len(prefix) > 1:
            z_curr_on = z_hat_on[0, 0]
            z_curr_off = z_hat_off[0, 0]
            
            for t in range(1, seq_len):
                token_id = torch.tensor([prefix[t]], dtype=torch.long).to(DEVICE)
                stay_count = stay_counts_on[0, t].unsqueeze(0)
                agent_id_t = torch.tensor([agent_id], dtype=torch.long).to(DEVICE)
                
                holiday_t = torch.tensor([holiday], dtype=torch.long).to(DEVICE)
                timezone_t = torch.tensor([time_zone], dtype=torch.long).to(DEVICE)
                event_on_t = torch.tensor([1 if prefix[t] in plaza_tokens else 0], dtype=torch.long).to(DEVICE)
                event_off_t = torch.tensor([0], dtype=torch.long).to(DEVICE)
                
                u_on = model.get_single_step_input(token_id, stay_count, agent_id_t, holiday_t, timezone_t, event_on_t)
                u_off = model.get_single_step_input(token_id, stay_count, agent_id_t, holiday_t, timezone_t, event_off_t)
                
                z_next_on, _ = model.forward_step(z_curr_on.unsqueeze(0), u_on)
                z_next_on = z_next_on.squeeze(0)
                z_next_off, _ = model.forward_step(z_curr_off.unsqueeze(0), u_off)
                z_next_off = z_next_off.squeeze(0)
                
                last_z_curr_on = z_curr_on
                last_z_curr_off = z_curr_off
                last_u_on = u_on
                last_u_off = u_off
                last_z_next_on = z_next_on
                last_z_next_off = z_next_off
            
            z_curr_on = z_next_on
            z_curr_off = z_next_off

        # --- Calculations ---
        last_token = prefix[-1]
        valid_nexts = get_valid_next_tokens(adj_matrix, last_token, network.N, end_token_id)
        if len(valid_nexts) == 0: continue
        valid_labels = [get_token_label(t, tokenizer) for t in valid_nexts]
        n_targets = len(valid_nexts)
        
        # Debug: Check if <e> is included
        print(f"  Scenario: {sc['name']}, Last token: {last_token}, Valid nexts: {valid_nexts}, Labels: {valid_labels}")

        # Convert
        z_curr_on_np = last_z_curr_on.detach().cpu().numpy().flatten()
        z_curr_off_np = last_z_curr_off.detach().cpu().numpy().flatten()
        u_on_np = last_u_on.detach().cpu().numpy().flatten()
        u_off_np = last_u_off.detach().cpu().numpy().flatten()
        z_next_on_np = last_z_next_on.detach().cpu().numpy().flatten()
        
        # Probs
        probs_on = F.softmax(model.to_logits(last_z_next_on), dim=-1).detach().cpu().numpy().flatten()
        probs_off = F.softmax(model.to_logits(last_z_next_off), dim=-1).detach().cpu().numpy().flatten()
        target_probs_on = probs_on[valid_nexts]
        target_probs_diff = target_probs_on - probs_off[valid_nexts]

        # 追加：<e>チェック
        e = end_token_id
        print("end_token_id", e, "label", get_token_label(e, tokenizer))
        print("probs_on[e]      =", float(probs_on[e]))
        print("logit_on[e]      =", float(model.to_logits(last_z_next_on)[e]))
        print("e in valid_nexts =", int(e in set(valid_nexts.tolist() if hasattr(valid_nexts,'tolist') else list(valid_nexts))))

        if e in set(valid_nexts.tolist() if hasattr(valid_nexts,'tolist') else list(valid_nexts)):
            idx = list(valid_nexts).index(e)
            print("target_probs_on[e] =", float(target_probs_on[idx]))


        # Mode Decomposition
        alpha_next_on = analyzer.transform_to_mode(z_next_on_np)
        abs_alpha_next = np.abs(alpha_next_on)
        W_modal_targets = analyzer.W_modal[valid_nexts, :].T 
        
        # State
        state_term_on = analyzer.eigvals * analyzer.transform_to_mode(z_curr_on_np)
        mat_state_on_real = (W_modal_targets * state_term_on[:, np.newaxis]).real
        state_term_off = analyzer.eigvals * analyzer.transform_to_mode(z_curr_off_np)
        mat_state_diff = mat_state_on_real - (W_modal_targets * state_term_off[:, np.newaxis]).real
        
        # Input
        beta_on = analyzer.get_input_mode_excitation(u_on_np)
        mat_input_on_real = (W_modal_targets * beta_on[:, np.newaxis]).real
        beta_off = analyzer.get_input_mode_excitation(u_off_np)
        mat_input_diff = mat_input_on_real - (W_modal_targets * beta_off[:, np.newaxis]).real
        
        # Total
        mat_total_on_real = mat_state_on_real + mat_input_on_real
        mat_total_diff = mat_state_diff + mat_input_diff
        
        # --- Collect Values for Scaling ---
        all_left_heatmap_vals.extend([mat_state_on_real, mat_input_on_real, mat_total_on_real])
        all_right_heatmap_vals.extend([mat_state_diff, mat_input_diff, mat_total_diff])
        all_prob_vals.append(target_probs_on)
        all_prob_diff_vals.append(target_probs_diff)

        # Store for plotting
        results.append({
            'name': sc['name'],
            'valid_labels': valid_labels,
            'n_targets': n_targets,
            'abs_alpha_next': abs_alpha_next,
            'mat_state_on': mat_state_on_real,
            'mat_state_diff': mat_state_diff,
            'mat_input_on': mat_input_on_real,
            'mat_input_diff': mat_input_diff,
            'mat_total_on': mat_total_on_real,
            'mat_total_diff': mat_total_diff,
            'probs_on': target_probs_on,
            'probs_diff': target_probs_diff
        })

    # ==========================================
    # Phase 2: Determine Global Scales
    # ==========================================
    print("Determining Global Scales...")
    
    if all_left_heatmap_vals:
        max_abs_left = max([np.abs(m).max() for m in all_left_heatmap_vals])
    else:
        max_abs_left = 1.0

    if all_right_heatmap_vals:
        max_abs_right = max([np.abs(m).max() for m in all_right_heatmap_vals])
    else:
        max_abs_right = 1.0
        
    if all_prob_vals:
        max_prob_y = max([p.max() for p in all_prob_vals]) * 1.1
    else:
        max_prob_y = 1.0

    if all_prob_diff_vals:
        max_diff_abs = max([np.abs(p).max() for p in all_prob_diff_vals]) * 1.1
    else:
        max_diff_abs = 1.0

    print(f"Global Scales -> Left_HM: +/-{max_abs_left:.3f}, Right_HM: +/-{max_abs_right:.3f}, Prob_Y: {max_prob_y:.3f}, Diff_Y: +/-{max_diff_abs:.3f}")

    # ==========================================
    # Phase 3: Plotting with Unified Scales
    # ==========================================
    eigen_labels = [f"λ={ev.real:.2f}{ev.imag:+.2f}j" for ev in analyzer.eigvals]
    
    for res in results:
        print(f"Plotting: {res['name']}")
        fig = plt.figure(figsize=(28, 36))
        fig.suptitle(f"Mode Decomposition: {res['name']}", fontsize=24, y=0.995, weight='bold')
        gs = gridspec.GridSpec(4, 3, figure=fig, width_ratios=[2, 5, 5], wspace=0.15, hspace=0.35)
        
        valid_labels = res['valid_labels']
        n_targets = res['n_targets']

        # Calculate column sums for each heatmap
        state_on_sum = res['mat_state_on'].sum(axis=0, keepdims=True)  # [1, n_targets]
        state_diff_sum = res['mat_state_diff'].sum(axis=0, keepdims=True)
        input_on_sum = res['mat_input_on'].sum(axis=0, keepdims=True)
        input_diff_sum = res['mat_input_diff'].sum(axis=0, keepdims=True)
        total_on_sum = res['mat_total_on'].sum(axis=0, keepdims=True)
        total_diff_sum = res['mat_total_diff'].sum(axis=0, keepdims=True)

        # Helper for Heatmaps with Total Row
        def plot_hm_with_total(ax, data, total_row, title, vmin, vmax, show_yticks=False):
            # Combine mode data with total row (add a small gap row)
            gap_row = np.full((1, data.shape[1]), np.nan)
            combined_data = np.vstack([data, gap_row, total_row])
            
            # Create labels
            y_labels = eigen_labels + ['', 'Total']
            
            sns.heatmap(combined_data, ax=ax, cmap="coolwarm", center=0, 
                        vmin=vmin, vmax=vmax, cbar=True,
                        yticklabels=y_labels if show_yticks else False,
                        xticklabels=valid_labels,
                        linewidths=0.5, linecolor='white',
                        cbar_kws={'shrink': 0.8})
            
            ax.set_title(title, fontsize=16, weight='bold')
            
            # Draw a thicker line to separate Total row from modes
            ax.axhline(y=data.shape[0] + 1, color='black', linewidth=3)
            
            if show_yticks:
                # Make Total label bold and larger
                labels = ax.get_yticklabels()
                for i, label in enumerate(labels):
                    if i == len(labels) - 1:  # Last label (Total)
                        label.set_weight('bold')
                        label.set_fontsize(13)
                    else:
                        label.set_fontsize(11)
                ax.set_yticklabels(labels, rotation=0)
            
            # Always show x labels
            ax.set_xticklabels(valid_labels, rotation=0, fontsize=13)

        # --- Row 0 (State) ---
        # Left: Empty (Alpha removed)
        
        # Middle: State ON
        plot_hm_with_total(fig.add_subplot(gs[0, 1]), res['mat_state_on'], state_on_sum,
                          "① State Contrib [ON]", 
                          vmin=-max_abs_left, vmax=max_abs_left, show_yticks=True)
        
        # Right: State Diff
        plot_hm_with_total(fig.add_subplot(gs[0, 2]), res['mat_state_diff'], state_diff_sum,
                          "② State Diff", 
                          vmin=-max_abs_right, vmax=max_abs_right, show_yticks=False)

        # --- Row 1 (Input) ---
        plot_hm_with_total(fig.add_subplot(gs[1, 1]), res['mat_input_on'], input_on_sum,
                          "③ Input Contrib [ON]", 
                          vmin=-max_abs_left, vmax=max_abs_left, show_yticks=True)
        
        plot_hm_with_total(fig.add_subplot(gs[1, 2]), res['mat_input_diff'], input_diff_sum,
                          "④ Input Diff", 
                          vmin=-max_abs_right, vmax=max_abs_right, show_yticks=False)

        # --- Row 2 (Total) ---
        plot_hm_with_total(fig.add_subplot(gs[2, 1]), res['mat_total_on'], total_on_sum,
                          "⑤ Total Contrib [ON]", 
                          vmin=-max_abs_left, vmax=max_abs_left, show_yticks=True)
        
        plot_hm_with_total(fig.add_subplot(gs[2, 2]), res['mat_total_diff'], total_diff_sum,
                          "⑥ Total Diff", 
                          vmin=-max_abs_right, vmax=max_abs_right, show_yticks=False)

        # --- Row 3 (Bars) ---
        ax7 = fig.add_subplot(gs[3, 1])
        bars1 = ax7.bar(range(n_targets), res['probs_on'], width=0.45, color='skyblue', edgecolor='black')
        ax7.set_title("Next Prob (ON)", fontsize=16)
        ax7.set_xticks(range(n_targets))
        ax7.set_xticklabels(valid_labels, rotation=0, fontsize=13)
        ax7.set_ylim(0, max_prob_y)
        for b in bars1: 
            ax7.text(b.get_x()+b.get_width()/2, b.get_height(), f"{b.get_height():.3f}", ha='center', va='bottom', fontsize=12)

        ax8 = fig.add_subplot(gs[3, 2])
        bars2 = ax8.bar(range(n_targets), res['probs_diff'], width=0.45,
                        color=['salmon' if x>=0 else 'lightblue' for x in res['probs_diff']], edgecolor='black')
        ax8.axhline(0, color='gray')
        ax8.set_title("Prob Diff (ON-OFF)", fontsize=16)
        ax8.set_xticks(range(n_targets))
        ax8.set_xticklabels(valid_labels, rotation=0, fontsize=13)
        ax8.set_ylim(-max_diff_abs, max_diff_abs)
        for b in bars2: 
            offset = max_diff_abs * 0.02
            va = 'bottom' if b.get_height() >= 0 else 'top'
            y_pos = b.get_height() + (offset if b.get_height() >= 0 else -offset)
            ax8.text(b.get_x()+b.get_width()/2, y_pos, f"{b.get_height():.3f}", ha='center', va=va, fontsize=12)
        
        plt.savefig(os.path.join(out_dir, f"decomposition_{res['name']}.png"), bbox_inches='tight')
        plt.close()
    
    print(f"Saved scenario decomposition plots to {out_dir}")


def visualize_token_selection_probability_heatmap(model, analyzer, network, base_scenario, out_dir):
    """
    【新規実装】トークン選択確率の時系列ヒートマップ
    
    縦軸: 全トークン（vocab_size）
    横軸: ステップ数（1〜len(full_prefix)）
    セルの値: prefix[:step]を入力したときの、次にトークンiが選択される確率
    
    3枚組: Plaza ON / Plaza OFF / Difference
    """
    print(f"\n=== Visualizing Token Selection Probability Heatmap ===")
    print(f"Base Scenario: {base_scenario['name']}")
    
    tokenizer = Tokenization(network)
    full_prefix = base_scenario['full_prefix']
    agent_id = base_scenario['agent_id']
    holiday = base_scenario['holiday']
    time_zone = base_scenario['time_zone']
    plaza_tokens = base_scenario.get('plaza_node_tokens', [])
    
    num_steps = len(full_prefix)
    vocab_size = model.to_logits.out_features
    
    # 確率行列の初期化 [vocab_size, num_steps]
    prob_matrix_on = np.zeros((vocab_size, num_steps))
    prob_matrix_off = np.zeros((vocab_size, num_steps))
    
    for step_idx in range(num_steps):
        current_prefix = full_prefix[:step_idx + 1]
        seq_len = len(current_prefix)
        
        # Prepare tokens
        tokens = torch.tensor([current_prefix], dtype=torch.long).to(DEVICE)
        stay_counts = tokenizer.calculate_stay_counts(tokens)
        agent_ids = torch.tensor([agent_id], dtype=torch.long).to(DEVICE)
        
        holidays = torch.tensor([[holiday] * seq_len], dtype=torch.long).to(DEVICE)
        time_zones = torch.tensor([[time_zone] * seq_len], dtype=torch.long).to(DEVICE)
        
        # Plaza ON
        events_on = torch.zeros((1, seq_len), dtype=torch.long).to(DEVICE)
        for pos, tok in enumerate(current_prefix):
            if tok in plaza_tokens:
                events_on[0, pos] = 1
        
        with torch.no_grad():
            logits_on, _, _, _ = model(
                tokens, stay_counts, agent_ids, 
                holidays, time_zones, events_on
            )
            # 最後のステップのlogitsから確率を計算
            probs_on = F.softmax(logits_on[0, -1], dim=-1).detach().cpu().numpy()
        
        # Plaza OFF
        events_off = torch.zeros_like(events_on)
        with torch.no_grad():
            logits_off, _, _, _ = model(
                tokens, stay_counts, agent_ids, 
                holidays, time_zones, events_off
            )
            probs_off = F.softmax(logits_off[0, -1], dim=-1).detach().cpu().numpy()
        
        prob_matrix_on[:, step_idx] = probs_on
        prob_matrix_off[:, step_idx] = probs_off
    
    # Difference matrix
    prob_matrix_diff = prob_matrix_on - prob_matrix_off
    
    # ==========================================
    # Plotting
    # ==========================================
    step_labels = [f"Step{i+1}" for i in range(num_steps)]
    
    # トークンラベルの作成
    token_labels = []
    for i in range(vocab_size):
        label = get_token_label(i, tokenizer)
        token_labels.append(label)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, max(12, vocab_size * 0.3)))
    fig.suptitle(f"Token Selection Probability: {base_scenario['name']}", 
                 fontsize=20, weight='bold', y=0.995)
    
    # 確率の最大値を取得（カラースケール用）
    max_prob_onoff = max(prob_matrix_on.max(), prob_matrix_off.max())
    max_prob_diff = np.abs(prob_matrix_diff).max()
    
    # Plot 1: Plaza ON
    sns.heatmap(prob_matrix_on, ax=axes[0], cmap="YlOrRd", 
                vmin=0, vmax=max_prob_onoff,
                xticklabels=step_labels, yticklabels=token_labels,
                cbar_kws={'label': 'Probability'})
    axes[0].set_title("Plaza ON", fontsize=16, weight='bold')
    axes[0].set_xlabel("Time Steps", fontsize=14)
    axes[0].set_ylabel("Token ID", fontsize=14)
    
    # Plot 2: Plaza OFF
    sns.heatmap(prob_matrix_off, ax=axes[1], cmap="YlOrRd",
                vmin=0, vmax=max_prob_onoff,
                xticklabels=step_labels, yticklabels=token_labels,
                cbar_kws={'label': 'Probability'})
    axes[1].set_title("Plaza OFF", fontsize=16, weight='bold')
    axes[1].set_xlabel("Time Steps", fontsize=14)
    axes[1].set_ylabel("Token ID", fontsize=14)
    
    # Plot 3: Difference (ON - OFF)
    sns.heatmap(prob_matrix_diff, ax=axes[2], cmap="coolwarm", center=0,
                vmin=-max_prob_diff, vmax=max_prob_diff,
                xticklabels=step_labels, yticklabels=token_labels,
                cbar_kws={'label': 'Probability Diff'})
    axes[2].set_title("Difference (ON - OFF)", fontsize=16, weight='bold')
    axes[2].set_xlabel("Time Steps", fontsize=14)
    axes[2].set_ylabel("Token ID", fontsize=14)
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"token_selection_prob_{base_scenario['name']}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Saved token selection probability heatmap: {save_path}")
    
    # CSVでも保存
    import pandas as pd
    
    df_on = pd.DataFrame(prob_matrix_on.T, 
                         columns=[f"Token{i}_{get_token_label(i, tokenizer)}" for i in range(vocab_size)],
                         index=[f"Step{i+1}" for i in range(num_steps)])
    df_on.to_csv(os.path.join(out_dir, f"token_selection_prob_ON_{base_scenario['name']}.csv"))
    
    df_off = pd.DataFrame(prob_matrix_off.T, 
                          columns=[f"Token{i}_{get_token_label(i, tokenizer)}" for i in range(vocab_size)],
                          index=[f"Step{i+1}" for i in range(num_steps)])
    df_off.to_csv(os.path.join(out_dir, f"token_selection_prob_OFF_{base_scenario['name']}.csv"))
    
    df_diff = pd.DataFrame(prob_matrix_diff.T, 
                           columns=[f"Token{i}_{get_token_label(i, tokenizer)}" for i in range(vocab_size)],
                           index=[f"Step{i+1}" for i in range(num_steps)])
    df_diff.to_csv(os.path.join(out_dir, f"token_selection_prob_DIFF_{base_scenario['name']}.csv"))
    
    print(f"Saved CSV files for token selection probability")


def visualize_z_pred_next_eigen_progression(model, analyzer, network, base_scenario, out_dir):
    """
    【新規実装】z_pred_nextの固有モード分解を時系列でヒートマップ化
    
    最長prefixのシナリオに対して、各ステップでのz_pred_nextをモード分解し、
    縦16次元（固有モード） × 横ステップ数 のヒートマップを生成
    """
    print(f"\n=== Visualizing z_pred_next Eigen Progression ===")
    print(f"Base Scenario: {base_scenario['name']}")
    
    tokenizer = Tokenization(network)
    full_prefix = base_scenario['full_prefix']
    agent_id = base_scenario['agent_id']
    holiday = base_scenario['holiday']
    time_zone = base_scenario['time_zone']
    plaza_tokens = base_scenario.get('plaza_node_tokens', [])
    
    num_steps = len(full_prefix)
    
    # 各ステップでのz_pred_nextを収集（Plaza ON/OFF両方）
    alpha_matrix_on = np.zeros((analyzer.z_dim, num_steps))
    alpha_matrix_off = np.zeros((analyzer.z_dim, num_steps))
    
    for step_idx in range(num_steps):
        current_prefix = full_prefix[:step_idx + 1]
        seq_len = len(current_prefix)
        
        # Prepare tokens
        tokens = torch.tensor([current_prefix], dtype=torch.long).to(DEVICE)
        stay_counts = tokenizer.calculate_stay_counts(tokens)
        agent_ids = torch.tensor([agent_id], dtype=torch.long).to(DEVICE)
        
        holidays = torch.tensor([[holiday] * seq_len], dtype=torch.long).to(DEVICE)
        time_zones = torch.tensor([[time_zone] * seq_len], dtype=torch.long).to(DEVICE)
        
        # Plaza ON
        events_on = torch.zeros((1, seq_len), dtype=torch.long).to(DEVICE)
        for pos, tok in enumerate(current_prefix):
            if tok in plaza_tokens:
                events_on[0, pos] = 1
        
        with torch.no_grad():
            _, _, z_pred_next_on, _ = model(
                tokens, stay_counts, agent_ids, 
                holidays, time_zones, events_on
            )
        
        # Plaza OFF
        events_off = torch.zeros_like(events_on)
        with torch.no_grad():
            _, _, z_pred_next_off, _ = model(
                tokens, stay_counts, agent_ids, 
                holidays, time_zones, events_off
            )
        
        # 最後のステップのz_pred_nextを取得
        z_pred_on = z_pred_next_on[0, -1].detach().cpu().numpy().flatten()
        z_pred_off = z_pred_next_off[0, -1].detach().cpu().numpy().flatten()
        
        # 固有モード分解
        alpha_on = analyzer.transform_to_mode(z_pred_on)
        alpha_off = analyzer.transform_to_mode(z_pred_off)
        
        alpha_matrix_on[:, step_idx] = alpha_on.real
        alpha_matrix_off[:, step_idx] = alpha_off.real
    
    # Difference matrix
    alpha_matrix_diff = alpha_matrix_on - alpha_matrix_off
    
    # ==========================================
    # Plotting
    # ==========================================
    eigen_labels = [f"λ{i}\n{ev.real:.2f}{ev.imag:+.2f}j" for i, ev in enumerate(analyzer.eigvals)]
    step_labels = [f"Step{i+1}\n{full_prefix[:i+1]}" for i in range(num_steps)]
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 10))
    fig.suptitle(f"z_pred_next Eigen Mode Progression: {base_scenario['name']}", 
                 fontsize=20, weight='bold', y=0.98)
    
    # Determine global scale for ON and OFF (same scale)
    max_abs_onoff = max(np.abs(alpha_matrix_on).max(), np.abs(alpha_matrix_off).max())
    max_abs_diff = np.abs(alpha_matrix_diff).max()
    
    # Plot 1: Plaza ON
    sns.heatmap(alpha_matrix_on, ax=axes[0], cmap="coolwarm", center=0,
                vmin=-max_abs_onoff, vmax=max_abs_onoff,
                xticklabels=step_labels, yticklabels=eigen_labels,
                cbar_kws={'label': 'Mode Amplitude'}, annot=True, fmt='.2f')
    axes[0].set_title("Plaza ON", fontsize=16, weight='bold')
    axes[0].set_xlabel("Time Steps", fontsize=14)
    axes[0].set_ylabel("Eigen Modes", fontsize=14)
    
    # Plot 2: Plaza OFF
    sns.heatmap(alpha_matrix_off, ax=axes[1], cmap="coolwarm", center=0,
                vmin=-max_abs_onoff, vmax=max_abs_onoff,
                xticklabels=step_labels, yticklabels=eigen_labels,
                cbar_kws={'label': 'Mode Amplitude'}, annot=True, fmt='.2f')
    axes[1].set_title("Plaza OFF", fontsize=16, weight='bold')
    axes[1].set_xlabel("Time Steps", fontsize=14)
    axes[1].set_ylabel("Eigen Modes", fontsize=14)
    
    # Plot 3: Difference (ON - OFF)
    sns.heatmap(alpha_matrix_diff, ax=axes[2], cmap="coolwarm", center=0,
                vmin=-max_abs_diff, vmax=max_abs_diff,
                xticklabels=step_labels, yticklabels=eigen_labels,
                cbar_kws={'label': 'Mode Amplitude Diff'}, annot=True, fmt='.2f')
    axes[2].set_title("Difference (ON - OFF)", fontsize=16, weight='bold')
    axes[2].set_xlabel("Time Steps", fontsize=14)
    axes[2].set_ylabel("Eigen Modes", fontsize=14)
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"z_pred_next_eigen_progression_{base_scenario['name']}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Saved z_pred_next eigen progression plot: {save_path}")
    
    # CSVでも保存（オプション）
    import pandas as pd
    
    df_on = pd.DataFrame(alpha_matrix_on.T, 
                         columns=[f"Mode{i}_lambda={analyzer.eigvals[i]:.2f}" for i in range(analyzer.z_dim)],
                         index=[f"Step{i+1}" for i in range(num_steps)])
    df_on.to_csv(os.path.join(out_dir, f"z_pred_next_eigen_ON_{base_scenario['name']}.csv"))
    
    df_off = pd.DataFrame(alpha_matrix_off.T, 
                          columns=[f"Mode{i}_lambda={analyzer.eigvals[i]:.2f}" for i in range(analyzer.z_dim)],
                          index=[f"Step{i+1}" for i in range(num_steps)])
    df_off.to_csv(os.path.join(out_dir, f"z_pred_next_eigen_OFF_{base_scenario['name']}.csv"))
    
    df_diff = pd.DataFrame(alpha_matrix_diff.T, 
                           columns=[f"Mode{i}_lambda={analyzer.eigvals[i]:.2f}" for i in range(analyzer.z_dim)],
                           index=[f"Step{i+1}" for i in range(num_steps)])
    df_diff.to_csv(os.path.join(out_dir, f"z_pred_next_eigen_DIFF_{base_scenario['name']}.csv"))
    
    print(f"Saved CSV files for eigen progression")

# =========================================================
#  Main
# =========================================================

def main():
    model, network, expanded_adj, config = load_model_and_network(MODEL_PATH, ADJ_PATH)
    analyzer = EigenAnalyzer(model)
    
    # 固有値の可視化
    analyzer.visualize_eigenvalues(OUT_DIR)
    
    # 全ベースシナリオに対して処理
    for base_scenario in BASE_SCENARIOS:
        print(f"\n{'='*60}")
        print(f"Processing Base Scenario: {base_scenario['name']}")
        print(f"{'='*60}")
        
        # 1. Prefix自動展開
        expanded_scenarios = expand_prefix_scenarios(base_scenario)
        print(f"Expanded into {len(expanded_scenarios)} scenarios:")
        for sc in expanded_scenarios:
            print(f"  - {sc['name']}: {sc['prefix']}")
        
        # 2. 既存の8枚組可視化（全展開パターン）
        analyze_scenario_decomposition(model, analyzer, network, expanded_adj, 
                                      expanded_scenarios, OUT_DIR)
        
        # 3. 新規ヒートマップ（最長パターンのみ）
        visualize_z_pred_next_eigen_progression(model, analyzer, network, 
                                               base_scenario, OUT_DIR)
        
        # 4. トークン選択確率ヒートマップ（最長パターンのみ）
        visualize_token_selection_probability_heatmap(model, analyzer, network,
                                                     base_scenario, OUT_DIR)
    
    print(f"\n{'='*60}")
    print(f"All analyses completed!")
    print(f"Output directory: {OUT_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
