import torch
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import seaborn as sns
import os
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

SCENARIOS = [
    {
        "name": "25から立ち寄り",
        "prefix": [6, 25, 5, 2],       
        "holiday": 1,              
        "time_zone": 0,            
        "agent_id": 0,
        "plaza_node_tokens": [2]
    },
    {
        "name": "1step滞在",
        "prefix": [6, 25, 5, 2, 21],       
        "holiday": 1,              
        "time_zone": 0,            
        "agent_id": 0,
        "plaza_node_tokens": [2, 21]
    },
    {
        "name": "2step滞在",
        "prefix": [6, 25, 5, 2, 21, 21],       
        "holiday": 1,              
        "time_zone": 0,            
        "agent_id": 0,
        "plaza_node_tokens": [2, 21]
    },
    {
        "name": "3step滞在",
        "prefix": [6, 25, 5, 2, 21, 21, 21],       
        "holiday": 1,              
        "time_zone": 0,            
        "agent_id": 0,
        "plaza_node_tokens": [2, 21]
    },
    {
        "name": "4step滞在",
        "prefix": [6, 25, 5, 2, 21, 21, 21, 21],       
        "holiday": 1,              
        "time_zone": 0,            
        "agent_id": 0,
        "plaza_node_tokens": [2, 21]
    },
    {
        "name": "5step滞在",
        "prefix": [6, 25, 5, 2, 21, 21, 21, 21, 21],       
        "holiday": 1,              
        "time_zone": 0,            
        "agent_id": 0,
        "plaza_node_tokens": [2, 21]
    },
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
        
        A_np = model.A.detach().cpu().numpy()
        eigvals, eigvecs = scipy.linalg.eig(A_np)
        
        sort_perm = np.argsort(np.abs(eigvals))[::-1]
        self.eigvals = eigvals[sort_perm]
        self.V = eigvecs[:, sort_perm]
        self.V_inv = scipy.linalg.inv(self.V)
        self.W = model.to_logits.weight.detach().cpu().numpy()
        self.W_modal = self.W @ self.V

    def transform_to_mode(self, z):
        return self.V_inv @ z

    def get_input_mode_excitation(self, u):
        delta_z = self.model.B.detach().cpu().numpy() @ u
        return self.V_inv @ delta_z

    def visualize_eigenvalues(self, out_dir):
        plt.figure(figsize=(6, 6))
        theta = np.linspace(0, 2*np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), linestyle='--', color='gray', label='Unit Circle')
        plt.scatter(self.eigvals.real, self.eigvals.imag, color='blue', alpha=0.7, label='Eigenvalues')
        
        for i, ev in enumerate(self.eigvals):
            plt.text(ev.real, ev.imag, str(i), fontsize=9, ha='right', va='bottom')
            
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.title(f"Eigenvalues of A (Sorted by Magnitude)")
        plt.xlabel("Real Part")
        plt.ylabel("Imaginary Part")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "eigenvalues_unit_circle.png"), dpi=300, bbox_inches='tight')
        plt.close()

# =========================================================
#  Detailed Scenario Analysis
# =========================================================

def run_simulation_step(model, z_curr, token_id, stay_count, agent_id, holiday, time_zone, event_flag):
    token_t = torch.tensor([token_id], device=DEVICE)
    stay_t  = torch.tensor([stay_count], device=DEVICE)
    agent_t = torch.tensor([agent_id], device=DEVICE)
    hol_t   = torch.tensor([holiday], device=DEVICE)
    tz_t    = torch.tensor([time_zone], device=DEVICE)
    evt_t   = torch.tensor([event_flag], device=DEVICE)
    
    u_curr = model.get_single_step_input(token_t, stay_t, agent_t, hol_t, tz_t, evt_t)
    term_A = torch.matmul(z_curr, model.A.t()) 
    term_B = torch.matmul(u_curr, model.B.t())
    z_next = term_A + term_B
    
    return z_next, u_curr

def analyze_scenario_decomposition(model, analyzer, network, adj_matrix, scenarios, out_dir):
    tokenizer = Tokenization(network)
    end_token_id = tokenizer.SPECIAL_TOKENS["<e>"]
    
    analyzer.visualize_eigenvalues(out_dir)

    for sc in scenarios:
        print(f"Analyzing Scenario: {sc['name']}")
        prefix = sc['prefix']
        plaza_nodes = sc['plaza_node_tokens']
        
        z_curr_on = torch.zeros((1, analyzer.z_dim), device=DEVICE)
        z_curr_off = torch.zeros((1, analyzer.z_dim), device=DEVICE)
        current_stay_count = 1
        
        last_z_curr_on = None
        last_z_curr_off = None
        last_u_on = None
        last_u_off = None
        last_z_next_on = None
        last_z_next_off = None
        
        for i, token in enumerate(prefix):
            if i > 0:
                if prefix[i] == prefix[i-1] and (prefix[i] >= network.N): 
                    current_stay_count += 1
                else:
                    current_stay_count = 1 
            
            evt_on_flag = 1 if token in plaza_nodes else 0
            evt_off_flag = 0
            
            ag, hol, tz = sc['agent_id'], sc['holiday'], sc['time_zone']
            
            z_next_on, u_on = run_simulation_step(model, z_curr_on, token, current_stay_count, ag, hol, tz, evt_on_flag)
            z_next_off, u_off = run_simulation_step(model, z_curr_off, token, current_stay_count, ag, hol, tz, evt_off_flag)
            
            if i == len(prefix) - 1:
                last_z_curr_on = z_curr_on
                last_z_curr_off = z_curr_off
                last_u_on = u_on
                last_u_off = u_off
                last_z_next_on = z_next_on
                last_z_next_off = z_next_off
            
            z_curr_on = z_next_on
            z_curr_off = z_next_off

        # Decomposition Analysis
        last_token = prefix[-1]
        valid_nexts = get_valid_next_tokens(adj_matrix, last_token, network.N, end_token_id)
        if len(valid_nexts) == 0: 
            print("No valid transitions.")
            continue
        
        valid_labels = [get_token_label(t, tokenizer) for t in valid_nexts]
        n_targets = len(valid_nexts)
        n_modes = analyzer.z_dim
        
        z_curr_on_np = last_z_curr_on.detach().cpu().numpy().flatten()
        z_curr_off_np = last_z_curr_off.detach().cpu().numpy().flatten()
        u_on_np = last_u_on.detach().cpu().numpy().flatten()
        u_off_np = last_u_off.detach().cpu().numpy().flatten()
        z_next_on_np = last_z_next_on.detach().cpu().numpy().flatten()
        
        logits_on = model.to_logits(last_z_next_on)
        logits_off = model.to_logits(last_z_next_off)
        probs_on = F.softmax(logits_on, dim=-1).detach().cpu().numpy().flatten()
        probs_off = F.softmax(logits_off, dim=-1).detach().cpu().numpy().flatten()
        
        target_probs_on = probs_on[valid_nexts]
        target_probs_diff = target_probs_on - probs_off[valid_nexts]
        
        # Mode Decomposition
        alpha_next_on = analyzer.transform_to_mode(z_next_on_np)
        abs_alpha_next = np.abs(alpha_next_on)
        
        W_modal_targets = analyzer.W_modal[valid_nexts, :].T 
        
        alpha_prev_on = analyzer.transform_to_mode(z_curr_on_np)
        state_term_on = analyzer.eigvals * alpha_prev_on
        mat_state_on = W_modal_targets * state_term_on[:, np.newaxis]
        mat_state_on_real = mat_state_on.real
        
        alpha_prev_off = analyzer.transform_to_mode(z_curr_off_np)
        state_term_off = analyzer.eigvals * alpha_prev_off
        mat_state_off = W_modal_targets * state_term_off[:, np.newaxis]
        mat_state_diff = (mat_state_on - mat_state_off).real
        
        beta_on = analyzer.get_input_mode_excitation(u_on_np)
        mat_input_on = W_modal_targets * beta_on[:, np.newaxis]
        mat_input_on_real = mat_input_on.real
        
        beta_off = analyzer.get_input_mode_excitation(u_off_np)
        mat_input_off = W_modal_targets * beta_off[:, np.newaxis]
        mat_input_diff = (mat_input_on - mat_input_off).real
        
        mat_total_on_real = mat_state_on_real + mat_input_on_real
        mat_total_diff = mat_state_diff + mat_input_diff
        
        # ============================================
        # Visualization (改善版)
        # ============================================
        
        # 図全体を大幅に拡大 (40x28インチ)
        fig = plt.figure(figsize=(40, 28))
        fig.suptitle(f"Mode Decomposition Analysis: {sc['name']}\n(Real Part of Logit Contributions)", 
                     fontsize=24, y=0.995, weight='bold')
        
        # GridSpec: 左列(⓪)+中央列(ON)+右列(OFF/Diff) で2列×4行
        # 列幅の比率: ⓪を2、ON/OFFをそれぞれ5に設定
        gs = gridspec.GridSpec(4, 3, figure=fig, 
                               width_ratios=[2, 5, 5],  # 繰り返し4回
                               wspace=0.35, hspace=0.4,
                               left=0.05, right=0.98, top=0.96, bottom=0.05)
        
        # y軸ラベル: 固有値情報のみ
        eigen_labels = [f"λ={ev.real:.2f}{ev.imag:+.2f}j" for ev in analyzer.eigvals]
        
        def plot_hm(ax, data, title, cmap="coolwarm", center=True, 
                    show_yticks=False, show_xticks=True, annot_size=10):
            if center:
                max_val = np.abs(data).max()
                vmin, vmax = -max_val, max_val
            else:
                vmin, vmax = None, None
                
            sns.heatmap(data, ax=ax, cmap=cmap, center=0 if center else None, 
                        vmin=vmin, vmax=vmax, annot=False, 
                        cbar=True, cbar_kws={'shrink': 0.8})
            ax.set_title(title, fontsize=16, pad=12, weight='bold')
            
            if show_xticks:
                ax.set_xlabel("Target Tokens", fontsize=14)
                ax.set_xticks(np.arange(len(valid_labels)) + 0.5)
                ax.set_xticklabels(valid_labels, rotation=0, fontsize=13)
            else:
                ax.set_xticks([])
                ax.set_xlabel("")
                
            if show_yticks:
                ax.set_yticks(np.arange(n_modes) + 0.5)
                ax.set_yticklabels(eigen_labels, rotation=0, va="center", fontsize=11)
                ax.set_ylabel("")
            else:
                ax.set_yticks([])
                ax.set_ylabel("")

        # --- Row 0: State ---
        # ⓪ |Alpha_next|
        ax0 = fig.add_subplot(gs[0, 0])
        sns.heatmap(abs_alpha_next[:, np.newaxis], ax=ax0, cmap="Reds", 
                    annot=True, fmt=".2f", cbar=False, annot_kws={'size': 10},
                    xticklabels=["|α|"], yticklabels=eigen_labels)
        ax0.set_title("⓪ |Alpha_next|\n(Result State)", fontsize=14, weight='bold')
        ax0.set_yticks(np.arange(n_modes) + 0.5)
        ax0.set_yticklabels(eigen_labels, rotation=0, va="center", fontsize=11)
        ax0.set_xticks([0.5])
        ax0.set_xticklabels(["|α|"], fontsize=12)
        
        # ① State ON
        ax1 = fig.add_subplot(gs[0, 1])
        plot_hm(ax1, mat_state_on_real, "① State Contrib (Az) [ON]", show_yticks=True, show_xticks=True)
        
        # ② State Diff
        ax2 = fig.add_subplot(gs[0, 2])
        plot_hm(ax2, mat_state_diff, "② State Diff (ON-OFF)", show_yticks=False, show_xticks=True)
        
        # --- Row 1: Input ---
        # ③ Input ON
        ax3 = fig.add_subplot(gs[1, 1])
        plot_hm(ax3, mat_input_on_real, "③ Input Contrib (Bu) [ON]", show_yticks=True, show_xticks=True)

        # ④ Input Diff
        ax4 = fig.add_subplot(gs[1, 2])
        plot_hm(ax4, mat_input_diff, "④ Input Diff (ON-OFF)", show_yticks=False, show_xticks=True)
        
        # --- Row 2: Total ---
        # ⑤ Total ON
        ax5 = fig.add_subplot(gs[2, 1])
        plot_hm(ax5, mat_total_on_real, "⑤ Total Contrib (Az+Bu) [ON]", show_yticks=True, show_xticks=True)

        # ⑥ Total Diff
        ax6 = fig.add_subplot(gs[2, 2])
        plot_hm(ax6, mat_total_diff, "⑥ Total Diff (ON-OFF)", show_yticks=False, show_xticks=True)
        
        # --- Row 3: Probability Bar Plots ---
        # Result Probability (ON)
        ax7 = fig.add_subplot(gs[3, 1])
        bars1 = ax7.bar(range(n_targets), target_probs_on, color='skyblue', edgecolor='black', linewidth=1.5)
        ax7.set_title("Next Token Probability (Plaza ON)", fontsize=16, weight='bold')
        ax7.set_ylabel("Probability", fontsize=14)
        ax7.set_xlabel("Target Tokens", fontsize=14)
        ax7.set_xticks(range(n_targets))
        ax7.set_xticklabels(valid_labels, rotation=0, fontsize=13)
        ax7.grid(axis='y', linestyle='--', alpha=0.4)
        for bar in bars1:
            h = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2, h, f"{h:.3f}", 
                    ha='center', va='bottom', fontsize=12, weight='bold')

        # Result Probability Diff (ON-OFF)
        ax8 = fig.add_subplot(gs[3, 2])
        colors = ['salmon' if x >= 0 else 'lightblue' for x in target_probs_diff]
        bars2 = ax8.bar(range(n_targets), target_probs_diff, color=colors, 
                       edgecolor='black', linewidth=1.5)
        ax8.axhline(0, color='gray', linewidth=1.2)
        ax8.set_title("Probability Difference (ON - OFF)", fontsize=16, weight='bold')
        ax8.set_ylabel("Diff", fontsize=14)
        ax8.set_xlabel("Target Tokens", fontsize=14)
        ax8.set_xticks(range(n_targets))
        ax8.set_xticklabels(valid_labels, rotation=0, fontsize=13)
        ax8.grid(axis='y', linestyle='--', alpha=0.4)
        for bar in bars2:
            h = bar.get_height()
            va = 'bottom' if h >= 0 else 'top'
            ax8.text(bar.get_x() + bar.get_width()/2, h, f"{h:.3f}", 
                    ha='center', va=va, fontsize=12, weight='bold')
        
        save_path = os.path.join(out_dir, f"decomposition_{sc['name']}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved decomposition plot to {save_path}")

def main():
    model, network, expanded_adj, config = load_model_and_network(MODEL_PATH, ADJ_PATH)
    analyzer = EigenAnalyzer(model)
    analyze_scenario_decomposition(model, analyzer, network, expanded_adj, SCENARIOS, OUT_DIR)
    print(f"All analyses completed. Check output at: {OUT_DIR}")

if __name__ == "__main__":
    main()