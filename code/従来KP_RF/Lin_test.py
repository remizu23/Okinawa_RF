import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from network import Network, expand_adjacency_matrix
from tokenization import Tokenization
from KP_RF import KoopmanRoutesFormer
import glob
from datetime import datetime
import networkx as nx

# =========================================================
#  設定エリア
# =========================================================
# データパス
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# ★修正: 圧縮版データを指定
DATA_PATH = '/home/mizutani/projects/RF/data/input_real_m4_emb.npz'
ADJ_PATH = '/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt'

# ★注意: 必ず「新コードで学習したモデル」のパスを指定してください
SPECIFIC_MODEL_PATH = '/home/mizutani/projects/RF/runs/20260123_210513/model_weights_20260123_210513.pth' 

# 出力先
OUT_DIR = f'/home/mizutani/projects/RF/runs/20260123_210513/linear'
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
#  データ＆モデル読み込み関数
# =========================================================

def load_data_and_model():
    # 1. ネットワーク準備
    if not os.path.exists(ADJ_PATH):
        raise FileNotFoundError(f"Adjacency matrix not found: {ADJ_PATH}")
    adj_matrix = torch.load(ADJ_PATH, weights_only=True)
    
    # 距離行列計算 (モデル初期化に必要)
    def compute_shortest_path_distance_matrix(adj: torch.Tensor, directed: bool = False) -> torch.Tensor:
        if not isinstance(adj, torch.Tensor): adj = torch.tensor(adj)
        adj_cpu = adj.detach().cpu()
        N = adj_cpu.shape[0]
        G = nx.DiGraph() if directed else nx.Graph()
        G.add_nodes_from(range(N))
        rows, cols = torch.nonzero(adj_cpu, as_tuple=True)
        edges = [(int(r), int(c)) for r, c in zip(rows, cols) if int(r) != int(c)]
        G.add_edges_from(edges)
        dist = torch.full((N, N), fill_value=N + 1, dtype=torch.long)
        for s in range(N):
            dist[s, s] = 0
            for t, d in nx.single_source_shortest_path_length(G, s).items():
                dist[s, t] = int(d)
        return dist

    dist_is_directed = False
    if adj_matrix.shape[0] == 38:
        base_N = 19
        base_adj = adj_matrix[:base_N, :base_N]
    else:
        base_adj = adj_matrix
        base_N = int(base_adj.shape[0])
    
    dist_mat_base = compute_shortest_path_distance_matrix(base_adj, directed=dist_is_directed)

    expanded_adj = expand_adjacency_matrix(adj_matrix)
    dummy_feat = torch.zeros((len(adj_matrix), 1))
    expanded_feat = torch.cat([dummy_feat, dummy_feat], dim=0)
    network = Network(expanded_adj, expanded_feat)

    # 2. テストデータ読み込み
    print(f"Loading data from {DATA_PATH}...")
    trip_arrz = np.load(DATA_PATH)
    route_arr = trip_arrz['route_arr']
    
    # ★追加: コンテキスト配列の読み込み
    holiday_arr = trip_arrz['holiday_arr']
    timezone_arr = trip_arrz['time_zone_arr']
    event_arr = trip_arrz['event_arr']
    
    # Agent ID処理
    if 'agent_ids' in trip_arrz:
        agent_ids = trip_arrz['agent_ids']
    else:
        agent_ids = np.zeros(len(route_arr), dtype=int)

    # 3. モデル読み込み
    model_path = SPECIFIC_MODEL_PATH
    if not os.path.exists(model_path):
        # 指定がなければ最新を探す
        # MODEL_DIR が未定義だったので runs から探すように修正
        search_dir = '/home/mizutani/projects/RF/runs/'
        list_of_files = glob.glob(os.path.join(search_dir, '*', '*.pth'))
        if not list_of_files:
            raise FileNotFoundError("No model found.")
        model_path = max(list_of_files, key=os.path.getctime)
    
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    config = checkpoint['config']
    
    # 滞在数の上限判定
    state_dict = checkpoint['model_state_dict']
    if 'stay_embedding.weight' in state_dict:
        max_stay = state_dict['stay_embedding.weight'].shape[0] - 1
    else:
        max_stay = config.get('max_stay_count', 100)

    # 新しいコンテキスト次元設定の読み込み（なければデフォルト4）
    h_dim = config.get('holiday_emb_dim', 4)
    tz_dim = config.get('time_zone_emb_dim', 4)
    e_dim = config.get('event_emb_dim', 4)

    model = KoopmanRoutesFormer(
        vocab_size=config['vocab_size'],
        token_emb_dim=config['token_emb_dim'],
        d_model=config['d_model'] if 'd_model' in config else config['d_ie'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        z_dim=config['z_dim'],
        pad_token_id=config['pad_token_id'],
        dist_mat_base=dist_mat_base,
        base_N=base_N,
        num_agents=config.get('num_agents', 1),
        agent_emb_dim=config.get('agent_emb_dim', 16),
        max_stay_count=max_stay,
        stay_emb_dim=config.get('stay_emb_dim', 16),
        # ★追加: 新しい埋め込み次元
        holiday_emb_dim=h_dim,
        time_zone_emb_dim=tz_dim,
        event_emb_dim=e_dim
    )
    
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    
    # 戻り値に追加
    return model, network, route_arr, agent_ids, holiday_arr, timezone_arr, event_arr

# =========================================================
#  検証コアロジック
# =========================================================

def calculate_dynamics_metrics(model, network, routes, agents, holidays, timezones, events):
    """
    Transformer出力(z_hat)と、線形モデル出力(z_lin)を比較する
    """
    print("\n=== Starting Dynamics Verification ===")
    
    tokenizer = Tokenization(network)
    
    BATCH_SIZE = 32
    num_samples = len(routes)
    num_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE

    metrics = {
        "mse_one_step": [], "mse_recursive": [],
        "r2_one_step": [], "r2_recursive": [],
        "error_ratio_one": [], "error_ratio_rec": [] 
    }
    
    A = model.A.detach()
    B = model.B.detach()
    
    vis_samples = []

    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, num_samples)
            
            # バッチ切り出し
            batch_routes_np = routes[start_idx:end_idx]
            batch_agents_np = agents[start_idx:end_idx]
            batch_h_np = holidays[start_idx:end_idx]
            batch_tz_np = timezones[start_idx:end_idx]
            batch_e_np = events[start_idx:end_idx]
            
            # Tensor化
            batch_routes = torch.from_numpy(batch_routes_np).long().to(DEVICE)
            batch_agents = torch.from_numpy(batch_agents_np).long().to(DEVICE)
            batch_h = torch.from_numpy(batch_h_np).long().to(DEVICE)
            batch_tz = torch.from_numpy(batch_tz_np).long().to(DEVICE)
            batch_e = torch.from_numpy(batch_e_np).long().to(DEVICE)
            
            # トークン化 (simpleモード = 先頭に<b>追加)
            input_tokens = tokenizer.tokenization(batch_routes, mode="simple").long().to(DEVICE)
            stay_counts = tokenizer.calculate_stay_counts(input_tokens)
            stay_counts = torch.clamp(stay_counts, max=model.stay_embedding.num_embeddings - 1)
            
            # ★★★ コンテキストのアライメント (学習時と同じロジック) ★★★
            B_size, T = input_tokens.shape
            
            def align_ctx(ctx, target_len):
                # 先頭(0番目)は<b>用なので0埋め、1番目以降にctxを配置
                out = torch.zeros((B_size, target_len), dtype=torch.long, device=DEVICE)
                copy_len = min(ctx.shape[1], target_len - 1)
                if copy_len > 0:
                    out[:, 1 : 1 + copy_len] = ctx[:, :copy_len]
                return out

            h_in = align_ctx(batch_h, T)
            tz_in = align_ctx(batch_tz, T)
            e_in = align_ctx(batch_e, T)
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            
            # 1. Transformer推論
            # 引数を追加して呼び出し
            _, z_hat, _, u_all = model(
                tokens=input_tokens, 
                stay_counts=stay_counts, 
                agent_ids=batch_agents,
                holidays=h_in,
                time_zones=tz_in,
                events=e_in
            )
            
            batch_size, seq_len, z_dim = z_hat.shape
            
            # 2. 線形モデルによる計算
            
            # (A) One-step Prediction
            curr_z = z_hat[:, :-1, :] 
            curr_u = u_all[:, :-1, :] 
            true_next_z = z_hat[:, 1:, :] 
            
            term_Az = torch.einsum('ij,btj->bti', A, curr_z)
            term_Bu = torch.einsum('ij,btj->bti', B, curr_u)
            z_pred_one = term_Az + term_Bu
            
            # (B) Recursive Prediction
            z_rec_list = []
            z_t = z_hat[:, 0, :] 
            
            for t in range(seq_len - 1):
                u_t = u_all[:, t, :] 
                z_next = (A @ z_t.T).T + (B @ u_t.T).T
                z_rec_list.append(z_next)
                z_t = z_next
                
            z_pred_rec = torch.stack(z_rec_list, dim=1) 
            
            # 3. メトリクス計算
            mask = (input_tokens[:, 1:] != network.N) 
            
            valid_true = true_next_z[mask]
            valid_one  = z_pred_one[mask]
            valid_rec  = z_pred_rec[mask]
            
            if len(valid_true) == 0: continue
            
            mse_one = torch.mean((valid_true - valid_one)**2).item()
            mse_rec = torch.mean((valid_true - valid_rec)**2).item()
            
            norm_true = torch.mean(torch.abs(valid_true)) + 1e-9
            err_one = torch.mean(torch.abs(valid_true - valid_one)) / norm_true
            err_rec = torch.mean(torch.abs(valid_true - valid_rec)) / norm_true
            
            v_true_np = valid_true.cpu().numpy()
            v_one_np  = valid_one.cpu().numpy()
            v_rec_np  = valid_rec.cpu().numpy()
            
            r2_one = r2_score(v_true_np, v_one_np)
            r2_rec = r2_score(v_true_np, v_rec_np)
            
            metrics["mse_one_step"].append(mse_one)
            metrics["mse_recursive"].append(mse_rec)
            metrics["error_ratio_one"].append(err_one.item())
            metrics["error_ratio_rec"].append(err_rec.item())
            metrics["r2_one_step"].append(r2_one)
            metrics["r2_recursive"].append(r2_rec)
            
            # 可視化用に保存
            if i == 0:
                for idx in range(batch_size):
                    length = mask[idx].sum().item()
                    if length > 8:
                        vis_samples.append({
                            "z_true": true_next_z[idx, :length].cpu().numpy(),
                            "z_one": z_pred_one[idx, :length].cpu().numpy(),
                            "z_rec": z_pred_rec[idx, :length].cpu().numpy()
                        })
                        if len(vis_samples) >= 3: break 

    return metrics, vis_samples

# =========================================================
#  可視化関数
# =========================================================

def plot_verification(vis_samples):
    """
    z_hat(Transformer) vs z_lin(Equation) の波形比較プロット
    ★修正: 全次元(16次元)を 4x4 グリッドで可視化
    """
    for i, sample in enumerate(vis_samples):
        z_true = sample['z_true']
        z_one  = sample['z_one']
        z_rec  = sample['z_rec']
        
        seq_len, z_dim = z_true.shape
        
        # --- 全次元表示 (4x4) ---
        # z_dim が 16 でない場合も柔軟に対応
        cols = 4
        rows = (z_dim + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 3 * rows), sharex=True)
        fig.suptitle(f"Linear Dynamics Verification (Sample {i}) - All Dimensions\n"
                     f"Blue: Transformer ($z_{{hat}}$), Green: One-step, Red: Recursive", fontsize=16)
        
        axes_flat = axes.flatten()
        
        for dim in range(z_dim):
            ax = axes_flat[dim]
            t = np.arange(seq_len)
            
            # Truth
            ax.plot(t, z_true[:, dim], label=f"True", color='blue', linewidth=2, alpha=0.5)
            # One-step
            ax.plot(t, z_one[:, dim], label=f"1-Step", color='green', linestyle='--', linewidth=1)
            # Recursive
            ax.plot(t, z_rec[:, dim], label=f"Rec", color='red', linestyle=':', linewidth=1.5)
            
            ax.set_title(f"Dim {dim}", fontsize=10)
            ax.grid(True, alpha=0.3)
            
            if dim == 0: # 凡例は最初のコマだけ
                ax.legend(loc="upper right", fontsize=8)
        
        # 余ったサブプロットを消す
        for k in range(z_dim, len(axes_flat)):
            axes_flat[k].axis('off')
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_path = os.path.join(OUT_DIR, f"verify_plot_sample_{i}_all.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"Saved all-dim plot: {save_path}")
        
        # 散布図 (True vs Recursive)
        plt.figure(figsize=(6, 6))
        plt.scatter(z_true.flatten(), z_rec.flatten(), alpha=0.1, s=1, color='purple')
        min_val = min(z_true.min(), z_rec.min())
        max_val = max(z_true.max(), z_rec.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
        plt.title(f"Correlation: Transformer vs Recursive (Sample {i})")
        plt.xlabel("Transformer Output")
        plt.ylabel("Recursive Linear Output")
        plt.grid(True)
        plt.savefig(os.path.join(OUT_DIR, f"verify_scatter_sample_{i}.png"), dpi=200)
        plt.close()

# =========================================================
#  Main
# =========================================================

def main():
    # データ読み込みの戻り値を受け取る変数を増やす
    model, network, routes, agents, holidays, timezones, events = load_data_and_model()
    
    # 計算関数へ渡す
    metrics, vis_samples = calculate_dynamics_metrics(
        model, network, routes, agents, holidays, timezones, events
    )
    
    # 結果の集計
    print("\n=== Verification Results ===")
    
    avg_r2_one = np.mean(metrics["r2_one_step"])
    avg_r2_rec = np.mean(metrics["r2_recursive"])
    avg_err_one = np.mean(metrics["error_ratio_one"]) * 100
    avg_err_rec = np.mean(metrics["error_ratio_rec"]) * 100
    
    print(f"One-step Prediction (Local Consistency):")
    print(f"  R2 Score: {avg_r2_one:.4f} (Target: > 0.90)")
    print(f"  Rel Error: {avg_err_one:.2f}%")
    
    print(f"-"*30)
    
    print(f"Recursive Prediction (Global Dynamics):")
    print(f"  R2 Score: {avg_r2_rec:.4f} (Target: > 0.80)")
    print(f"  Rel Error: {avg_err_rec:.2f}%")
    
    if avg_r2_rec > 0.8:
        print("\n✅ SUCCESS: The Transformer has successfully learned linear dynamics.")
    else:
        print("\n⚠️ WARNING: Recursive prediction accuracy is low.")
    
    plot_verification(vis_samples)
    print(f"\nResults saved to {OUT_DIR}")

if __name__ == "__main__":
    main()