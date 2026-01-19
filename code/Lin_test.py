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

# =========================================================
#  設定エリア
# =========================================================
# データパス
DATA_PATH = '/home/mizutani/projects/RF/data/input_real3.npz'
ADJ_PATH = '/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt'

SPECIFIC_MODEL_PATH = '/home/mizutani/projects/RF/runs/20260118_170512/model_weights_20260118_170512.pth' # 指定する場合

# 出力先
OUT_DIR = '/home/mizutani/projects/RF/runs/Lin_test_2'
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
    expanded_adj = expand_adjacency_matrix(adj_matrix)
    dummy_feat = torch.zeros((len(adj_matrix), 1))
    expanded_feat = torch.cat([dummy_feat, dummy_feat], dim=0)
    network = Network(expanded_adj, expanded_feat)

    # 2. テストデータ読み込み
    print(f"Loading data from {DATA_PATH}...")
    trip_arrz = np.load(DATA_PATH)
    route_arr = trip_arrz['route_arr']
    
    # Agent ID処理
    if 'agent_ids' in trip_arrz:
        agent_ids = trip_arrz['agent_ids']
    else:
        agent_ids = np.zeros(len(route_arr), dtype=int)

    # 3. モデル読み込み
    model_path = SPECIFIC_MODEL_PATH
    if not os.path.exists(model_path):
        # 指定がなければ最新を探す
        list_of_files = glob.glob(os.path.join(MODEL_DIR, '*', '*.pth'))
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

    model = KoopmanRoutesFormer(
        vocab_size=config['vocab_size'],
        token_emb_dim=config['token_emb_dim'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        z_dim=config['z_dim'],
        pad_token_id=config['pad_token_id'],
        num_agents=config.get('num_agents', 1),
        agent_emb_dim=config.get('agent_emb_dim', 16),
        max_stay_count=max_stay,
        stay_emb_dim=config.get('stay_emb_dim', 16)
    )
    
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    
    return model, network, route_arr, agent_ids

# =========================================================
#  検証コアロジック
# =========================================================

def calculate_dynamics_metrics(model, network, routes, agents):
    """
    Transformer出力(z_hat)と、線形モデル出力(z_lin)を比較する
    """
    print("\n=== Starting Dynamics Verification ===")
    
    tokenizer = Tokenization(network)
    
    # バッチサイズ (GPUメモリに合わせて調整)
    BATCH_SIZE = 32
    num_samples = len(routes)
    num_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE

    # 評価用リスト
    metrics = {
        "mse_one_step": [], "mse_recursive": [],
        "r2_one_step": [], "r2_recursive": [],
        "error_ratio_one": [], "error_ratio_rec": [] # 相対誤差 (%)
    }
    
    # 行列A, Bを取得
    A = model.A.detach() # [z_dim, z_dim]
    B = model.B.detach() # [z_dim, u_dim]
    
    # 可視化用にいくつかサンプルを保存
    vis_samples = []

    with torch.no_grad():
        for i in range(num_batches):
            # バッチ作成
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, num_samples)
            
            batch_routes_np = routes[start_idx:end_idx]
            batch_agents_np = agents[start_idx:end_idx]
            
            batch_routes = torch.from_numpy(batch_routes_np).long().to(DEVICE)
            batch_agents = torch.from_numpy(batch_agents_np).long().to(DEVICE)
            
            # トークン化
            input_tokens = tokenizer.tokenization(batch_routes, mode="simple").long().to(DEVICE)
            stay_counts = tokenizer.calculate_stay_counts(input_tokens)
            
            # クリップ処理 (念のため)
            stay_counts = torch.clamp(stay_counts, max=model.stay_embedding.num_embeddings - 1)
            
            # --- 1. Transformerによる推論 (Ground Truth扱い) ---
            # u_all も取得する
            _, z_hat, _, u_all = model(input_tokens, stay_counts, batch_agents)
            # z_hat: [B, Seq, z_dim]
            # u_all: [B, Seq, u_dim]
            
            batch_size, seq_len, z_dim = z_hat.shape
            
            # --- 2. 線形モデルによる計算 ---
            
            # (A) One-step Prediction (検証A-1)
            # z[t+1] = A * z_hat[t] + B * u[t]
            # 入力は t=0 ~ t=T-2, 出力は t=1 ~ t=T-1
            
            curr_z = z_hat[:, :-1, :] # t
            curr_u = u_all[:, :-1, :] # t
            true_next_z = z_hat[:, 1:, :] # t+1 (比較対象)
            
            # 行列演算: (A @ z.T).T + (B @ u.T).T
            # Einsumで記述: b=batch, t=time, i=out_dim, j=in_dim
            term_Az = torch.einsum('ij,btj->bti', A, curr_z)
            term_Bu = torch.einsum('ij,btj->bti', B, curr_u)
            z_pred_one = term_Az + term_Bu
            
            # (B) Recursive Prediction (検証A-2: 最強の証明)
            # z[0] は z_hat[0] を使い、以降は自分の予測を使う
            # z_rec[t+1] = A * z_rec[t] + B * u[t]
            
            z_rec_list = []
            # 初期状態 (t=0)
            z_t = z_hat[:, 0, :] # [B, z_dim]
            
            # ループで再帰計算
            for t in range(seq_len - 1):
                u_t = u_all[:, t, :] # [B, u_dim]
                
                # 線形遷移
                # z_{t+1} = A z_t + B u_t
                z_next = (A @ z_t.T).T + (B @ u_t.T).T
                z_rec_list.append(z_next)
                
                # 次のステップへ更新
                z_t = z_next
                
            z_pred_rec = torch.stack(z_rec_list, dim=1) # [B, Seq-1, z_dim]
            
            # --- 3. メトリクス計算 ---
            # マスク処理 (パディング部分は計算しない)
            # input_tokens[:, 1:] がパディングでない部分
            mask = (input_tokens[:, 1:] != network.N) # [B, Seq-1]
            
            # マスク適用後のフラット化
            valid_true = true_next_z[mask]
            valid_one  = z_pred_one[mask]
            valid_rec  = z_pred_rec[mask]
            
            if len(valid_true) == 0: continue
            
            # MSE
            mse_one = torch.mean((valid_true - valid_one)**2).item()
            mse_rec = torch.mean((valid_true - valid_rec)**2).item()
            
            # 相対誤差 (Error Ratio) = |Diff| / |True|
            # ノルムの平均で割るなど
            norm_true = torch.mean(torch.abs(valid_true)) + 1e-9
            err_one = torch.mean(torch.abs(valid_true - valid_one)) / norm_true
            err_rec = torch.mean(torch.abs(valid_true - valid_rec)) / norm_true
            
            # R2 Score (CPUへ)
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
            
            # 可視化用にデータを保存 (最初のバッチの最初のサンプルだけ)
            if i == 0:
                # 最初の有効なサンプルを探す
                for idx in range(batch_size):
                    # 長さが十分あるものを
                    length = mask[idx].sum().item()
                    if length > 20:
                        vis_samples.append({
                            "z_true": true_next_z[idx, :length].cpu().numpy(),
                            "z_one": z_pred_one[idx, :length].cpu().numpy(),
                            "z_rec": z_pred_rec[idx, :length].cpu().numpy()
                        })
                        if len(vis_samples) >= 3: break # 3つ保存したら終了

    return metrics, vis_samples

# =========================================================
#  可視化関数
# =========================================================

def plot_verification(vis_samples):
    """
    z_hat(Transformer) vs z_lin(Equation) の波形比較プロット
    """
    for i, sample in enumerate(vis_samples):
        z_true = sample['z_true']
        z_one  = sample['z_one']
        z_rec  = sample['z_rec']
        
        seq_len, z_dim = z_true.shape
        
        # 全次元は多いので、分散が大きい（動きがある）上位3次元をプロット
        vars = np.var(z_true, axis=0)
        top_dims = np.argsort(vars)[::-1][:3]
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f"Linear Dynamics Verification (Sample {i})\nBlue: Transformer Output ($z_{{hat}}$), Orange: Linear Equation ($z_{{pred}}$)", fontsize=14)
        
        for j, dim in enumerate(top_dims):
            ax = axes[j]
            t = np.arange(seq_len)
            
            # Truth
            ax.plot(t, z_true[:, dim], label=f"Transformer (True)", color='blue', linewidth=2, alpha=0.6)
            
            # One-step
            ax.plot(t, z_one[:, dim], label=f"One-step Pred", color='green', linestyle='--', linewidth=1.5)
            
            # Recursive
            ax.plot(t, z_rec[:, dim], label=f"Recursive Pred", color='red', linestyle=':', linewidth=2)
            
            ax.set_ylabel(f"Dim {dim}")
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.legend(loc="upper right")
                
        axes[-1].set_xlabel("Time Step")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"verify_plot_sample_{i}.png"), dpi=300)
        plt.close()
        print(f"Saved plot: verify_plot_sample_{i}.png")
        
        # 散布図 (True vs Recursive)
        plt.figure(figsize=(6, 6))
        plt.scatter(z_true.flatten(), z_rec.flatten(), alpha=0.1, s=1, color='purple')
        min_val = min(z_true.min(), z_rec.min())
        max_val = max(z_true.max(), z_rec.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
        plt.title(f"Correlation: Transformer vs Linear Recursive (Sample {i})")
        plt.xlabel("Transformer Output")
        plt.ylabel("Linear Model Output")
        plt.grid(True)
        plt.savefig(os.path.join(OUT_DIR, f"verify_scatter_sample_{i}.png"), dpi=300)
        plt.close()

# =========================================================
#  Main
# =========================================================

def main():
    model, network, routes, agents = load_data_and_model()
    
    metrics, vis_samples = calculate_dynamics_metrics(model, network, routes, agents)
    
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
        print("先輩への反論: 「再帰的に線形予測した値だけでも、R2スコア〇〇以上の精度でTransformerの出力を再現できています。」")
    else:
        print("\n⚠️ WARNING: Recursive prediction accuracy is low.")
        print("Linear Lossが下がっていても、大域的には非線形性が残っている、または誤差が蓄積しています。")
    
    plot_verification(vis_samples)
    print(f"\nResults saved to {OUT_DIR}")

if __name__ == "__main__":
    main()