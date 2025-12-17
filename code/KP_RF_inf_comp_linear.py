### Aでの線型変換とのzの遷移の比較コード ###

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from network import Network
from tokenization import Tokenization
from KP_RF import KoopmanRoutesFormer

# =========================================================
#  設定
# =========================================================
MODEL_PATH_WITH = '/home/mizutani/projects/RF/runs/20251217_203408/model_weights_20251217_203408.pth'
# アブレーション版のパスを指定してください
MODEL_PATH_WITHOUT = '/home/mizutani/projects/RF/runs/20251217_202852/model_weights_20251217_202852.pth'

ADJ_PATH = '/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt'
out_dir = "/home/mizutani/projects/RF/runs/comparison_linear"
os.makedirs(out_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path, network):
    print(f"Loading: {os.path.basename(path)}")
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint['config']
    model = KoopmanRoutesFormer(
        vocab_size=config['vocab_size'], token_emb_dim=config['token_emb_dim'],
        d_model=config['d_model'], nhead=config['nhead'], num_layers=config['num_layers'],
        d_ff=config['d_ff'], z_dim=config['z_dim'], pad_token_id=config['pad_token_id']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def predict_trajectory_linear(model, tokens):
    """
    1. Transformerで正解の z 系列 (z_true) を取得
    2. 初期の z0 だけを使って、あとは行列 A だけで予測した系列 (z_pred_linear) を計算
       z_pred[t] = A * z_pred[t-1]
    """
    with torch.no_grad():
        # Transformerによる推論 (Ground Truth latent)
        _, z_hat, _ = model(tokens.unsqueeze(0))
        z_true = z_hat[0].cpu().numpy() # [T, z_dim]
        
        # 行列 A の取得
        A = model.A.cpu().numpy() # [z_dim, z_dim]
        
        # 線形予測 (Linear Prediction)
        z_linear = []
        # 初期値は Transformer の z0 を使う
        curr_z = z_true[0]
        z_linear.append(curr_z)
        
        for _ in range(len(z_true) - 1):
            # 次の状態を行列演算だけで予測: z_{t+1} = A @ z_t
            next_z = A @ curr_z
            z_linear.append(next_z)
            curr_z = next_z
            
        z_linear = np.array(z_linear)
        
    return z_true, z_linear

def main():
    # 準備
    adj_matrix = torch.load(ADJ_PATH, weights_only=True)
    network = Network(adj_matrix, torch.zeros((len(adj_matrix), 1)))
    tokenizer = Tokenization(network)
    
    model_with = load_model(MODEL_PATH_WITH, network)
    model_without = load_model(MODEL_PATH_WITHOUT, network)

    # 評価用ルート検索 (ランダム)
    trip_arrz = np.load('/home/mizutani/projects/RF/data/input_a.npz')
    all_routes = torch.from_numpy(trip_arrz['route_arr'])
    
    np.random.seed(999) # シード変更
    shuffled_indices = np.random.permutation(len(all_routes))
    
    target_idx = None
    ID_PAD, ID_EOS, ID_START = 19, 20, 21
    
    # ちょうど良い長さのルートを1つ探す
    for i in shuffled_indices:
        tokens = tokenizer.tokenization(all_routes[i].unsqueeze(0), mode="simple").long()[0]
        end_mask = (tokens == ID_PAD) | (tokens == ID_EOS)
        length = end_mask.nonzero(as_tuple=True)[0][0].item() if end_mask.any() else len(tokens)
        
        if 15 <= length <= 30: # 差が見えやすいように少し長めを推奨
            target_idx = i
            print(f"Target Route ID: {target_idx} (Len: {length})")
            break
            
    if target_idx is None:
        print("No suitable route found.")
        return

    # データ準備
    tokens = tokenizer.tokenization(all_routes[target_idx].unsqueeze(0), mode="simple").long()[0].to(device)
    
    # Start削除 & Endトリミング
    if tokens[0] == ID_START: tokens = tokens[1:]
    end_mask = (tokens == ID_PAD) | (tokens == ID_EOS)
    if end_mask.any():
        valid_len = end_mask.nonzero(as_tuple=True)[0][0].item()
        if tokens[valid_len] == ID_EOS: valid_len += 1
    else:
        valid_len = len(tokens)
    tokens = tokens[:valid_len]
    
    print(f"Tracing Route: {tokens.cpu().numpy()}")

    # 線形予測を実行
    z_true_w, z_lin_w = predict_trajectory_linear(model_with, tokens)
    z_true_wo, z_lin_wo = predict_trajectory_linear(model_without, tokens)

    # PCA (空間統一)
    all_z = np.concatenate([z_true_w, z_lin_w, z_true_wo, z_lin_wo], axis=0)
    pca = PCA(n_components=2)
    pca.fit(all_z)
    
    z_true_w_2d = pca.transform(z_true_w)
    z_lin_w_2d  = pca.transform(z_lin_w)
    z_true_wo_2d = pca.transform(z_true_wo)
    z_lin_wo_2d  = pca.transform(z_lin_wo)

    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. With Koopman
    axes[0].plot(z_true_w_2d[:,0], z_true_w_2d[:,1], 'o-', label='Ground Truth (Transformer)', color='blue', alpha=0.6)
    axes[0].plot(z_lin_w_2d[:,0], z_lin_w_2d[:,1], 'x--', label='Linear Prediction (A^t z0)', color='red', linewidth=2)
    axes[0].set_title("With Koopman (Proposed)")
    axes[0].legend()
    axes[0].grid(True)
    
    # Start/End注釈
    axes[0].text(z_true_w_2d[0,0], z_true_w_2d[0,1], "Start", fontsize=12, fontweight='bold')
    
    # 2. Without Koopman
    axes[1].plot(z_true_wo_2d[:,0], z_true_wo_2d[:,1], 'o-', label='Ground Truth (Transformer)', color='blue', alpha=0.6)
    axes[1].plot(z_lin_wo_2d[:,0], z_lin_wo_2d[:,1], 'x--', label='Linear Prediction (A^t z0)', color='red', linewidth=2)
    axes[1].set_title("Without Koopman (Ablation)")
    axes[1].legend()
    axes[1].grid(True)
    
    save_path = os.path.join(out_dir, f"linear_predictability_{target_idx}.png")
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")

if __name__ == "__main__":
    main()