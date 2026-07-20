### 個別にPCAをかけて可視化するバージョン ###

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from network import Network
from tokenization import Tokenization
from datetime import datetime

from KP_RF import KoopmanRoutesFormer

# =========================================================
#  設定：比較する2つのモデルパス
# =========================================================
# ★ここにそれぞれのパスを入れてください
MODEL_PATH_WITH = '/home/mizutani/projects/RF/runs/20251218_033604/model_weights_20251218_033604.pth'
MODEL_PATH_WITHOUT = '/home/mizutani/projects/RF/runs/20251218_034727/model_weights_20251218_034727.pth'

trip_arrz = np.load('/home/mizutani/projects/RF/data/input_c.npz')

ADJ_PATH = '/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"/home/mizutani/projects/RF/runs/comparison_PCA"
os.makedirs(out_dir, exist_ok=True)

def stamp(name):
    return os.path.join(out_dir, name)


# =========================================================
#  共通関数
# =========================================================
def load_model_for_eval(model_path, network):
    print(f"Loading: {os.path.basename(model_path)}")
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    model = KoopmanRoutesFormer(
        vocab_size=config['vocab_size'],
        token_emb_dim=config['token_emb_dim'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        z_dim=config['z_dim'],
        pad_token_id=config['pad_token_id']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def get_z_sequence(model, tokens):
    """ルートを入力して z の系列と、Azによる予測誤差を計算"""
    with torch.no_grad():
        # z_hat: [1, T, z_dim]
        # z_pred: [1, T, z_dim] (Aかけて予測したもの)
        _, z_hat, z_pred = model(tokens.unsqueeze(0))
        
        z_seq = z_hat[0].cpu().numpy()
        
        # 線形性エラーの計算 (参考値)
        # z_hatの未来(1~) と z_pred(0~last-1から予測) を比較
        # KP_RFの実装によっては z_pred は既にシフトされている可能性があるが、
        # ここでは推論時の挙動として単純にmseをとる
        
        # shape合わせ: z_predは通常1ステップ未来を予測している
        # z_hat[:, 1:] (t=1...T) vs z_pred[:, :-1] (t=0...T-1からの予測)
        # ※モデルの実装詳細によるため、大まかな目安として
        valid_len = min(z_hat.size(1)-1, z_pred.size(1))
        mse = torch.nn.functional.mse_loss(z_pred[:, :valid_len, :], z_hat[:, 1:1+valid_len, :])
        
        return z_seq, mse.item()

def main():
    # 1. 準備
    adj_matrix = torch.load(ADJ_PATH, weights_only=True)
    network = Network(adj_matrix, torch.zeros((len(adj_matrix), 1)))
    tokenizer = Tokenization(network)

    # 2. 2つのモデルをロード
    model_with = load_model_for_eval(MODEL_PATH_WITH, network)
    model_without = load_model_for_eval(MODEL_PATH_WITHOUT, network)

    # 3. 評価用ルートの検索 (共通のルートを使う)
    all_routes = torch.from_numpy(trip_arrz['route_arr'])
    
    np.random.seed(44)
    shuffled_indices = np.random.permutation(len(all_routes))
    
    target_indices = []
    ID_PAD, ID_EOS, ID_START = 19, 20, 21

    for i in shuffled_indices:
        tokens = tokenizer.tokenization(all_routes[i].unsqueeze(0), mode="simple").long()[0]
        # End/Pad判定
        end_mask = (tokens == ID_PAD) | (tokens == ID_EOS)
        length = end_mask.nonzero(as_tuple=True)[0][0].item() if end_mask.any() else len(tokens)
        
        if 8 <= length <= 40:
            target_indices.append(i)
        if len(target_indices) >= 7: break
            
    print(f"Target Routes: {target_indices}")

    # 4. データ収集
    data_with = []
    data_without = []
    errors_with = []
    errors_without = []
    
    for idx in target_indices:
        tokens = tokenizer.tokenization(all_routes[idx].unsqueeze(0), mode="simple").long()[0].to(device)
        
        # トリミング
        if tokens[0] == ID_START: tokens = tokens[1:]
        end_mask = (tokens == ID_PAD) | (tokens == ID_EOS)
        if end_mask.any():
            valid_len = end_mask.nonzero(as_tuple=True)[0][0].item()
            if tokens[valid_len] == ID_EOS: valid_len += 1
        else:
            valid_len = len(tokens)
        
        tokens = tokens[:valid_len]
        
        # 推論
        z_w, err_w = get_z_sequence(model_with, tokens)
        z_wo, err_wo = get_z_sequence(model_without, tokens)
        
        z_w = z_w[:len(tokens)]
        z_wo = z_wo[:len(tokens)]
        
        data_with.append(z_w)
        data_without.append(z_wo)
        errors_with.append(err_w)
        errors_without.append(err_wo)

    # 5. 線形性エラーの比較表示
    print("\n=== Linearity Error Comparison (Lower is better) ===")
    print(f"With Koopman (Avg MSE):    {np.mean(errors_with):.6f}")
    print(f"Without Koopman (Avg MSE): {np.mean(errors_without):.6f}")
    print("====================================================")

    # 6. PCA個別可視化 (ここを修正)
    
    # --- With Koopman 用のPCA ---
    z_with_concat = np.concatenate(data_with, axis=0)
    pca_w = PCA(n_components=2)
    pca_w.fit(z_with_concat)

    # --- Without Koopman 用のPCA ---
    z_without_concat = np.concatenate(data_without, axis=0)
    pca_wo = PCA(n_components=2)
    pca_wo.fit(z_without_concat)
    
    # 描画
    # sharex, shareyはFalseにする（スケールや分布が異なるため）
    fig, axes = plt.subplots(1, 2, figsize=(16, 6)) 
    
    cmap = plt.get_cmap('tab10')
    
    # Plot With Koopman
    for i, z in enumerate(data_with):
        z_2d = pca_w.transform(z)
        axes[0].plot(z_2d[:,0], z_2d[:,1], marker='.', color=cmap(i), label=f"Route {target_indices[i]}")
        axes[0].text(z_2d[0,0], z_2d[0,1], "S", color=cmap(i), fontweight='bold')
    axes[0].set_title("With Koopman (Individual PCA)")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].grid(True)
    axes[0].legend()
    
    # Plot Without Koopman
    for i, z in enumerate(data_without):
        z_2d = pca_wo.transform(z)
        axes[1].plot(z_2d[:,0], z_2d[:,1], marker='.', color=cmap(i))
        axes[1].text(z_2d[0,0], z_2d[0,1], "S", color=cmap(i), fontweight='bold')
    axes[1].set_title("Without Koopman (Individual PCA)")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].grid(True)

    plt.suptitle("Latent Trajectories Comparison (Separate PCA Spaces)")
    save_path = os.path.join(out_dir, f"separate_pca_{run_id}.png")
    plt.savefig(save_path)
    print(f"Saved comparison plot: {save_path}")

if __name__ == "__main__":
    main()