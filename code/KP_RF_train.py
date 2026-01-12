import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm
import wandb
import os
from datetime import datetime
from KP_RF import KoopmanRoutesFormer
import matplotlib.pyplot as plt

# 自作モジュール
from network import Network, expand_adjacency_matrix
from tokenization import Tokenization

# --- 設定周り ---
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
# WandBの設定（ダミー）
class Dummy: pass
wandb = Dummy()
wandb.config = type("C", (), {
    "learning_rate": 1e-4, 
    "epochs": 1000, # EarlyStoppingが入るので大きめに設定
    "batch_size": 128, # 系列が長くなるので、メモリ溢れするならここを減らす
    "d_ie": 64,
    "head_num": 4, 
    "d_ff": 32, 
    "B_de": 6,
    "z_dim": 16,
    "eos_weight": 3.0, 
    "stay_weight": 1,
    "savefilename": "model_weights.pth",

    # 0105埋め込み変更後
    "agent_emb_dim": 16,
    "stay_emb_dim": 16,
    "max_stay_count": 500, # 必要に応じて調整
    
    # ★★★ Ablation Study用設定 ★★★
    "use_koopman_loss": True,  # True: 提案手法(Koopmanあり), False: 比較手法(なし) ←ここを切り替えて2回実験！
    "koopman_alpha": 0.1       # Koopman Lossの重み
})()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 保存ディレクトリ作成 ---
out_dir = f"/home/mizutani/projects/RF/runs/{run_id}"
os.makedirs(out_dir, exist_ok=True)

def stamp(name):
    return os.path.join(out_dir, name)

# --- データの準備 ---
trip_arrz = np.load('/home/mizutani/projects/RF/data/input_real.npz') ##インプットを変えたら変える！

adj_matrix = torch.load('/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt', weights_only=True)

# 滞在トークン用に隣接行列を拡張
expanded_adj = expand_adjacency_matrix(adj_matrix)

dummy_feature_dim = 1
# 元の行列サイズでzerosを作る
dummy_node_features = torch.zeros((len(adj_matrix), dummy_feature_dim))
# 縦に結合して倍にする
expanded_features = torch.cat([dummy_node_features, dummy_node_features], dim=0)

# ★ 拡張された行列と特徴量でNetworkインスタンス化
network = Network(expanded_adj, expanded_features)

trip_arr = trip_arrz['route_arr']
time_arr = trip_arrz['time_arr']

# ★ユーザーIDの読み込み (npzに含まれていると仮定)
if 'agent_ids' in trip_arrz:
    agent_ids_arr = trip_arrz['agent_ids']
else:
    # なければ仮のID (全員0) を作成、あるいはエラーにする
    print("Warning: 'agent_ids' not found in npz. Using dummy IDs.")
    agent_ids_arr = np.zeros(len(trip_arr), dtype=int)



route = torch.from_numpy(trip_arr)
time_pt = torch.from_numpy(time_arr)
agent_pt = torch.from_numpy(agent_ids_arr).long() # ★テンソル化
vocab_size = network.N + 4

# データセット定義
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data1, data2, data3):
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3
    def __len__(self):
        return len(self.data1)
    def __getitem__(self, idx):
        return self.data1[idx], self.data2[idx], self.data3[idx]

dataset = MyDataset(route, time_pt, agent_pt)

# データ分割 (8:2)
num_samples = len(dataset)
train_size = int(num_samples * 0.8)
val_size = num_samples - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

# パディング関数の定義
def collate_fn_pad(batch):
    # batchは (route, time, agent) のタプルのリスト
    routes_raw, times, agents = zip(*batch)

    # 1. まず各データの「本当の長さ（38以外の部分）」を特定して切り出す
    trimmed_routes = []
    for r in routes_raw:
        # r が Tensor か numpy かで分岐
        if isinstance(r, torch.Tensor):
            r_np = r.cpu().numpy()
        else:
            r_np = np.array(r)
            
        # 38 (パディング) が最初に現れる位置を探す
        # もし38がなければそのまま、あればそこまでで切る
        pad_indices = np.where(r_np == 38)[0]
        if len(pad_indices) > 0:
            real_len = pad_indices[0]
            # ただし、長さ0になってしまうと困るので最低1は残すなどのケアが必要かも
            # ここではシンプルにスライス
            trimmed_r = r_np[:real_len]
        else:
            trimmed_r = r_np # パディングなし（フル）

        # Tensorにしてリストに追加
        trimmed_routes.append(torch.tensor(trimmed_r, dtype=torch.long))

    # 2. このバッチ内での最大長を取得
    lengths = [len(r) for r in trimmed_routes]
    max_len = max(lengths) if lengths else 0

    # 3. 最大長に合わせてパディングし直す
    padded_routes = torch.zeros(len(trimmed_routes), max_len, dtype=torch.long) + 38 
    
    for i, r in enumerate(trimmed_routes):
        end = len(r)
        padded_routes[i, :end] = r
            
    # Time, Agentはそのまま
    times = torch.tensor(times)
    agents = torch.tensor(agents)
    
    return padded_routes, times, agents

# DataLoaderにセット
train_loader = DataLoader(
    train_data, 
    batch_size=wandb.config.batch_size, 
    shuffle=True, 
    collate_fn=collate_fn_pad, # ★これを追加
    drop_last=True
)

val_loader = DataLoader(
    val_data,   
    batch_size=wandb.config.batch_size, 
    shuffle=False, 
    collate_fn=collate_fn_pad, # ★ここにも追加！
    drop_last=True
)

# --- モデル構築 ---
d_model = wandb.config.d_ie 
num_agents = int(agent_pt.max().item()) + 1



model = KoopmanRoutesFormer(
    vocab_size=vocab_size,
    token_emb_dim=wandb.config.d_ie, 
    d_model=d_model,                 
    nhead=wandb.config.head_num,     
    num_layers=wandb.config.B_de,
    d_ff=wandb.config.d_ff,
    z_dim=wandb.config.z_dim,
    pad_token_id=network.N,
    # ★追加引数
    num_agents=num_agents,
    agent_emb_dim=wandb.config.agent_emb_dim,
    max_stay_count=wandb.config.max_stay_count,
    stay_emb_dim=wandb.config.stay_emb_dim
)
model = model.to(device)

# # 滞在トークン・移動トークンの初期埋め込みを似せる
# # トークンID 0~N-1 (移動) の重みを、N~2N-1 (滞在) にコピーする
# with torch.no_grad():
#     # Embedding層の重みを取得 [vocab_size, dim]
#     emb_weight = model.token_embedding.token_embedding.weight
    
#     # 0番目からN-1番目までの重みをコピー
#     original_weights = emb_weight[:network.N]
    
#     # N番目から2N-1番目（滞在トークン）に上書き
#     # 少しだけノイズを加えて「似ているが少し違う」状態からスタートさせる
#     emb_weight[network.N : network.N*2] = original_weights + torch.randn_like(original_weights) * 0.01

# print("滞在トークンのEmbeddingを移動トークンベースで初期化しました。")

optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

# ★重要: パディングを無視するためのMSE損失関数
def masked_mse_loss(input, target, mask):
    """
    maskがTrueの部分だけでMSEを計算する
    input, target: [Batch, Seq, Dim] or [Batch, Seq]
    mask: [Batch, Seq] (True=Valid, False=Padding)
    """
    diff = input - target
    squared_diff = diff ** 2
    
    # マスクを適用 (次元を合わせる)
    if mask.dim() < squared_diff.dim():
        mask = mask.unsqueeze(-1).expand_as(squared_diff)
        
    masked_diff = squared_diff * mask.float()
    
    # 有効な要素数で割る (0除算回避)
    num_valid = mask.sum()
    if num_valid > 0:
        return masked_diff.sum() / num_valid
    else:
        return torch.tensor(0.0, device=input.device)

# CE Lossは ignore_index でパディング(network.N)を無視
ce_loss_fn = nn.CrossEntropyLoss(ignore_index=network.N)

# =========================================================
#  ★ Early Stopping の準備
# =========================================================
patience = 15
early_stopping_counter = 0
best_val_loss = float('inf')

# ベストモデルと最終モデルの保存名
final_savefilename = stamp(wandb.config.savefilename.replace(".pth", f"_final_{run_id}.pth"))
best_savefilename  = stamp(wandb.config.savefilename.replace(".pth", f"_best_{run_id}.pth"))

print(f"Training Start... (Koopman Loss: {wandb.config.use_koopman_loss})")
history = {"train_loss": [], "val_loss": []}


# =========================================================
#  学習ループ (Training Phase)
# =========================================================
print(f"Training Start... (Koopman Loss: {wandb.config.use_koopman_loss})")
history = {"train_loss": [], "val_loss": []}

for epoch in range(wandb.config.epochs):
    # --- Training ---
    model.train()
    train_epoch_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    
    for batch in pbar:
        route_batch, _, agent_batch = batch 
        route_batch = route_batch.to(device)
        agent_batch = agent_batch.to(device)
        
        tokenizer = Tokenization(network)
        input_tokens = tokenizer.tokenization(route_batch, mode="simple").long().to(device)
        target_tokens = tokenizer.tokenization(route_batch, mode="next").long().to(device)
        
        stay_counts = tokenizer.calculate_stay_counts(input_tokens)
        
        # ★修正: 4つの戻り値を受け取る (u_allが必要)
        logits, z_hat, z_pred_next, u_all = model(input_tokens, stay_counts, agent_batch)        
        
        # 1. CE Loss (全員共通)
        loss_ce = ce_loss_fn(logits.view(-1, vocab_size), target_tokens.view(-1))
        
        # 初期化
        loss = loss_ce
        loss_count = torch.tensor(0.0, device=device)
        loss_dyn = torch.tensor(0.0, device=device)
        loss_k = torch.tensor(0.0, device=device)
        loss_mode = torch.tensor(0.0, device=device) # モード損失

        # ★ Koopmanモードの時だけ追加Lossを計算
        if wandb.config.use_koopman_loss:
            # ★マスク作成: パディング(network.N)以外の場所がTrue
            # input_tokensの形状 [Batch, Seq]
            valid_mask = (input_tokens != network.N)

            # 2. Count Reconstruction (zの意味付け)
            pred_count = model.count_decoder(z_hat).squeeze(-1)
            
            # ★カウント予測をやめる場合はここをコメントアウトし、loss_count=0のままにする
            loss_count = masked_mse_loss(pred_count, stay_counts.float(), valid_mask)

            # 3. Multi-step Dynamics Loss (再帰予測)
            K_steps = 5  # 何歩先まで予測するか
            
            # t=0 から t=T-K までの z を初期状態とする
            current_z = z_hat[:, :-K_steps, :]
            
            # 再帰ループ
            for k in range(K_steps):
                # 入力 u を取得 (t+k のタイミングのもの)
                # u_all: [Batch, Seq, Dim]
                # スライス: t=k から t=Seq-K+k まで
                end_idx = -K_steps + k if (-K_steps + k) != 0 else None
                u_curr_step = u_all[:, k : end_idx, :]
                
                # 線形遷移: z_{t+1} = A*z_t + B*u_t
                term_A = torch.einsum("ij,btj->bti", model.A, current_z)
                term_B = torch.einsum("ij,btj->bti", model.B, u_curr_step)
                pred_z_next = term_A + term_B
                
                # 正解データ (Transformerの出力) と比較
                start_true = k + 1
                end_true = -K_steps + k + 1 if (-K_steps + k + 1) != 0 else None
                true_z_next = z_hat[:, start_true : end_true, :]
                
                # ★マスクも未来に合わせてずらす
                # 未来の時点がパディングなら、そこは学習しない
                future_mask = valid_mask[:, start_true : end_true]
                
                decay = 0.8 ** k
                loss_dyn += masked_mse_loss(pred_z_next, true_z_next, future_mask) * decay
                
                current_z = pred_z_next

            # 4. Single-step Loss (念のため直近精度も担保)
            z_true_next = z_hat[:, 1:, :] 
            # z_pred_next は forward内で計算済み
            # マスクは1つずらす
            next_step_mask = valid_mask[:, 1:]
            loss_k = masked_mse_loss(z_pred_next, z_true_next, next_step_mask)
            
            # 5. モード損失
            # 1. まず、すべてを「無視 (-100)」で初期化
            target_modes = torch.full_like(target_tokens, -100)  # target_modes: [Batch, Seq]

            # 2. Moveトークン (0 <= ID < 19) の場所を "1" に設定
            is_move = (target_tokens >= 0) & (target_tokens < network.N)
            target_modes[is_move] = 1

            # 3. Stayトークン (19 <= ID < 38) の場所を "0" に設定
            # STAY_OFFSET = 19, PAD_TOKEN = 38 (network.N * 2)
            STAY_OFFSET = network.N
            PAD_TOKEN = network.N * 2
            is_stay = (target_tokens >= STAY_OFFSET) & (target_tokens < PAD_TOKEN)
            target_modes[is_stay] = 0

            # --- Loss計算 ---
            # model.mode_classifier(z_hat) -> [Batch, Seq, 2]
            pred_modes = model.mode_classifier(z_hat)

            # 形状を合わせる: [Batch*Seq, 2] vs [Batch*Seq]
            # ignore_index=-100 なので、パディング部分は自動的に無視され、
            # Move(1)とStay(0)の部分だけが学習されます。
            loss_mode = nn.CrossEntropyLoss(ignore_index=-100)(
                pred_modes.view(-1, 2), 
                target_modes.view(-1)
            )

            # 合計Loss
            # 係数はタスクに応じて調整 (Alphaは重め、Count/Dynは補助的)
            loss = loss_ce + \
                   wandb.config.koopman_alpha * loss_k + \
                   0.001 * loss_count + \
                   0.1 * loss_dyn
                #    0.1 * loss_mode

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_epoch_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    
    avg_train_loss = train_epoch_loss / len(train_loader)
    history["train_loss"].append(avg_train_loss)

# --- Validation (ロジックはTrainと全く同じ) ---
    model.eval()
    val_epoch_loss = 0
    
    with torch.no_grad():
        for route_batch, _, agent_batch in val_loader:
            route_batch = route_batch.to(device)
            agent_batch = agent_batch.to(device)
            
            tokenizer = Tokenization(network)
            input_tokens = tokenizer.tokenization(route_batch, mode="simple").long().to(device)
            target_tokens = tokenizer.tokenization(route_batch, mode="next").long().to(device)
            stay_counts = tokenizer.calculate_stay_counts(input_tokens)

            logits, z_hat, z_pred_next, u_all = model(input_tokens, stay_counts, agent_batch)

            loss_ce = ce_loss_fn(logits.view(-1, vocab_size), target_tokens.view(-1))
            loss = loss_ce 
            
            if wandb.config.use_koopman_loss:
                valid_mask = (input_tokens != network.N)
                
                pred_count = model.count_decoder(z_hat).squeeze(-1)
                loss_count = masked_mse_loss(pred_count, stay_counts.float(), valid_mask)

                loss_dyn = 0
                K_steps = 5
                current_z = z_hat[:, :-K_steps, :]
                for k in range(K_steps):
                    end_idx = -K_steps + k if (-K_steps + k) != 0 else None
                    u_curr_step = u_all[:, k : end_idx, :]
                    term_A = torch.einsum("ij,btj->bti", model.A, current_z)
                    term_B = torch.einsum("ij,btj->bti", model.B, u_curr_step)
                    pred_z_next = term_A + term_B
                    
                    start_true = k + 1
                    end_true = -K_steps + k + 1 if (-K_steps + k + 1) != 0 else None
                    true_z_next = z_hat[:, start_true : end_true, :]
                    
                    future_mask = valid_mask[:, start_true : end_true]
                    loss_dyn += masked_mse_loss(pred_z_next, true_z_next, future_mask) * (0.8 ** k)
                    current_z = pred_z_next

                z_true_next = z_hat[:, 1:, :]
                next_step_mask = valid_mask[:, 1:]
                loss_k = masked_mse_loss(z_pred_next, z_true_next, next_step_mask)
            
                target_modes = torch.full_like(target_tokens, -100)
                is_move = (target_tokens >= 0) & (target_tokens < network.N)
                target_modes[is_move] = 1
                STAY_OFFSET = network.N
                PAD_TOKEN = network.N * 2
                is_stay = (target_tokens >= STAY_OFFSET) & (target_tokens < PAD_TOKEN)
                target_modes[is_stay] = 0

                pred_modes = model.mode_classifier(z_hat)
                loss_mode = nn.CrossEntropyLoss(ignore_index=-100)(
                    pred_modes.view(-1, 2), 
                    target_modes.view(-1)
                )

                loss = loss_ce + \
                       wandb.config.koopman_alpha * loss_k + \
                       0.001 * loss_count + \
                       0.1 * loss_dyn
                    #    0.1 * loss_mode

            val_epoch_loss += loss.item()

    avg_val_loss = val_epoch_loss / len(val_loader)
    history["val_loss"].append(avg_val_loss)

    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")


# =========================================================
    # ★ Early Stopping & Best Model Saving Logic (C案)
    # =========================================================
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stopping_counter = 0
        print(f"--> Best Model Found! Val Loss: {best_val_loss:.4f}. Saving...")
        
        # ベストモデルの保存
        save_data = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "vocab_size": vocab_size,
                "token_emb_dim": wandb.config.d_ie,
                "d_model": d_model,
                "nhead": wandb.config.head_num,
                "num_layers": wandb.config.B_de,
                "d_ff": wandb.config.d_ff,
                "z_dim": wandb.config.z_dim,
                "pad_token_id": network.N,
                "num_agents": num_agents,
                "agent_emb_dim": wandb.config.agent_emb_dim,
                "max_stay_count": wandb.config.max_stay_count,
                "stay_emb_dim": wandb.config.stay_emb_dim,
                "use_koopman_loss": wandb.config.use_koopman_loss,
                "koopman_alpha": wandb.config.koopman_alpha
            },
            "history": history,
            "best_val_loss": best_val_loss,
            "epoch": epoch
        }
        torch.save(save_data, best_savefilename)
    else:
        early_stopping_counter += 1
        print(f"--> No Improvement. Counter: {early_stopping_counter}/{patience}")
        
        if early_stopping_counter >= patience:
            print("Early Stopping Triggered. Stopping Training.")
            break

# =========================================================
#  最終モデルの保存 (ループ終了後)
# =========================================================
save_data_final = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "config": {
        "vocab_size": vocab_size,
        "token_emb_dim": wandb.config.d_ie,
        "d_model": d_model,
        "nhead": wandb.config.head_num,
        "num_layers": wandb.config.B_de,
        "d_ff": wandb.config.d_ff,
        "z_dim": wandb.config.z_dim,
        "pad_token_id": network.N,
        "num_agents": num_agents,
        "agent_emb_dim": wandb.config.agent_emb_dim,
        "max_stay_count": wandb.config.max_stay_count,
        "stay_emb_dim": wandb.config.stay_emb_dim,
        "use_koopman_loss": wandb.config.use_koopman_loss,
        "koopman_alpha": wandb.config.koopman_alpha
    },
    "history": history
}
torch.save(save_data_final, final_savefilename)
print(f"Final Model saved: {final_savefilename}")
print(f"Best Model saved at: {best_savefilename}")

# --- グラフ描画 ---
try:
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, len(history["train_loss"]) + 1)
    
    plt.plot(epochs_range, history["train_loss"], label='Training Loss', marker='.')
    plt.plot(epochs_range, history["val_loss"], label='Validation Loss', marker='.')

    title_str = "With Koopman" if wandb.config.use_koopman_loss else "Without Koopman"
    plt.title(f'Loss History ({title_str}) - Run ID: {run_id}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    graph_filename = stamp(f"loss_graph_{run_id}.png")
    plt.savefig(graph_filename)
    plt.close()
    print(f"Loss graph saved at: {graph_filename}")

except Exception as e:
    print(f"Failed to plot loss graph: {e}")