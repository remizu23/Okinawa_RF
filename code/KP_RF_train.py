import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm
import wandb
import os
from datetime import datetime
from KP_RF import KoopmanRoutesFormer
import matplotlib.pyplot as plt # 追加 

# 自作モジュール
from network import Network
from tokenization import Tokenization

# --- 設定周り ---
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
# WandBの設定
class Dummy: pass
wandb = Dummy()
wandb.config = type("C", (), {
    "learning_rate": 1e-4, 
    "epochs": 200, 
    "batch_size": 256,
    "d_ie": 24,       # d_ie: 24 (4で割り切れる数に変更済)
    "head_num": 4, 
    "d_ff": 32, 
    "B_de": 6,
    "z_dim": 16,
    "eos_weight": 3.0, 
    "stay_weight": 1,
    "savefilename": "model_weights.pth"
})()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- データの準備 (Featuresなし版) ---
# 隣接行列のみロード
adj_matrix = torch.load('/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt', weights_only=True)

# ★修正点1: 特徴量ファイルがないので、Networkクラスを黙らせるためのダミーを作成
# (Networkクラスの初期化でエラーが出ないようにするためだけに使います)
dummy_feature_dim = 1
dummy_node_features = torch.zeros((len(adj_matrix), dummy_feature_dim))

# Networkクラスの初期化
network = Network(adj_matrix, dummy_node_features)

# ルートデータのロード
trip_arrz = np.load('/home/mizutani/projects/RF/data/input_a.npz')
trip_arr = trip_arrz['route_arr']
time_arr = trip_arrz['time_arr']

route = torch.from_numpy(trip_arr)
time_pt = torch.from_numpy(time_arr)
vocab_size = network.N + 4

# --- データセットとローダー (既存と同じ) ---
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2
    def __len__(self):
        return len(self.data1)
    def __getitem__(self, idx):
        return self.data1[idx], self.data2[idx]

dataset = MyDataset(route, time_pt)

# ★修正1: データを分割する (80% Train, 20% Val)
num_samples = len(dataset)
train_size = int(num_samples * 0.8)
val_size = num_samples - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

# ★修正2: ローダーを2つ作る
train_loader = DataLoader(train_data, batch_size=wandb.config.batch_size, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_data,   batch_size=wandb.config.batch_size, shuffle=False, drop_last=True)

# --- モデル構築 (B案：埋め込み利用) ---
# d_modelの計算 (特徴量を結合しないので、d_ie をそのまま d_model として使うか、調整する)
# ここでは d_ie (token_emb_dim) をそのままモデル次元として扱います
# --- モデル構築 ---
d_model = wandb.config.d_ie 

model = KoopmanRoutesFormer(
    vocab_size=vocab_size,
    token_emb_dim=wandb.config.d_ie, 
    d_model=d_model,                 
    nhead=wandb.config.head_num,     
    num_layers=wandb.config.B_de,
    d_ff=wandb.config.d_ff,
    z_dim=wandb.config.z_dim,        # ★ここも config.z_dim を参照させる
    pad_token_id=network.N
)

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
criterion = nn.MSELoss() # ダイナミクス用
# CE Lossの定義は省略（既存のものを使ってください）
ce_loss_fn = nn.CrossEntropyLoss(ignore_index=network.N)

# --- 学習ループ ---
print("Training Start...")
history = {"train_loss": [], "val_loss": []}

for epoch in range(wandb.config.epochs):
    model.train()
    epoch_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
    
    for batch, _ in pbar:
        route_batch = batch.to(device)
        
        tokenizer = Tokenization(network)
        
        # 入力データ作成
        input_tokens = tokenizer.tokenization(route_batch, mode="simple").long().to(device)
        
        # ★修正点3: featuresを作らない、渡さない
        # target (次のトークン)
        target_tokens = tokenizer.tokenization(route_batch, mode="next").long().to(device)
        
        # モデル実行 (引数はトークンのみ！)
        logits, z_hat, z_pred_next = model(input_tokens)
        
        # --- Loss計算 ---
        # 1. Cross Entropy
        # logits: [B, T, V] -> [B*T, V]
        # target: [B, T]    -> [B*T]
        loss_ce = ce_loss_fn(logits.view(-1, vocab_size), target_tokens.view(-1))
        
        # 2. Dynamics (MSE)
        # z_hatの1ステップ先(正解) と 予測値 z_pred_next の誤差
        z_true_next = z_hat[:, 1:, :] 
        loss_dyn = criterion(z_pred_next, z_true_next)
        
        loss = loss_ce + 0.1 * loss_dyn
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
        
    # ★追加: 1エポック終わるごとに平均Lossを記録する
    avg_loss = epoch_loss / len(train_loader)
    history["train_loss"].append(avg_loss)
    
    # ==========================
    #  Validation Phase
    # ==========================
    model.eval() # 評価モード (Dropoutなどが無効化される)
    val_epoch_loss = 0
    
    # 勾配計算をしない (メモリ節約・高速化)
    with torch.no_grad():
        for batch, _ in val_loader:
            route_batch = batch.to(device)
            tokenizer = Tokenization(network)
            
            input_tokens = tokenizer.tokenization(route_batch, mode="simple").long().to(device)
            target_tokens = tokenizer.tokenization(route_batch, mode="next").long().to(device)
            
            logits, z_hat, z_pred_next = model(input_tokens)
            
            loss_ce = ce_loss_fn(logits.view(-1, vocab_size), target_tokens.view(-1))
            z_true_next = z_hat[:, 1:, :] 
            loss_dyn = criterion(z_pred_next, z_true_next)
            
            loss = loss_ce + 0.1 * loss_dyn
            val_epoch_loss += loss.item()

    avg_val_loss = val_epoch_loss / len(val_loader)
    history["val_loss"].append(avg_val_loss)

    print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

# 保存先ディレクトリの作成（run_idごとのフォルダなど）
# 既存コードの out_dir = f"/home/.../runs/{run_id}" を活かす場合
out_dir = f"/home/mizutani/projects/RF/runs/{run_id}"
os.makedirs(out_dir, exist_ok=True)

def stamp(name):
    return os.path.join(out_dir, name)

# --- 保存処理 ---
savefilename = stamp(wandb.config.savefilename.replace(".pth", f"_{run_id}.pth"))

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
        "pad_token_id": network.N
    },
    "history": history,
    "train_indices": train_data.indices, # 分割したのでインデックスを保存可能
    "val_indices": val_data.indices
}

torch.save(save_data, savefilename)
print(f"Model weights saved successfully at: {savefilename}")

# --- グラフ描画 ---
try:
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, len(history["train_loss"]) + 1)
    
    plt.plot(epochs_range, history["train_loss"], label='Training Loss', marker='.')
    plt.plot(epochs_range, history["val_loss"], label='Validation Loss', marker='.') # Valも描画

    plt.title(f'Loss History (Run ID: {run_id})')
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