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
from network import Network
from tokenization import Tokenization

# --- 設定周り ---
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
# WandBの設定（ダミー）
class Dummy: pass
wandb = Dummy()
wandb.config = type("C", (), {
    "learning_rate": 1e-4, 
    "epochs": 300, 
    "batch_size": 256,
    "d_ie": 64,
    "head_num": 4, 
    "d_ff": 32, 
    "B_de": 6,
    "z_dim": 16,
    "eos_weight": 3.0, 
    "stay_weight": 1,
    "savefilename": "model_weights.pth",
    
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
trip_arrz = np.load('/home/mizutani/projects/RF/data/input_c.npz') ##インプットを変えたら変える！

adj_matrix = torch.load('/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt', weights_only=True)
dummy_feature_dim = 1
dummy_node_features = torch.zeros((len(adj_matrix), dummy_feature_dim))
network = Network(adj_matrix, dummy_node_features)

trip_arr = trip_arrz['route_arr']
time_arr = trip_arrz['time_arr']

route = torch.from_numpy(trip_arr)
time_pt = torch.from_numpy(time_arr)
vocab_size = network.N + 4

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2
    def __len__(self):
        return len(self.data1)
    def __getitem__(self, idx):
        return self.data1[idx], self.data2[idx]

dataset = MyDataset(route, time_pt)

# データ分割 (8:2)
num_samples = len(dataset)
train_size = int(num_samples * 0.8)
val_size = num_samples - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=wandb.config.batch_size, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_data,   batch_size=wandb.config.batch_size, shuffle=False, drop_last=True)

# --- モデル構築 ---
d_model = wandb.config.d_ie 

model = KoopmanRoutesFormer(
    vocab_size=vocab_size,
    token_emb_dim=wandb.config.d_ie, 
    d_model=d_model,                 
    nhead=wandb.config.head_num,     
    num_layers=wandb.config.B_de,
    d_ff=wandb.config.d_ff,
    z_dim=wandb.config.z_dim,
    pad_token_id=network.N
)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
criterion_mse = nn.MSELoss() 
ce_loss_fn = nn.CrossEntropyLoss(ignore_index=network.N)

# --- 学習ループ ---
print(f"Training Start... (Koopman Loss: {wandb.config.use_koopman_loss})")
history = {"train_loss": [], "val_loss": []}

for epoch in range(wandb.config.epochs):
    # Training
    model.train()
    train_epoch_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    
    for batch, _ in pbar:
        route_batch = batch.to(device)
        tokenizer = Tokenization(network)
        
        input_tokens = tokenizer.tokenization(route_batch, mode="simple").long().to(device)
        target_tokens = tokenizer.tokenization(route_batch, mode="next").long().to(device)
        
        logits, z_hat, z_pred_next = model(input_tokens)
        
        # 1. CE Loss (予測精度)
        loss_ce = ce_loss_fn(logits.view(-1, vocab_size), target_tokens.view(-1))
        
        # 2. Dynamics Loss (Koopman制約) ★ここをスイッチで切り替え
        if wandb.config.use_koopman_loss:
            z_true_next = z_hat[:, 1:, :] 
            loss_dyn = criterion_mse(z_pred_next, z_true_next)
            loss = loss_ce + wandb.config.koopman_alpha * loss_dyn
        else:
            loss = loss_ce # Koopman制約なし
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_epoch_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    
    avg_train_loss = train_epoch_loss / len(train_loader)
    history["train_loss"].append(avg_train_loss)

    # Validation
    model.eval()
    val_epoch_loss = 0
    with torch.no_grad():
        for batch, _ in val_loader:
            route_batch = batch.to(device)
            tokenizer = Tokenization(network)
            
            input_tokens = tokenizer.tokenization(route_batch, mode="simple").long().to(device)
            target_tokens = tokenizer.tokenization(route_batch, mode="next").long().to(device)
            
            logits, z_hat, z_pred_next = model(input_tokens)
            
            loss_ce = ce_loss_fn(logits.view(-1, vocab_size), target_tokens.view(-1))
            
            # Valでも同様に計算
            if wandb.config.use_koopman_loss:
                z_true_next = z_hat[:, 1:, :] 
                loss_dyn = criterion_mse(z_pred_next, z_true_next)
                loss = loss_ce + wandb.config.koopman_alpha * loss_dyn
            else:
                loss = loss_ce
            
            val_epoch_loss += loss.item()

    avg_val_loss = val_epoch_loss / len(val_loader)
    history["val_loss"].append(avg_val_loss)

    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

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
        "pad_token_id": network.N,
        # Ablation設定も保存
        "use_koopman_loss": wandb.config.use_koopman_loss,
        "koopman_alpha": wandb.config.koopman_alpha
    },
    "history": history,
    "train_indices": train_data.indices,
    "val_indices": val_data.indices
}

torch.save(save_data, savefilename)
print(f"Model weights saved successfully at: {savefilename}")

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