"""
Transformer Ablation Model - Training Script

純粋な自己回帰Transformerの学習
KP_RF_train_final.pyと同じデータ・設定で学習
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os
from datetime import datetime
from Transformer_Ablation import TransformerAblation
# 自作モジュール
from network import Network, expand_adjacency_matrix
from tokenization import Tokenization
import matplotlib.pyplot as plt


# ========================================
# 設定エリア - ここを変更してください
# ========================================
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# 設定クラス
class Dummy: pass
wandb = Dummy()
wandb.config = type("C", (), {
    "learning_rate": 1e-4, 
    "epochs": 100, 
    "batch_size": 32, 
    "d_ie": 64,
    "head_num": 4, 
    "d_ff": 128, 
    "B_de": 3,
    "agent_emb_dim": 16,
    "stay_emb_dim": 16,
    "max_stay_count": 500,
    "holiday_emb_dim": 4,
    "time_zone_emb_dim": 4,
    "event_emb_dim": 4,

    "savefilename": "ablation_weights.pth",
    
    # ★ Prefix設定
    "use_variable_prefix": False,
    "prefix_lengths": [4, 6, 8],
    "fixed_prefix_length": 5,
    "min_future_length": 4,
    
    # ★ 地理的距離損失
    "geo_alpha": 0.0,  # 地理的距離損失の重み
    
    # ★ データ分割（Train 70%, Val 15%, Test 15%）
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    
    # ★ 分割インデックス保存先
    "save_split_indices": True,  # 推論時に使うため保存
})()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 保存ディレクトリ ---
out_dir = f"/home/mizutani/projects/RF/runs/{run_id}"
os.makedirs(out_dir, exist_ok=True)
def stamp(name): return os.path.join(out_dir, name)

print(f"Run ID: {run_id}")
print(f"Output directory: {out_dir}")

# ========================================
# データの準備
# ========================================

print("\n=== Loading Data ===")
trip_arrz = np.load('/home/mizutani/projects/RF/data/input_real_m5long.npz')
common_split_path = "/home/mizutani/projects/RF/data/common_split_indices_m5long.npz"

adj_matrix = torch.load('/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt', weights_only=True)

# 距離行列計算
import networkx as nx
def compute_shortest_path_distance_matrix(adj, directed=False):
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

if adj_matrix.shape[0] == 38:
    base_N = 19
    base_adj = adj_matrix[:base_N, :base_N]
else:
    base_adj = adj_matrix
    base_N = int(base_adj.shape[0])

print(f"Base N: {base_N}")

dist_mat_base = compute_shortest_path_distance_matrix(base_adj, directed=False)
expanded_adj = expand_adjacency_matrix(adj_matrix)
dummy_node_features = torch.zeros((len(adj_matrix), 1))
expanded_features = torch.cat([dummy_node_features, dummy_node_features], dim=0)
network = Network(expanded_adj, expanded_features)

trip_arr = trip_arrz['route_arr']
time_arr = trip_arrz['time_arr']
agent_ids_arr = trip_arrz['agent_ids'] if 'agent_ids' in trip_arrz else np.zeros(len(trip_arr), dtype=int)

holiday_arr = trip_arrz['holiday_arr']
timezone_arr = trip_arrz['time_zone_arr']
event_arr = trip_arrz['event_arr']

# Tensor化
route_pt = torch.from_numpy(trip_arr).long()
time_pt = torch.from_numpy(time_arr)
agent_pt = torch.from_numpy(agent_ids_arr).long()
holiday_pt = torch.from_numpy(holiday_arr).long()
timezone_pt = torch.from_numpy(timezone_arr).long()
event_pt = torch.from_numpy(event_arr).long()

vocab_size = network.N + 4
print(f"Vocabulary size: {vocab_size}")

# ========================================
# データセット（KP_RF_train_final.pyから流用）
# ========================================

class VariablePrefixDataset(torch.utils.data.Dataset):
    def __init__(self, routes, times, agents, holidays, timezones, events, 
                 prefix_lengths, min_future_len, pad_token_id=38):
        self.pad_token_id = pad_token_id
        self.prefix_lengths = prefix_lengths
        self.min_future_len = min_future_len
        
        self.samples = []
        self._create_samples(routes, times, agents, holidays, timezones, events)
    
    def _create_samples(self, routes, times, agents, holidays, timezones, events):
        for idx in range(len(routes)):
            r = routes[idx]
            t = times[idx]
            a = agents[idx]
            h = holidays[idx]
            tz = timezones[idx]
            e = events[idx]
            
            r_np = r.cpu().numpy()
            pad_indices = np.where(r_np == self.pad_token_id)[0]
            real_len = pad_indices[0] if len(pad_indices) > 0 else len(r_np)
            
            if real_len < self.min_future_len + min(self.prefix_lengths):
                continue
            
            for prefix_len in self.prefix_lengths:
                if real_len < prefix_len + self.min_future_len:
                    continue
                
                prefix_data = {
                    'tokens': r[:prefix_len],
                    'time': t,
                    'agent': a,
                    'holidays': h[:prefix_len],
                    'timezones': tz[:prefix_len],
                    'events': e[:prefix_len],
                }
                
                future_data = {
                    'tokens': r[prefix_len:real_len],
                    'holidays': h[prefix_len:real_len],
                    'timezones': tz[prefix_len:real_len],
                    'events': e[prefix_len:real_len],
                }
                
                self.samples.append({
                    'prefix': prefix_data,
                    'future': future_data,
                    'prefix_len': prefix_len,
                    'future_len': real_len - prefix_len,
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class FixedPrefixDataset(torch.utils.data.Dataset):
    def __init__(self, routes, times, agents, holidays, timezones, events,
                 prefix_len, pad_token_id=38):
        self.routes = routes
        self.times = times
        self.agents = agents
        self.holidays = holidays
        self.timezones = timezones
        self.events = events
        self.prefix_len = prefix_len
        self.pad_token_id = pad_token_id
        
        self.valid_indices = []
        for idx in range(len(routes)):
            r_np = routes[idx].cpu().numpy()
            pad_indices = np.where(r_np == pad_token_id)[0]
            real_len = pad_indices[0] if len(pad_indices) > 0 else len(r_np)
            if real_len > prefix_len:
                self.valid_indices.append(idx)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        r = self.routes[actual_idx]
        
        r_np = r.cpu().numpy()
        pad_indices = np.where(r_np == self.pad_token_id)[0]
        real_len = pad_indices[0] if len(pad_indices) > 0 else len(r_np)
        
        prefix_data = {
            'tokens': r[:self.prefix_len],
            'time': self.times[actual_idx],
            'agent': self.agents[actual_idx],
            'holidays': self.holidays[actual_idx][:self.prefix_len],
            'timezones': self.timezones[actual_idx][:self.prefix_len],
            'events': self.events[actual_idx][:self.prefix_len],
        }
        
        future_data = {
            'tokens': r[self.prefix_len:real_len],
            'holidays': self.holidays[actual_idx][self.prefix_len:real_len],
            'timezones': self.timezones[actual_idx][self.prefix_len:real_len],
            'events': self.events[actual_idx][self.prefix_len:real_len],
        }
        
        return {
            'prefix': prefix_data,
            'future': future_data,
            'prefix_len': self.prefix_len,
            'future_len': real_len - self.prefix_len,
        }


def collate_variable_prefix(batch):
    max_prefix_len = max(item['prefix_len'] for item in batch)
    max_future_len = max(item['future_len'] for item in batch)
    
    batch_size = len(batch)
    pad_token_id = 38
    
    prefix_tokens = torch.full((batch_size, max_prefix_len), pad_token_id, dtype=torch.long)
    prefix_holidays = torch.zeros((batch_size, max_prefix_len), dtype=torch.long)
    prefix_timezones = torch.zeros((batch_size, max_prefix_len), dtype=torch.long)
    prefix_events = torch.zeros((batch_size, max_prefix_len), dtype=torch.long)
    prefix_mask = torch.ones((batch_size, max_prefix_len), dtype=torch.bool)
    
    future_tokens = torch.full((batch_size, max_future_len), pad_token_id, dtype=torch.long)
    future_holidays = torch.zeros((batch_size, max_future_len), dtype=torch.long)
    future_timezones = torch.zeros((batch_size, max_future_len), dtype=torch.long)
    future_events = torch.zeros((batch_size, max_future_len), dtype=torch.long)
    future_mask = torch.ones((batch_size, max_future_len), dtype=torch.bool)
    
    times = []
    agents = []
    
    for i, item in enumerate(batch):
        plen = item['prefix_len']
        flen = item['future_len']
        
        prefix_tokens[i, :plen] = item['prefix']['tokens']
        prefix_holidays[i, :plen] = item['prefix']['holidays']
        prefix_timezones[i, :plen] = item['prefix']['timezones']
        prefix_events[i, :plen] = item['prefix']['events']
        prefix_mask[i, :plen] = False
        
        times.append(item['prefix']['time'])
        agents.append(item['prefix']['agent'])
        
        future_tokens[i, :flen] = item['future']['tokens']
        future_holidays[i, :flen] = item['future']['holidays']
        future_timezones[i, :flen] = item['future']['timezones']
        future_events[i, :flen] = item['future']['events']
        future_mask[i, :flen] = False
    
    return {
        'prefix_tokens': prefix_tokens,
        'prefix_holidays': prefix_holidays,
        'prefix_timezones': prefix_timezones,
        'prefix_events': prefix_events,
        'prefix_mask': prefix_mask,
        'prefix_agents': torch.tensor(agents, dtype=torch.long),
        'times': torch.tensor(times),
        'future_tokens': future_tokens,
        'future_holidays': future_holidays,
        'future_timezones': future_timezones,
        'future_events': future_events,
        'future_mask': future_mask,
    }


# ========================================
# 共通分割インデックスのロード（系列レベル）
# ========================================

print("\n=== Loading Common Split Indices ===")

if not os.path.exists(common_split_path):
    raise FileNotFoundError(
        f"Common split file not found: {common_split_path}\n"
        "Please run 'python create_common_split.py' first!"
    )

split_data = np.load(common_split_path)
train_seq_indices = split_data['train_sequences']
val_seq_indices = split_data['val_sequences']
test_seq_indices = split_data['test_sequences']

print(f"Train sequences: {len(train_seq_indices)}")
print(f"Val sequences: {len(val_seq_indices)}")
print(f"Test sequences: {len(test_seq_indices)}")

# ========================================
# 各分割でデータセット作成
# ========================================

print("\n=== Creating Datasets ===")

# Train用データセット
if wandb.config.use_variable_prefix:
    print(f"Using variable prefix with lengths: {wandb.config.prefix_lengths}")
    train_dataset = VariablePrefixDataset(
        route_pt[train_seq_indices], 
        time_pt[train_seq_indices], 
        agent_pt[train_seq_indices],
        holiday_pt[train_seq_indices], 
        timezone_pt[train_seq_indices], 
        event_pt[train_seq_indices],
        prefix_lengths=wandb.config.prefix_lengths,
        min_future_len=wandb.config.min_future_length,
    )
    val_dataset = VariablePrefixDataset(
        route_pt[val_seq_indices], 
        time_pt[val_seq_indices], 
        agent_pt[val_seq_indices],
        holiday_pt[val_seq_indices], 
        timezone_pt[val_seq_indices], 
        event_pt[val_seq_indices],
        prefix_lengths=wandb.config.prefix_lengths,
        min_future_len=wandb.config.min_future_length,
    )
    test_dataset = VariablePrefixDataset(
        route_pt[test_seq_indices], 
        time_pt[test_seq_indices], 
        agent_pt[test_seq_indices],
        holiday_pt[test_seq_indices], 
        timezone_pt[test_seq_indices], 
        event_pt[test_seq_indices],
        prefix_lengths=wandb.config.prefix_lengths,
        min_future_len=wandb.config.min_future_length,
    )
else:
    print(f"Using fixed prefix length: {wandb.config.fixed_prefix_length}")
    train_dataset = FixedPrefixDataset(
        route_pt[train_seq_indices], 
        time_pt[train_seq_indices], 
        agent_pt[train_seq_indices],
        holiday_pt[train_seq_indices], 
        timezone_pt[train_seq_indices], 
        event_pt[train_seq_indices],
        prefix_len=wandb.config.fixed_prefix_length,
    )
    val_dataset = FixedPrefixDataset(
        route_pt[val_seq_indices], 
        time_pt[val_seq_indices], 
        agent_pt[val_seq_indices],
        holiday_pt[val_seq_indices], 
        timezone_pt[val_seq_indices], 
        event_pt[val_seq_indices],
        prefix_len=wandb.config.fixed_prefix_length,
    )
    test_dataset = FixedPrefixDataset(
        route_pt[test_seq_indices], 
        time_pt[test_seq_indices], 
        agent_pt[test_seq_indices],
        holiday_pt[test_seq_indices], 
        timezone_pt[test_seq_indices], 
        event_pt[test_seq_indices],
        prefix_len=wandb.config.fixed_prefix_length,
    )

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# ★ 分割情報を保存（推論時に使用）
if wandb.config.save_split_indices:
    split_path = stamp("split_info.npz")
    np.savez(
        split_path,
        train_sequences=train_seq_indices,
        val_sequences=val_seq_indices,
        test_sequences=test_seq_indices,
        common_split_path=common_split_path,
    )
    print(f"Saved split info to: {split_path}")

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True, 
                          collate_fn=collate_variable_prefix, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=False, 
                        collate_fn=collate_variable_prefix, drop_last=True)


# ========================================
# モデル初期化
# ========================================

print("\n=== Initializing Ablation Model ===")

d_model = wandb.config.d_ie
num_agents = int(agent_pt.max().item()) + 1

model = TransformerAblation(
    vocab_size=vocab_size,
    token_emb_dim=wandb.config.d_ie,
    d_model=d_model,
    nhead=wandb.config.head_num,
    num_layers=wandb.config.B_de,
    d_ff=wandb.config.d_ff,
    pad_token_id=38,
    dist_mat_base=dist_mat_base if wandb.config.geo_alpha > 0 else None,
    base_N=base_N,
    num_agents=num_agents,
    agent_emb_dim=wandb.config.agent_emb_dim,
    max_stay_count=wandb.config.max_stay_count,
    stay_emb_dim=wandb.config.stay_emb_dim,
    holiday_emb_dim=wandb.config.holiday_emb_dim,
    time_zone_emb_dim=wandb.config.time_zone_emb_dim,
    event_emb_dim=wandb.config.event_emb_dim,
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

# Tokenizer
tokenizer = Tokenization(network)

# ========================================
# 学習ループ
# ========================================

print("\n=== Starting Training ===")

history = {
    "train_loss": [], "val_loss": [],
    "train_ce": [], "val_ce": [],
    "train_geo": [], "val_geo": [],
}

for epoch in range(wandb.config.epochs):
    # --- Training ---
    model.train()
    epoch_metrics_train = {"loss": 0.0, "ce": 0.0, "geo": 0.0}
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{wandb.config.epochs} [Train]")
    
    for batch in pbar:
        prefix_tokens = batch['prefix_tokens'].to(device)
        prefix_holidays = batch['prefix_holidays'].to(device)
        prefix_timezones = batch['prefix_timezones'].to(device)
        prefix_events = batch['prefix_events'].to(device)
        prefix_mask = batch['prefix_mask'].to(device)
        prefix_agents = batch['prefix_agents'].to(device)
        
        future_tokens = batch['future_tokens'].to(device)
        future_holidays = batch['future_holidays'].to(device)
        future_timezones = batch['future_timezones'].to(device)
        future_events = batch['future_events'].to(device)
        future_mask = batch['future_mask'].to(device)
        
        # 滞在カウント計算
        prefix_stay_counts = tokenizer.calculate_stay_counts(prefix_tokens).to(device)
        future_stay_counts = tokenizer.calculate_stay_counts(future_tokens).to(device)
        
        # Forward
        outputs = model.forward_with_prefix_future(
            prefix_tokens, prefix_stay_counts, prefix_agents,
            prefix_holidays, prefix_timezones, prefix_events,
            future_tokens, future_stay_counts,
            future_holidays, future_timezones, future_events,
            prefix_mask, future_mask
        )
        
        logits = outputs['logits']  # [B, T_total, vocab]
        prefix_len = outputs['prefix_len']
        
        # ターゲット：入力を1ステップずらしたもの
        # logits[:, t] は tokens[:, t+1] を予測
        all_tokens = torch.cat([prefix_tokens, future_tokens], dim=1)
        target_tokens = all_tokens[:, 1:]  # [B, T_total-1]
        pred_logits = logits[:, :-1, :]   # [B, T_total-1, vocab]
        
        # パディングマスク
        all_mask = torch.cat([prefix_mask, future_mask], dim=1)
        valid_mask = ~all_mask[:, 1:]  # [B, T_total-1]
        
        # CE損失
        ce_loss = nn.functional.cross_entropy(
            pred_logits.reshape(-1, vocab_size),
            target_tokens.reshape(-1),
            reduction='none'
        )
        ce_loss = (ce_loss.view(valid_mask.shape) * valid_mask.float()).sum() / (valid_mask.sum() + 1e-8)
        
        # Geo損失
        if wandb.config.geo_alpha > 0:
            geo_loss = model.calc_geo_loss(pred_logits, target_tokens)
        else:
            geo_loss = torch.tensor(0.0, device=device)
        
        total_loss = ce_loss + wandb.config.geo_alpha * geo_loss
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_metrics_train["loss"] += total_loss.item()
        epoch_metrics_train["ce"] += ce_loss.item()
        epoch_metrics_train["geo"] += geo_loss.item()
        
        pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})
    
    # Train平均
    n_train = len(train_loader)
    history["train_loss"].append(epoch_metrics_train["loss"] / n_train)
    history["train_ce"].append(epoch_metrics_train["ce"] / n_train)
    history["train_geo"].append(epoch_metrics_train["geo"] / n_train)
    
    # --- Validation ---
    model.eval()
    epoch_metrics_val = {"loss": 0.0, "ce": 0.0, "geo": 0.0}
    
    with torch.no_grad():
        for batch in val_loader:
            prefix_tokens = batch['prefix_tokens'].to(device)
            prefix_holidays = batch['prefix_holidays'].to(device)
            prefix_timezones = batch['prefix_timezones'].to(device)
            prefix_events = batch['prefix_events'].to(device)
            prefix_mask = batch['prefix_mask'].to(device)
            prefix_agents = batch['prefix_agents'].to(device)
            
            future_tokens = batch['future_tokens'].to(device)
            future_holidays = batch['future_holidays'].to(device)
            future_timezones = batch['future_timezones'].to(device)
            future_events = batch['future_events'].to(device)
            future_mask = batch['future_mask'].to(device)
            
            prefix_stay_counts = tokenizer.calculate_stay_counts(prefix_tokens).to(device)
            future_stay_counts = tokenizer.calculate_stay_counts(future_tokens).to(device)
            
            outputs = model.forward_with_prefix_future(
                prefix_tokens, prefix_stay_counts, prefix_agents,
                prefix_holidays, prefix_timezones, prefix_events,
                future_tokens, future_stay_counts,
                future_holidays, future_timezones, future_events,
                prefix_mask, future_mask
            )
            
            logits = outputs['logits']
            all_tokens = torch.cat([prefix_tokens, future_tokens], dim=1)
            target_tokens = all_tokens[:, 1:]
            pred_logits = logits[:, :-1, :]
            
            all_mask = torch.cat([prefix_mask, future_mask], dim=1)
            valid_mask = ~all_mask[:, 1:]
            
            ce_loss = nn.functional.cross_entropy(
                pred_logits.reshape(-1, vocab_size),
                target_tokens.reshape(-1),
                reduction='none'
            )
            ce_loss = (ce_loss.view(valid_mask.shape) * valid_mask.float()).sum() / (valid_mask.sum() + 1e-8)
            
            if wandb.config.geo_alpha > 0:
                geo_loss = model.calc_geo_loss(pred_logits, target_tokens)
            else:
                geo_loss = torch.tensor(0.0, device=device)
            
            total_loss = ce_loss + wandb.config.geo_alpha * geo_loss
            
            epoch_metrics_val["loss"] += total_loss.item()
            epoch_metrics_val["ce"] += ce_loss.item()
            epoch_metrics_val["geo"] += geo_loss.item()
    
    # Val平均
    n_val = len(val_loader)
    history["val_loss"].append(epoch_metrics_val["loss"] / n_val)
    history["val_ce"].append(epoch_metrics_val["ce"] / n_val)
    history["val_geo"].append(epoch_metrics_val["geo"] / n_val)
    
    print(f"Epoch {epoch+1}: Train Loss = {history['train_loss'][-1]:.4f} | Val Loss = {history['val_loss'][-1]:.4f}")

# ========================================
# 保存処理
# ========================================

print("\n=== Saving Model ===")
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
        "pad_token_id": 38,
        "base_N": base_N,
        "geo_alpha": wandb.config.geo_alpha,
        "use_variable_prefix": wandb.config.use_variable_prefix,
        "prefix_lengths": wandb.config.prefix_lengths,
    },
    "history": history,
    "split_sequences": {
        "train": train_seq_indices,
        "val": val_seq_indices,
        "test": test_seq_indices,
        "common_split_path": common_split_path,
    }
}

torch.save(save_data, savefilename)
print(f"Model saved to: {savefilename}")

# ========================================
# グラフ描画
# ========================================

print("\n=== Plotting Loss History ===")
try:
    epochs_range = range(1, len(history["train_loss"]) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Ablation Model Loss History - Run ID: {run_id}', fontsize=14)
    
    # 1. Total Loss
    ax = axes[0]
    ax.plot(epochs_range, history["train_loss"], label='Train', marker='.')
    ax.plot(epochs_range, history["val_loss"], label='Val', marker='.')
    ax.set_title('Total Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # 2. CE Loss
    ax = axes[1]
    ax.plot(epochs_range, history["train_ce"], label='Train', marker='.')
    ax.plot(epochs_range, history["val_ce"], label='Val', marker='.')
    ax.set_title('CE Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True)
    
    # 3. Geo Loss
    ax = axes[2]
    ax.plot(epochs_range, history["train_geo"], label='Train', marker='.')
    ax.plot(epochs_range, history["val_geo"], label='Val', marker='.')
    ax.set_title('Geo Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    graph_filename = stamp(f"loss_graph_{run_id}.png")
    plt.savefig(graph_filename)
    plt.close()
    print(f"Loss graph saved to: {graph_filename}")

except Exception as e:
    print(f"Failed to plot loss graph: {e}")

print("\n=== Training Complete ===")