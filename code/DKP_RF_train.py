import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import wandb
import os
from datetime import datetime
from DKP_RF import KoopmanRoutesFormer  # ★修正版モデルを使用
# 自作モジュール
from network import Network, expand_adjacency_matrix
from tokenization import Tokenization
import matplotlib.pyplot as plt


# ========================================
# 設定エリア - ここを変更してください
# ========================================
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# WandBの設定（ダミー）
class Dummy: pass
wandb = Dummy()
wandb.config = type("C", (), {
    "learning_rate": 1e-4, 
    "epochs": 200, 
    "batch_size": 32, 
    "d_ie": 64,
    "head_num": 4, 
    "d_ff": 128, 
    "B_de": 3,
    "z_dim": 16,
    "agent_emb_dim": 16,
    "stay_emb_dim": 16,
    "max_stay_count": 500,
    "holiday_emb_dim": 4,
    "time_zone_emb_dim": 4,
    "event_emb_dim": 4,

    "savefilename": "model_weights.pth",
    
    # ★ Prefix設定（新規）
    "use_variable_prefix": False,      # 可変長Prefixを使うか（False=固定長）
    "fixed_prefix_length": 8,        # 固定長の場合の長さ

    "prefix_lengths": [4, 6, 8],      # 可変長の場合の候補
    "min_future_length": 4,           # Futureの最小長
    
    # ★ 補助損失の重み（新規）
    "aux_count_weight": 0.01,  # カウント復元損失の重み (0で無効化)
    "aux_mode_weight": 0.01,   # モード分類損失の重み (0で無効化)

    # ★ Lyapunov正則化
    "lyap_alpha": 0.001,  # Lyapunov正則化の重み
    "lyap_eps": 1e-3,     # 許容（0でもOKだが少し入れると安定）
    
    # ★ 地理的距離損失
    "geo_alpha": 0.0,  # 地理的距離損失の重み (0で無効化)
    
    # ★ データ分割（Train 70%, Val 15%, Test 15%）
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    
    # ★ 分割インデックス保存
    "save_split_indices": True,
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

# 既存のコンテキスト配列もロード
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
print(f"Number of sequences: {len(trip_arr)}")

# ========================================
# 可変長Prefixデータセット
# ========================================

class VariablePrefixDataset(torch.utils.data.Dataset):
    """
    1つの系列から複数の (prefix, future) ペアを作成
    """
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
            
            # パディングを除いた実際の長さ
            r_np = r.cpu().numpy()
            pad_indices = np.where(r_np == self.pad_token_id)[0]
            real_len = pad_indices[0] if len(pad_indices) > 0 else len(r_np)
            
            if real_len < self.min_future_len + min(self.prefix_lengths):
                continue
            
            # 各prefix長でサンプル作成
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
    """
    固定長Prefixデータセット
    """
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
        
        # 有効なサンプルのみ抽出
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
    """可変長Prefixのcollate関数"""
    max_prefix_len = max(item['prefix_len'] for item in batch)
    max_future_len = max(item['future_len'] for item in batch)
    
    batch_size = len(batch)
    pad_token_id = 38
    
    # Prefix用テンソル
    prefix_tokens = torch.full((batch_size, max_prefix_len), pad_token_id, dtype=torch.long)
    prefix_holidays = torch.zeros((batch_size, max_prefix_len), dtype=torch.long)
    prefix_timezones = torch.zeros((batch_size, max_prefix_len), dtype=torch.long)
    prefix_events = torch.zeros((batch_size, max_prefix_len), dtype=torch.long)
    prefix_mask = torch.ones((batch_size, max_prefix_len), dtype=torch.bool)
    
    # Future用テンソル
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
        
        # Prefix
        prefix_tokens[i, :plen] = item['prefix']['tokens']
        prefix_holidays[i, :plen] = item['prefix']['holidays']
        prefix_timezones[i, :plen] = item['prefix']['timezones']
        prefix_events[i, :plen] = item['prefix']['events']
        prefix_mask[i, :plen] = False
        
        times.append(item['prefix']['time'])
        agents.append(item['prefix']['agent'])
        
        # Future
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

print("\n=== Initializing Model ===")

d_model = wandb.config.d_ie
num_agents = int(agent_pt.max().item()) + 1

# ★ 補助損失を使うかどうか
use_aux_loss = (wandb.config.aux_count_weight > 0 or wandb.config.aux_mode_weight > 0)

model = KoopmanRoutesFormer(
    vocab_size=vocab_size,
    token_emb_dim=wandb.config.d_ie, 
    d_model=d_model, 
    nhead=wandb.config.head_num,     
    num_layers=wandb.config.B_de,
    d_ff=wandb.config.d_ff,
    z_dim=wandb.config.z_dim,
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
    use_aux_loss=use_aux_loss,
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config.learning_rate, weight_decay=1e-4)

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
    "train_count": [], "val_count": [],
    "train_mode": [], "val_mode": [],
    "train_lyap": [], "val_lyap": [],
}

for epoch in range(wandb.config.epochs):
    # --- Training ---
    model.train()
    epoch_metrics_train = {
        "loss": 0.0, "ce": 0.0, "geo": 0.0,
        "count": 0.0, "mode": 0.0, "lyap": 0.0
    }
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{wandb.config.epochs} [Train]")
    
    for batch in pbar:
        # データ取得
        prefix_tokens = batch['prefix_tokens'].to(device)
        prefix_holidays = batch['prefix_holidays'].to(device)
        prefix_timezones = batch['prefix_timezones'].to(device)
        prefix_events = batch['prefix_events'].to(device)
        prefix_mask = batch['prefix_mask'].to(device)
        prefix_agents = batch['prefix_agents'].to(device)
        
        future_tokens = batch['future_tokens'].to(device)
        future_mask = batch['future_mask'].to(device)
        
        # 滞在カウント計算
        prefix_stay_counts = tokenizer.calculate_stay_counts(prefix_tokens).to(device)
        
        # Future の有効長
        K = (~future_mask).sum(dim=1).max().item()
        if K == 0:
            continue
        
        # ★ Forward (自律ロールアウト)
        outputs = model.forward_rollout(
            prefix_tokens=prefix_tokens,
            prefix_stay_counts=prefix_stay_counts,
            prefix_agent_ids=prefix_agents,
            prefix_holidays=prefix_holidays,
            prefix_time_zones=prefix_timezones,
            prefix_events=prefix_events,
            K=K,
            future_tokens=future_tokens[:, :K],
            prefix_mask=prefix_mask,
        )
        
        pred_logits = outputs['pred_logits']  # [B, K, vocab]
        
        # CE損失（パディング除外）
        valid_mask = ~future_mask[:, :K]
        ce_loss = nn.functional.cross_entropy(
            pred_logits.reshape(-1, vocab_size),
            future_tokens[:, :K].reshape(-1),
            reduction='none'
        )
        ce_loss = (ce_loss.view(-1, K) * valid_mask.float()).sum() / (valid_mask.sum() + 1e-8)
        
        # Geo損失
        if wandb.config.geo_alpha > 0:
            geo_loss = model.calc_geo_loss(pred_logits, future_tokens[:, :K])
        else:
            geo_loss = torch.tensor(0.0, device=device)
        
        # 補助損失
        aux_count_loss = torch.tensor(0.0, device=device)
        aux_mode_loss = torch.tensor(0.0, device=device)
        
        if outputs['aux_losses']:
            if 'count' in outputs['aux_losses']:
                aux_count_loss = outputs['aux_losses']['count']
            if 'mode' in outputs['aux_losses']:
                aux_mode_loss = outputs['aux_losses']['mode']
        
        # ★ Lyapunov正則化
        z_traj = outputs["z_traj"]  # [B, K+1, z_dim]
        
        # V(z) = ||z||^2
        V = (z_traj ** 2).sum(dim=-1)  # [B, K+1]
        dV = V[:, 1:] - (1.0 + wandb.config.lyap_eps) * V[:, :-1]  # [B, K]
        
        valid_mask_k = (~future_mask[:, :K]).float()  # [B, K]
        lyap_step = torch.relu(dV)  # [B, K]
        lyap_loss = (lyap_step * valid_mask_k).sum() / (valid_mask_k.sum() + 1e-8)
        
        # Total Loss
        total_loss = (
            ce_loss + 
            wandb.config.geo_alpha * geo_loss +
            wandb.config.aux_count_weight * aux_count_loss +
            wandb.config.aux_mode_weight * aux_mode_loss +
            wandb.config.lyap_alpha * lyap_loss
        )
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # 統計
        epoch_metrics_train["loss"] += total_loss.item()
        epoch_metrics_train["ce"] += ce_loss.item()
        epoch_metrics_train["geo"] += geo_loss.item()
        epoch_metrics_train["count"] += aux_count_loss.item()
        epoch_metrics_train["mode"] += aux_mode_loss.item()
        epoch_metrics_train["lyap"] += lyap_loss.item()
        
        pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})
    
    # Train平均
    n_train = len(train_loader)
    history["train_loss"].append(epoch_metrics_train["loss"] / n_train)
    history["train_ce"].append(epoch_metrics_train["ce"] / n_train)
    history["train_geo"].append(epoch_metrics_train["geo"] / n_train)
    history["train_count"].append(epoch_metrics_train["count"] / n_train)
    history["train_mode"].append(epoch_metrics_train["mode"] / n_train)
    history["train_lyap"].append(epoch_metrics_train["lyap"] / n_train)
    
    # --- Validation ---
    model.eval()
    epoch_metrics_val = {
        "loss": 0.0, "ce": 0.0, "geo": 0.0,
        "count": 0.0, "mode": 0.0, "lyap": 0.0
    }
    
    with torch.no_grad():
        for batch in val_loader:
            prefix_tokens = batch['prefix_tokens'].to(device)
            prefix_holidays = batch['prefix_holidays'].to(device)
            prefix_timezones = batch['prefix_timezones'].to(device)
            prefix_events = batch['prefix_events'].to(device)
            prefix_mask = batch['prefix_mask'].to(device)
            prefix_agents = batch['prefix_agents'].to(device)
            
            future_tokens = batch['future_tokens'].to(device)
            future_mask = batch['future_mask'].to(device)
            
            prefix_stay_counts = tokenizer.calculate_stay_counts(prefix_tokens).to(device)
            
            K = (~future_mask).sum(dim=1).max().item()
            if K == 0:
                continue
            
            outputs = model.forward_rollout(
                prefix_tokens=prefix_tokens,
                prefix_stay_counts=prefix_stay_counts,
                prefix_agent_ids=prefix_agents,
                prefix_holidays=prefix_holidays,
                prefix_time_zones=prefix_timezones,
                prefix_events=prefix_events,
                K=K,
                future_tokens=future_tokens[:, :K],
                prefix_mask=prefix_mask,
            )
            
            pred_logits = outputs['pred_logits']
            
            valid_mask = ~future_mask[:, :K]
            ce_loss = nn.functional.cross_entropy(
                pred_logits.reshape(-1, vocab_size),
                future_tokens[:, :K].reshape(-1),
                reduction='none'
            )
            ce_loss = (ce_loss.view(-1, K) * valid_mask.float()).sum() / (valid_mask.sum() + 1e-8)
            
            if wandb.config.geo_alpha > 0:
                geo_loss = model.calc_geo_loss(pred_logits, future_tokens[:, :K])
            else:
                geo_loss = torch.tensor(0.0, device=device)
            
            aux_count_loss = torch.tensor(0.0, device=device)
            aux_mode_loss = torch.tensor(0.0, device=device)
            if outputs['aux_losses']:
                if 'count' in outputs['aux_losses']:
                    aux_count_loss = outputs['aux_losses']['count']
                if 'mode' in outputs['aux_losses']:
                    aux_mode_loss = outputs['aux_losses']['mode']
            
            # Lyapunov正則化
            z_traj = outputs["z_traj"]  # [B, K+1, z_dim]
            V = (z_traj ** 2).sum(dim=-1)
            dV = V[:, 1:] - (1.0 + wandb.config.lyap_eps) * V[:, :-1]
            valid_mask_k = (~future_mask[:, :K]).float()
            lyap_step = torch.relu(dV)
            lyap_loss = (lyap_step * valid_mask_k).sum() / (valid_mask_k.sum() + 1e-8)
            
            total_loss = (
                ce_loss + 
                wandb.config.geo_alpha * geo_loss +
                wandb.config.aux_count_weight * aux_count_loss +
                wandb.config.aux_mode_weight * aux_mode_loss +
                wandb.config.lyap_alpha * lyap_loss
            )
            
            epoch_metrics_val["loss"] += total_loss.item()
            epoch_metrics_val["ce"] += ce_loss.item()
            epoch_metrics_val["geo"] += geo_loss.item()
            epoch_metrics_val["count"] += aux_count_loss.item()
            epoch_metrics_val["mode"] += aux_mode_loss.item()
            epoch_metrics_val["lyap"] += lyap_loss.item()
    
    # Val平均
    n_val = len(val_loader)
    history["val_loss"].append(epoch_metrics_val["loss"] / n_val)
    history["val_ce"].append(epoch_metrics_val["ce"] / n_val)
    history["val_geo"].append(epoch_metrics_val["geo"] / n_val)
    history["val_count"].append(epoch_metrics_val["count"] / n_val)
    history["val_mode"].append(epoch_metrics_val["mode"] / n_val)
    history["val_lyap"].append(epoch_metrics_val["lyap"] / n_val)
    
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
        "z_dim": wandb.config.z_dim,
        "pad_token_id": 38,
        "base_N": base_N,
        "use_aux_loss": use_aux_loss,
        "geo_alpha": wandb.config.geo_alpha,
        "aux_count_weight": wandb.config.aux_count_weight,
        "aux_mode_weight": wandb.config.aux_mode_weight,
        "lyap_alpha": wandb.config.lyap_alpha,
        "lyap_eps": wandb.config.lyap_eps,
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
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Detailed Loss History - Run ID: {run_id}', fontsize=16)
    
    # 1. Total Loss
    ax = axes[0, 0]
    ax.plot(epochs_range, history["train_loss"], label='Train', marker='.')
    ax.plot(epochs_range, history["val_loss"], label='Val', marker='.')
    ax.set_title('Total Weighted Loss')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # 2. CE Loss
    ax = axes[0, 1]
    ax.plot(epochs_range, history["train_ce"], label='Train', marker='.', color='orange')
    ax.plot(epochs_range, history["val_ce"], label='Val', marker='.', color='red')
    ax.set_title('Next Token Prediction (CE Loss)')
    ax.legend()
    ax.grid(True)
    
    # 3. Geo Loss
    ax = axes[0, 2]
    ax.plot(epochs_range, history["train_geo"], label='Train', marker='.', color='green')
    ax.plot(epochs_range, history["val_geo"], label='Val', marker='.', color='lime')
    ax.set_title('Geo Distance Loss')
    ax.legend()
    ax.grid(True)
    
    # 4. Lyapunov Loss
    ax = axes[1, 0]
    ax.plot(epochs_range, history["train_lyap"], label='Train', marker='.', color='purple')
    ax.plot(epochs_range, history["val_lyap"], label='Val', marker='.', color='magenta')
    ax.set_title('Lyapunov Loss')
    ax.legend()
    ax.grid(True)
    
    # 5. Count Loss
    ax = axes[1, 1]
    ax.plot(epochs_range, history["train_count"], label='Train', marker='.', color='brown')
    ax.plot(epochs_range, history["val_count"], label='Val', marker='.', color='pink')
    ax.set_title('Stay Count Reconstruction')
    ax.legend()
    ax.grid(True)
    
    # 6. Mode Loss
    ax = axes[1, 2]
    ax.plot(epochs_range, history["train_mode"], label='Train', marker='.', color='olive')
    ax.plot(epochs_range, history["val_mode"], label='Val', marker='.', color='yellowgreen')
    ax.set_title('Mode Reconstruction')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    graph_filename = stamp(f"loss_graph_detailed_{run_id}.png")
    plt.savefig(graph_filename)
    plt.close()
    print(f"Detailed loss graph saved at: {graph_filename}")

except Exception as e:
    print(f"Failed to plot detailed loss graph: {e}")

print("\n=== Training Complete ===")