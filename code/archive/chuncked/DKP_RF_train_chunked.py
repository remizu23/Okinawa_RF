"""
Koopman-Transformer Training Script with Chunked Dataset
従来のDKP_RF_train.pyを踏襲し、K, Lパラメータ対応版に拡張

学習データ生成方針:
- 元経路が長さ N の場合、以下のように分割:
  サンプル1: prefix=[0:K], target=[K:K+L]
  サンプル2: prefix=[0:K+L], target=[K+L:K+2L] (存在する場合)
  ...
- 最後のサンプルの target が L 未満でも実際の長さで学習
- 経路長が K 未満のサンプルはスキップ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# =========================================================
# Config
# =========================================================
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# データパス
DATA_PATH = '/home/mizutani/projects/RF/data/input_real_m5.npz'
SPLIT_INDICES_PATH = '/home/mizutani/projects/RF/data/common_split_indices_m5.npz'

# Chunking パラメータ
K = 4  # Prefix length
L = 4  # Rollout length

# モデルパラメータ
PAD_TOKEN_ID = 38
BASE_N = 19  # Move/Stay の境界
VOCAB_SIZE = 38 + 4

Z_DIM = 16
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 3
D_FF = 128

MAX_STAY_COUNT = 500
TOKEN_EMB_DIM = 64
AGENT_EMB_DIM = 16
STAY_EMB_DIM = 16
HOLIDAY_EMB_DIM = 4
TIME_ZONE_EMB_DIM = 4
EVENT_EMB_DIM = 4

# 学習パラメータ
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# 損失の重み（従来コードを踏襲）
LAMBDA_CE = 1.0
LAMBDA_GEO = 0.0          # 使用しない
LAMBDA_COUNT = 0.01       # aux_count_weight
LAMBDA_MODE = 0.01        # aux_mode_weight
LAMBDA_LYAPUNOV = 0.001   # lyap_alpha
LYAP_EPS = 1e-3           # lyap_eps

# 出力ディレクトリ
OUTPUT_DIR = f'/home/mizutani/projects/RF/runs/{run_id}_K{K}_L{L}'

# =========================================================
# Dataset
# =========================================================

class ChunkedDataset(Dataset):
    """
    教師系列で「prefixを累積で伸ばしながら」多段サンプルを作るデータセット。
    Global Config (K, L, BASE_N, etc.) を直接参照する直打ち仕様。
    """

    def __init__(self, split_indices, split='train'):
        """
        Args:
            split_indices: main()から渡される辞書 {'train_sequences': [...], ...}
            split: 'train' / 'val'
        """
        # --- Global Configs (Hard-coded) ---
        self.K = int(K)
        self.L = int(L)
        self.pad_token = int(PAD_TOKEN_ID)
        self.base_N = int(BASE_N)
        
        # --- Fixed Defaults (Requested) ---
        self.stop_at_end_token = False
        self.end_token = 39
        self.add_start_offsets = False
        self.start_offsets = 0

        # --- Data Loading (Global Path) ---
        data = np.load(DATA_PATH)
        routes = data["route_arr"]
        holidays = data["holiday_arr"]
        time_zones = data["time_zone_arr"]
        events = data["event_arr"]
        agent_ids = data["agent_ids"] if "agent_ids" in data else np.zeros(len(routes), dtype=np.int64)

        indices = split_indices[f"{split}_sequences"]

        self.samples = []
        n_traj_used = 0

        for idx in indices:
            route = routes[idx]
            holiday = holidays[idx]
            time_zone = time_zones[idx]
            event = events[idx]
            agent_id = agent_ids[idx]

            # --- 有効部分を切り出し（PADで打ち切り） ---
            valid_mask = (route != self.pad_token)
            valid_len = int(valid_mask.sum())

            if valid_len < self.K + 1:
                continue

            route_valid = route[:valid_len]
            holiday_valid = holiday[:valid_len]
            time_zone_valid = time_zone[:valid_len]
            event_valid = event[:valid_len]

            # --- <e> で打ち切り（False設定なのでスキップ） ---
            if self.stop_at_end_token:
                end_pos = np.where(route_valid == self.end_token)[0]
                if len(end_pos) > 0:
                    cut = int(end_pos[0]) + 1
                    route_valid = route_valid[:cut]
                    holiday_valid = holiday_valid[:cut]
                    time_zone_valid = time_zone_valid[:cut]
                    event_valid = event_valid[:cut]
                    valid_len = cut
                    if valid_len < self.K + 1:
                        continue

            # --- 1軌跡から複数サンプル生成 ---
            # add_start_offsets = False なので [0] 固定
            if self.add_start_offsets:
                if self.start_offsets <= 0:
                    start_list = list(range(self.L))
                else:
                    start_list = list(range(self.start_offsets))
            else:
                start_list = [0]

            any_used = False

            for s in start_list:
                if s >= valid_len:
                    continue

                # prefix_end を K から始めて L ずつ伸ばす
                prefix_end = s + self.K
                if prefix_end > valid_len:
                    continue

                while True:
                    target_start = prefix_end
                    target_end = min(prefix_end + self.L, valid_len)
                    target_len = int(target_end - target_start)

                    if target_len <= 0:
                        break

                    prefix_tokens = route_valid[s:prefix_end]
                    prefix_holidays = holiday_valid[s:prefix_end]
                    prefix_time_zones = time_zone_valid[s:prefix_end]
                    prefix_events = event_valid[s:prefix_end]

                    target_tokens = route_valid[target_start:target_end]

                    prefix_stay_counts = self._calculate_stay_counts(prefix_tokens)

                    self.samples.append({
                        "prefix_tokens": torch.tensor(prefix_tokens, dtype=torch.long),
                        "prefix_stay_counts": torch.tensor(prefix_stay_counts, dtype=torch.long),
                        "prefix_holidays": torch.tensor(prefix_holidays, dtype=torch.long),
                        "prefix_time_zones": torch.tensor(prefix_time_zones, dtype=torch.long),
                        "prefix_events": torch.tensor(prefix_events, dtype=torch.long),
                        "agent_id": torch.tensor(int(agent_id), dtype=torch.long),
                        "target_tokens": torch.tensor(target_tokens, dtype=torch.long),
                        "target_length": target_len,
                    })

                    any_used = True

                    # 次：教師系列で prefix を L だけ伸ばす
                    prefix_end += self.L
                    if prefix_end > valid_len:
                        break

            if any_used:
                n_traj_used += 1

        print(f"[{split.upper()}] Generated {len(self.samples)} samples from {n_traj_used} trajectories (given {len(indices)} trajectories)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        return self.samples[i]

    def _calculate_stay_counts(self, token_seq_np: np.ndarray):
        counts = np.zeros(len(token_seq_np), dtype=np.int64)
        c = 0
        for i, tok in enumerate(token_seq_np):
            if tok >= self.base_N and tok != self.pad_token:
                c += 1
                counts[i] = c
            else:
                c = 0
                counts[i] = 0
        return counts


def collate_fn(batch):
    """
    ChunkedDataset 用の collate (旧 collate_chunked)
    - prefix が可変長になったため、パディング処理を追加
    - PAD_TOKEN_ID (Global) を使用
    """
    B = len(batch)
    if B == 0:
        return None

    pad_token = int(PAD_TOKEN_ID)

    # バッチ内の最大長を取得
    max_lp = max(int(item["prefix_tokens"].shape[0]) for item in batch)
    max_lt = max(int(item["target_tokens"].shape[0]) for item in batch)

    # Prefix側の初期化（PADで埋める）
    prefix_tokens = torch.full((B, max_lp), pad_token, dtype=torch.long)
    prefix_stay_counts = torch.zeros((B, max_lp), dtype=torch.long)
    prefix_holidays = torch.zeros((B, max_lp), dtype=torch.long)
    prefix_time_zones = torch.zeros((B, max_lp), dtype=torch.long)
    prefix_events = torch.zeros((B, max_lp), dtype=torch.long)
    prefix_mask = torch.ones((B, max_lp), dtype=torch.bool)   # True=PAD（無効）

    # Target側の初期化
    target_tokens = torch.full((B, max_lt), pad_token, dtype=torch.long)
    target_mask = torch.ones((B, max_lt), dtype=torch.bool)   # True=PAD（無効）
    target_lengths = torch.zeros((B,), dtype=torch.long)

    agent_ids = torch.zeros((B,), dtype=torch.long)

    for i, item in enumerate(batch):
        lp = item["prefix_tokens"].shape[0]
        lt = item["target_tokens"].shape[0]

        # Prefix詰め込み
        prefix_tokens[i, :lp] = item["prefix_tokens"]
        prefix_stay_counts[i, :lp] = item["prefix_stay_counts"]
        prefix_holidays[i, :lp] = item["prefix_holidays"]
        prefix_time_zones[i, :lp] = item["prefix_time_zones"]
        prefix_events[i, :lp] = item["prefix_events"]
        prefix_mask[i, :lp] = False  # データ部分はFalse

        # Target詰め込み
        target_tokens[i, :lt] = item["target_tokens"]
        target_mask[i, :lt] = False
        target_lengths[i] = int(item["target_length"])

        agent_ids[i] = item["agent_id"]

    return {
        "prefix_tokens": prefix_tokens,
        "prefix_stay_counts": prefix_stay_counts,
        "prefix_holidays": prefix_holidays,
        "prefix_time_zones": prefix_time_zones,
        "prefix_events": prefix_events,
        "prefix_mask": prefix_mask,    # 重要: モデルへ渡すマスク
        "prefix_agents": agent_ids,
        "target_tokens": target_tokens,
        "target_mask": target_mask,
        "target_lengths": target_lengths,
    }

# =========================================================
# Training Functions
# =========================================================

def compute_losses(model, batch, device):
    """
    損失を計算
    ※ PrefixMaskをモデルに渡すように変更
    """
    if batch is None:
        return None

    # バッチデータをデバイスに移動
    prefix_tokens = batch['prefix_tokens'].to(device)
    prefix_stay_counts = batch['prefix_stay_counts'].to(device)
    prefix_holidays = batch['prefix_holidays'].to(device)
    prefix_time_zones = batch['prefix_time_zones'].to(device)
    prefix_events = batch['prefix_events'].to(device)
    prefix_agents = batch['prefix_agents'].to(device)
    
    # 新規: Prefix Mask (可変長対応のため必須)
    prefix_mask = batch['prefix_mask'].to(device)

    target_tokens = batch['target_tokens'].to(device)
    target_mask = batch['target_mask'].to(device)  # True=パディング
    
    # Future の有効長
    K_rollout = (~target_mask).sum(dim=1).max().item()
    if K_rollout == 0:
        return None
    
    # Forward rollout
    # 変更点: prefix_mask=prefix_mask を指定してパディングを無視させる
    outputs = model.forward_rollout(
        prefix_tokens=prefix_tokens,
        prefix_stay_counts=prefix_stay_counts,
        prefix_agent_ids=prefix_agents,
        prefix_holidays=prefix_holidays,
        prefix_time_zones=prefix_time_zones,
        prefix_events=prefix_events,
        K=K_rollout,  # Rollout steps
        future_tokens=target_tokens[:, :K_rollout],
        prefix_mask=prefix_mask,  # <--- ここでマスクを渡す
    )
    
    pred_logits = outputs['pred_logits']  # [B, K_rollout, Vocab]
    
    # 1. CE Loss（パディング除外）
    valid_mask = ~target_mask[:, :K_rollout]  # [B, K_rollout], False=パディング
    ce_loss = F.cross_entropy(
        pred_logits.reshape(-1, VOCAB_SIZE),
        target_tokens[:, :K_rollout].reshape(-1),
        reduction='none'
    )
    ce_loss = (ce_loss.view(-1, K_rollout) * valid_mask.float()).sum() / (valid_mask.sum() + 1e-8)
    
    # 2. Geo Loss
    if LAMBDA_GEO > 0:
        geo_loss = model.calc_geo_loss(pred_logits, target_tokens[:, :K_rollout])
    else:
        geo_loss = torch.tensor(0.0, device=device)
    
    # 3. Auxiliary losses
    aux_count_loss = torch.tensor(0.0, device=device)
    aux_mode_loss = torch.tensor(0.0, device=device)
    
    if outputs['aux_losses']:
        if 'count' in outputs['aux_losses']:
            aux_count_loss = outputs['aux_losses']['count']
        if 'mode' in outputs['aux_losses']:
            aux_mode_loss = outputs['aux_losses']['mode']
    
    # 4. Lyapunov正則化
    z_traj = outputs["z_traj"]
    V = (z_traj ** 2).sum(dim=-1)
    dV = V[:, 1:] - (1.0 + LYAP_EPS) * V[:, :-1]
    
    valid_mask_k = (~target_mask[:, :K_rollout]).float()
    lyap_step = torch.relu(dV)
    lyap_loss = (lyap_step * valid_mask_k).sum() / (valid_mask_k.sum() + 1e-8)
    
    # Total Loss
    total_loss = (
        LAMBDA_CE * ce_loss + 
        LAMBDA_GEO * geo_loss +
        LAMBDA_COUNT * aux_count_loss +
        LAMBDA_MODE * aux_mode_loss +
        LAMBDA_LYAPUNOV * lyap_loss
    )
    
    losses = {
        'loss': total_loss,
        'ce': ce_loss,
        'geo': geo_loss,
        'count': aux_count_loss,
        'mode': aux_mode_loss,
        'lyap': lyap_loss,
    }
    
    return losses

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    epoch_metrics = {"loss": 0.0, "ce": 0.0, "geo": 0.0, "count": 0.0, "mode": 0.0, "lyap": 0.0}
    n_eff = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        losses = compute_losses(model, batch, device)
        if losses is None:
            continue

        loss = losses["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k, v in losses.items():
            epoch_metrics[k] += float(v.detach().item())

        n_eff += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    if n_eff == 0:
        return epoch_metrics

    for k in epoch_metrics:
        epoch_metrics[k] /= n_eff
    return epoch_metrics


def validate(model, dataloader, device):
    model.eval()
    epoch_metrics = {"loss": 0.0, "ce": 0.0, "geo": 0.0, "count": 0.0, "mode": 0.0, "lyap": 0.0}
    n_eff = 0

    with torch.no_grad():
        for batch in dataloader:
            losses = compute_losses(model, batch, device)
            if losses is None:
                continue

            for k, v in losses.items():
                epoch_metrics[k] += float(v.detach().item())
            n_eff += 1

    if n_eff == 0:
        return epoch_metrics

    for k in epoch_metrics:
        epoch_metrics[k] /= n_eff
    return epoch_metrics


# =========================================================
# Main
# =========================================================

def main():
    print("="*60)
    print("Koopman-Transformer Training with Chunked Dataset")
    print("="*60)
    print(f"Run ID: {run_id}")
    print(f"K (Prefix length):  {K}")
    print(f"L (Rollout length): {L}")
    print(f"Data: {DATA_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*60)
    
    # 出力ディレクトリ作成
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # デバイス
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # データセット読み込み
    print("=== Loading Data ===")
    split_data = np.load(SPLIT_INDICES_PATH)
    
    print("\n=== Creating Datasets ===")
    train_dataset = ChunkedDataset(split_data, split='train')
    val_dataset = ChunkedDataset(split_data, split='val')  

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0,  # データロード高速化
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}\n")
    
    # モデル初期化
    print("=== Initializing Model ===")
    from DKP_RF import KoopmanRoutesFormer
    
    model = KoopmanRoutesFormer(
        vocab_size=VOCAB_SIZE,
        token_emb_dim=TOKEN_EMB_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        z_dim=Z_DIM,
        pad_token_id=PAD_TOKEN_ID,
        base_N=BASE_N,
        num_agents=1,
        agent_emb_dim=AGENT_EMB_DIM,
        max_stay_count=MAX_STAY_COUNT,
        stay_emb_dim=STAY_EMB_DIM,
        holiday_emb_dim=HOLIDAY_EMB_DIM,
        time_zone_emb_dim=TIME_ZONE_EMB_DIM,
        event_emb_dim=EVENT_EMB_DIM,
        use_aux_loss=True,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}\n")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    
    # History（従来コードを踏襲）
    history = {
        "train_loss": [], "val_loss": [],
        "train_ce": [], "val_ce": [],
        "train_geo": [], "val_geo": [],
        "train_count": [], "val_count": [],
        "train_mode": [], "val_mode": [],
        "train_lyap": [], "val_lyap": [],
    }
    
    # 学習ループ
    print("=== Starting Training ===\n")
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        
        # History更新
        history["train_loss"].append(train_metrics["loss"])
        history["train_ce"].append(train_metrics["ce"])
        history["train_geo"].append(train_metrics["geo"])
        history["train_count"].append(train_metrics["count"])
        history["train_mode"].append(train_metrics["mode"])
        history["train_lyap"].append(train_metrics["lyap"])
        
        history["val_loss"].append(val_metrics["loss"])
        history["val_ce"].append(val_metrics["ce"])
        history["val_geo"].append(val_metrics["geo"])
        history["val_count"].append(val_metrics["count"])
        history["val_mode"].append(val_metrics["mode"])
        history["val_lyap"].append(val_metrics["lyap"])
        
        # ログ出力（従来コードを踏襲）
        print(f"Epoch {epoch+1}: Train Loss = {train_metrics['loss']:.4f} | Val Loss = {val_metrics['loss']:.4f}")
        
        # Best model保存
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            
            save_data = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {
                    "K": K,
                    "L": L,
                    "vocab_size": VOCAB_SIZE,
                    "token_emb_dim": TOKEN_EMB_DIM,
                    "d_model": D_MODEL,
                    "nhead": NHEAD,
                    "num_layers": NUM_LAYERS,
                    "d_ff": D_FF,
                    "z_dim": Z_DIM,
                    "pad_token_id": PAD_TOKEN_ID,
                    "base_N": BASE_N,
                    "use_aux_loss": True,
                    "lambda_ce": LAMBDA_CE,
                    "lambda_geo": LAMBDA_GEO,
                    "lambda_count": LAMBDA_COUNT,
                    "lambda_mode": LAMBDA_MODE,
                    "lambda_lyapunov": LAMBDA_LYAPUNOV,
                    "lyap_eps": LYAP_EPS,
                },
                "history": history,
                "epoch": epoch,
            }
            
            checkpoint_path = output_dir / f'model_weights_{run_id}.pth'
            torch.save(save_data, checkpoint_path)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {checkpoint_path}")
    print("="*60)
    
    # =========================================================
    # グラフ描画（従来コードを踏襲：2×3レイアウト）
    # =========================================================
    
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
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True)
        
        # 2. CE Loss
        ax = axes[0, 1]
        ax.plot(epochs_range, history["train_ce"], label='Train', marker='.', color='orange')
        ax.plot(epochs_range, history["val_ce"], label='Val', marker='.', color='red')
        ax.set_title('Next Token Prediction (CE Loss)')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True)
        
        # 3. Geo Loss
        ax = axes[0, 2]
        ax.plot(epochs_range, history["train_geo"], label='Train', marker='.', color='green')
        ax.plot(epochs_range, history["val_geo"], label='Val', marker='.', color='lime')
        ax.set_title('Geo Distance Loss')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True)
        
        # 4. Lyapunov Loss
        ax = axes[1, 0]
        ax.plot(epochs_range, history["train_lyap"], label='Train', marker='.', color='purple')
        ax.plot(epochs_range, history["val_lyap"], label='Val', marker='.', color='magenta')
        ax.set_title('Lyapunov Loss')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True)
        
        # 5. Count Loss
        ax = axes[1, 1]
        ax.plot(epochs_range, history["train_count"], label='Train', marker='.', color='brown')
        ax.plot(epochs_range, history["val_count"], label='Val', marker='.', color='pink')
        ax.set_title('Stay Count Reconstruction')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True)
        
        # 6. Mode Loss
        ax = axes[1, 2]
        ax.plot(epochs_range, history["train_mode"], label='Train', marker='.', color='olive')
        ax.plot(epochs_range, history["val_mode"], label='Val', marker='.', color='yellowgreen')
        ax.set_title('Mode Reconstruction')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        graph_filename = output_dir / f"loss_graph_detailed_{run_id}.png"
        plt.savefig(graph_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Detailed loss graph saved at: {graph_filename}")
    
    except Exception as e:
        print(f"Failed to plot detailed loss graph: {e}")
    
    print("\n=== Training Complete ===")

if __name__ == "__main__":
    main()