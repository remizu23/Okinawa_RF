import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from captum.attr import GradientShap

# 自作モジュール
try:
    from KP_RF import KoopmanRoutesFormer
    from network import Network, expand_adjacency_matrix
    from tokenization import Tokenization
except ImportError as e:
    raise ImportError(f"Module import failed: {e}. Make sure KP_RF.py, network.py, tokenization.py are in the same directory.")

# =========================================================
# 1. 設定 & パス
# =========================================================
# ★解析したいモデルのパス
MODEL_PATH = '/home/mizutani/projects/RF/runs/20260105_202734/model_weights_20260105_202734.pth'

# ★データのパス
DATA_PATH = '/home/mizutani/projects/RF/data/input_e.npz'
ADJ_PATH = '/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt'

# 出力先
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"/home/mizutani/projects/RF/runs/SHAP_{run_id}"
os.makedirs(out_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =========================================================
# 2. データ準備
# =========================================================
print("Loading data...")
trip_arrz = np.load(DATA_PATH)
trip_arr = trip_arrz['route_arr']

if 'agent_ids' in trip_arrz:
    agent_ids_arr = trip_arrz['agent_ids']
else:
    print("Warning: 'agent_ids' not found. Using dummy IDs.")
    agent_ids_arr = np.zeros(len(trip_arr), dtype=int)

# Network構築
if not os.path.exists(ADJ_PATH):
    raise FileNotFoundError(f"{ADJ_PATH} not found.")
    
adj_matrix = torch.load(ADJ_PATH, weights_only=True)
expanded_adj = expand_adjacency_matrix(adj_matrix)
dummy_feature = torch.zeros((len(adj_matrix)*2, 1)) # ダミー特徴量
network = Network(expanded_adj, dummy_feature)
tokenizer = Tokenization(network)

# Tensor化
route_pt = torch.from_numpy(trip_arr).long()
agent_pt = torch.from_numpy(agent_ids_arr).long()

# データセット定義
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, routes, agents):
        self.routes = routes
        self.agents = agents
    def __len__(self):
        return len(self.routes)
    def __getitem__(self, idx):
        return self.routes[idx], self.agents[idx]

full_dataset = SimpleDataset(route_pt, agent_pt)

# SHAP計算用サブセット (先頭50件)
num_shap_samples = 50
shap_indices = range(num_shap_samples)
shap_dataset = Subset(full_dataset, shap_indices)
shap_loader = DataLoader(shap_dataset, batch_size=10, shuffle=False) # 小さいバッチで

# 背景（ベースライン）用データ (ランダムに100件)
bg_indices = np.random.choice(len(full_dataset), 100, replace=False)
bg_dataset = Subset(full_dataset, bg_indices)
bg_loader = DataLoader(bg_dataset, batch_size=100, shuffle=False)

# =========================================================
# 3. モデルロード & 埋め込み取得用ラッパー
# =========================================================
print(f"Loading model from {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location=device)
config = checkpoint['config']

# モデル初期化
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
    max_stay_count=config.get('max_stay_count', 100),
    stay_emb_dim=config.get('stay_emb_dim', 16)
)
# strict=Falseでロード
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.to(device)
model.eval()

# --- SHAP用の特別なForward関数 ---
def forward_emb_input(token_emb, stay_emb, agent_emb, target_indices=None):
    """
    埋め込みベクトルを入力として受け取り、ターゲットの対数確率を返す。
    Captumが内部でバッチサイズを拡張するため、target_indicesのサイズ合わせが必要。
    """
    # バッチサイズ拡張処理
    if target_indices is not None:
        batch_size_input = token_emb.size(0)
        batch_size_target = target_indices.size(0)
        
        # 入力がターゲットより多い場合 (n_samples倍されている場合)
        if batch_size_input != batch_size_target:
            if batch_size_input % batch_size_target == 0:
                n_repeats = batch_size_input // batch_size_target
                # target_indices をリピートしてサイズを合わせる
                target_indices = target_indices.repeat(n_repeats, 1)
            else:
                raise RuntimeError(f"Batch size mismatch: input {batch_size_input}, target {batch_size_target}")

    # 1. 結合
    u_all = torch.cat([token_emb, stay_emb, agent_emb], dim=-1)
    
    # 2. モデル内部処理
    x = model.input_proj(u_all)
    x = model.pos_encoder(x)
    
    seq_len = x.size(1)
    causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
    
    h = model.transformer_block(src=x, mask=causal_mask)
    z_hat = model.to_z(h)
    logits = model.to_logits(z_hat) # [B, T, vocab]
    
    # 3. ターゲット確率の取得
    if target_indices is not None:
        log_probs = F.log_softmax(logits, dim=-1)
        # gatherで正解ラベルの確率だけを取り出す
        target_log_probs = log_probs.gather(2, target_indices.unsqueeze(-1)).squeeze(-1)
        return target_log_probs.sum(dim=-1)
    else:
        return logits.sum()

# =========================================================
# 4. ベースライン（背景データ）の作成
# =========================================================
print("Calculating baselines...")
bg_token_embs = []
bg_stay_embs = []
bg_agent_embs = []

with torch.no_grad():
    for routes, agents in bg_loader:
        routes = routes.to(device)
        agents = agents.to(device)
        
        # Tokenize
        input_tokens = tokenizer.tokenization(routes, mode="simple").long()
        stay_counts = tokenizer.calculate_stay_counts(input_tokens)
        
        # Embedding
        t_emb = model.token_embedding(input_tokens)
        s_emb = model.stay_embedding(stay_counts)
        a_emb = model.agent_embedding(agents).unsqueeze(1).expand(-1, input_tokens.size(1), -1)
        
        bg_token_embs.append(t_emb)
        bg_stay_embs.append(s_emb)
        bg_agent_embs.append(a_emb)

# 平均をとる
baseline_token = torch.cat(bg_token_embs, dim=0).mean(dim=0, keepdim=True) # [1, T, dim]
baseline_stay  = torch.cat(bg_stay_embs, dim=0).mean(dim=0, keepdim=True)
baseline_agent = torch.cat(bg_agent_embs, dim=0).mean(dim=0, keepdim=True)

print(f"Baseline shapes: {baseline_token.shape}")

# =========================================================
# 5. GradientSHAP 計算実行
# =========================================================
gradient_shap = GradientShap(forward_emb_input)

shap_values_token = []
shap_values_stay = []
shap_values_agent = []

# ★追加関数: ベースラインの長さを入力に合わせる（足りなければゼロ埋め）
def adjust_baseline_length(baseline, target_len):
    """
    baseline: [1, seq_len, dim]
    target_len: int
    """
    curr_len = baseline.size(1)
    if curr_len >= target_len:
        # 長すぎる場合は切る
        return baseline[:, :target_len, :]
    else:
        # 短すぎる場合はゼロパディング
        diff = target_len - curr_len
        padding = torch.zeros(baseline.size(0), diff, baseline.size(2)).to(baseline.device)
        return torch.cat([baseline, padding], dim=1)

print(f"Computing SHAP values for {num_shap_samples} samples...")

for batch_idx, (routes, agents) in enumerate(shap_loader):
    routes = routes.to(device)
    agents = agents.to(device)
    
    # 入力準備
    input_tokens = tokenizer.tokenization(routes, mode="simple").long()
    target_tokens = tokenizer.tokenization(routes, mode="next").long()
    stay_counts = tokenizer.calculate_stay_counts(input_tokens)
    
    # .clone().detach() でLeaf化
    input_emb_token = model.token_embedding(input_tokens).clone().detach()
    input_emb_stay  = model.stay_embedding(stay_counts).clone().detach()
    _agent_emb = model.agent_embedding(agents).unsqueeze(1).expand(-1, input_tokens.size(1), -1)
    input_emb_agent = _agent_emb.clone().detach()
    
    input_emb_token.requires_grad = True
    input_emb_stay.requires_grad = True
    input_emb_agent.requires_grad = True

    # ベースライン調整 (サイズ合わせ)
    curr_batch_size = routes.size(0)
    # ★修正: 実際のトークン化後のシーケンス長を使用
    seq_len = input_tokens.size(1)  # routes.size(1)ではなく、トークン化後の長さを使用
    
    # ★修正: adjust_baseline_length を使用して長さを確実に一致させる
    bl_t = adjust_baseline_length(baseline_token, seq_len).repeat(curr_batch_size, 1, 1)
    bl_s = adjust_baseline_length(baseline_stay, seq_len).repeat(curr_batch_size, 1, 1)
    bl_a = adjust_baseline_length(baseline_agent, seq_len).repeat(curr_batch_size, 1, 1)

    # SHAP計算
    attributions = gradient_shap.attribute(
        inputs=(input_emb_token, input_emb_stay, input_emb_agent),
        baselines=(bl_t, bl_s, bl_a),
        additional_forward_args=(target_tokens,), 
        n_samples=50 
    )
    
    # 結果格納
    shap_values_token.append(attributions[0].detach().cpu().numpy())
    shap_values_stay.append(attributions[1].detach().cpu().numpy())
    shap_values_agent.append(attributions[2].detach().cpu().numpy())
    
    print(f"Batch {batch_idx+1} done.")

# 結合
shap_token_all = np.concatenate(shap_values_token, axis=0) # [N, T, dim]
shap_stay_all  = np.concatenate(shap_values_stay, axis=0)
shap_agent_all = np.concatenate(shap_values_agent, axis=0)

# =========================================================
# 6. 結果の保存と可視化
# =========================================================
# 次元ごとの値を合計
attr_token_score = np.sum(shap_token_all, axis=-1) # [N, T]
attr_stay_score  = np.sum(shap_stay_all, axis=-1)
attr_agent_score = np.sum(shap_agent_all, axis=-1)

# 保存
np.savez(
    os.path.join(out_dir, "shap_values.npz"),
    token=attr_token_score,
    stay=attr_stay_score,
    agent=attr_agent_score
)
print(f"Saved SHAP values to {out_dir}/shap_values.npz")

# --- 可視化 ---
plt.figure(figsize=(12, 6))
# 絶対値の平均
mean_token = np.mean(np.abs(attr_token_score), axis=0)
mean_stay  = np.mean(np.abs(attr_stay_score), axis=0)
mean_agent = np.mean(np.abs(attr_agent_score), axis=0)

steps = range(len(mean_token))
plt.plot(steps, mean_token, label='Node ID Importance', marker='o')
plt.plot(steps, mean_stay, label='Stay Count Importance', marker='s')
plt.plot(steps, mean_agent, label='Agent ID Importance', marker='^')

plt.title("Average Feature Importance (SHAP) over Time Steps")
plt.xlabel("Time Step")
plt.ylabel("Importance (Mean Absolute SHAP)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(out_dir, "shap_importance_plot.png"))
plt.close()

print("Done.")