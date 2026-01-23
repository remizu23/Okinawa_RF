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
import networkx as nx

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
    "epochs": 200, 
    "batch_size": 32, # 系列が長くなるので、メモリ溢れするならここを減らす
    "d_ie": 64,
    "head_num": 4, 
    "d_ff": 128, 
    "B_de": 3,
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
trip_arrz = np.load('/home/mizutani/projects/RF/data/input_real_m4.npz') ##インプットを変えたら変える！

adj_matrix = torch.load('/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt', weights_only=True)


def compute_shortest_path_distance_matrix(adj: torch.Tensor, directed: bool = False) -> torch.Tensor:
    """Compute all-pairs shortest-path hop distances from an adjacency matrix.

    Parameters
    ----------
    adj:
        [N,N] adjacency matrix (0/1 or weights). Non-zero entries are treated as edges.
    directed:
        If True, treat edges as directed; else undirected.

    Returns
    -------
    dist:
        [N,N] LongTensor with hop distances. Unreachable pairs get a large value (N+1).
    """
    if not isinstance(adj, torch.Tensor):
        adj = torch.tensor(adj)
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



# --- Delta-distance-to-plaza: precompute base shortest-path distances ---
# NOTE: adj_matrix is assumed to be the *base* adjacency (e.g., 19x19). If your saved
# adjacency is already expanded (move+stay), compute distances on the first half only.
dist_is_directed = False
if adj_matrix.shape[0] == 38:
    base_N = 19
    base_adj = adj_matrix[:base_N, :base_N]
else:
    base_adj = adj_matrix
    base_N = int(base_adj.shape[0])

dist_mat_base = compute_shortest_path_distance_matrix(base_adj, directed=dist_is_directed)

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
    # --- Δ距離(広場)バイアス用 ---
    dist_mat_base=dist_mat_base,
    base_N=base_N,
    delta_bias_move_only=True,
    dist_is_directed=dist_is_directed,
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
#  学習ループ (Training Phase)
# =========================================================
print(f"Training Start... (Koopman Loss: {wandb.config.use_koopman_loss})")

# ★変更: 各ロスの履歴を保持する辞書を拡張
history = {
    "train_loss": [], "val_loss": [],
    "train_ce": [], "val_ce": [],         # 次トークン予測
    "train_dyn": [], "val_dyn": [],       # 多ステップ予測
    "train_linear": [], "val_linear": [], # 単ステップ線形 (Loss K)
    "train_count": [], "val_count": [],   # 滞在数予測
    "train_mode": [], "val_mode": []    # モード予測
}


for epoch in range(wandb.config.epochs):
    # --- Training ---
    model.train()
    epoch_metrics = {
        "loss": 0.0, "ce": 0.0, "dyn": 0.0, "linear": 0.0, "count": 0.0, "mode":0.0
    }
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    
    for batch in pbar:
        route_batch, time_batch, agent_batch = batch 
        route_batch = route_batch.to(device)
        agent_batch = agent_batch.to(device)
        time_batch = time_batch.to(device)
        
        tokenizer = Tokenization(network)
        input_tokens = tokenizer.tokenization(route_batch, mode="simple").long().to(device)
        target_tokens = tokenizer.tokenization(route_batch, mode="next").long().to(device)
        
        stay_counts = tokenizer.calculate_stay_counts(input_tokens)
        
        # === ★デバッグ用コード追加開始 ===
        # モデルが許容する最大値
        max_vocab = model.token_embedding.num_embeddings
        max_stay = model.stay_embedding.num_embeddings
        max_agent = model.agent_embedding.num_embeddings

        # 入力データの最大値
        curr_token_max = input_tokens.max().item()
        curr_stay_max = stay_counts.max().item()
        curr_agent_max = agent_batch.max().item()

        # チェック
        if curr_token_max >= max_vocab:
            print(f"【エラー原因】トークンID超過: 入力{curr_token_max} >= 許容{max_vocab}")
        if curr_stay_max >= max_stay:
            print(f"【エラー原因】滞在カウント超過: 入力{curr_stay_max} >= 許容{max_stay}")
        if curr_agent_max >= max_agent:
            print(f"【エラー原因】Agent ID超過: 入力{curr_agent_max} >= 許容{max_agent}")
        # === ★デバッグ用コード追加終了 ===

        # ★修正: 4つの戻り値を受け取る (u_allが必要)
        logits, z_hat, z_pred_next, u_all = model(input_tokens, stay_counts, agent_batch, time_tensor=time_batch)        
        
        # 1. CE Loss (全員共通)
        loss_ce = ce_loss_fn(logits.view(-1, vocab_size), target_tokens.view(-1))
        
        # 初期化
        loss_total = loss_ce
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
            # ★★★ ここを変えたらvalidationの方も変える！！★★★
            loss_total = loss_ce + \
                        wandb.config.koopman_alpha * loss_k + \
                        0.01 * loss_count + \
                        1 * loss_dyn + \
                        0.1 * loss_mode

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        
        # ★集計 (重み付け前の生の値を保存すると解釈しやすい)
        epoch_metrics["loss"] += loss_total.item()
        epoch_metrics["ce"] += loss_ce.item()
        epoch_metrics["dyn"] += loss_dyn.item() if wandb.config.use_koopman_loss else 0
        epoch_metrics["linear"] += loss_k.item() if wandb.config.use_koopman_loss else 0
        epoch_metrics["count"] += loss_count.item() if wandb.config.use_koopman_loss else 0
        epoch_metrics["mode"] += loss_mode.item() if wandb.config.use_koopman_loss else 0

        pbar.set_postfix(loss=loss_total.item())

    # 平均計算 & 履歴保存
    n_batches = len(train_loader)
    history["train_loss"].append(epoch_metrics["loss"] / n_batches)
    history["train_ce"].append(epoch_metrics["ce"] / n_batches)
    history["train_dyn"].append(epoch_metrics["dyn"] / n_batches)
    history["train_linear"].append(epoch_metrics["linear"] / n_batches)
    history["train_count"].append(epoch_metrics["count"] / n_batches)
    history["train_mode"].append(epoch_metrics["mode"] / n_batches)


# --- Validation (ロジックはTrainと全く同じ) ---
    model.eval()
    epoch_metrics_val = {
        "loss": 0.0, "ce": 0.0, "dyn": 0.0, "linear": 0.0, "count": 0.0, "mode": 0.0
    }    

    with torch.no_grad():
        for route_batch, time_batch, agent_batch in val_loader:
            route_batch = route_batch.to(device)
            agent_batch = agent_batch.to(device)
            time_batch = time_batch.to(device)
            
            tokenizer = Tokenization(network)
            input_tokens = tokenizer.tokenization(route_batch, mode="simple").long().to(device)
            target_tokens = tokenizer.tokenization(route_batch, mode="next").long().to(device)
            stay_counts = tokenizer.calculate_stay_counts(input_tokens)

            logits, z_hat, z_pred_next, u_all = model(input_tokens, stay_counts, agent_batch, time_tensor=time_batch)

            loss_ce = ce_loss_fn(logits.view(-1, vocab_size), target_tokens.view(-1))
            loss_total = loss_ce 
            
            val_dyn = torch.tensor(0.0)
            val_k = torch.tensor(0.0)
            val_count = torch.tensor(0.0)

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

                loss_total = loss_ce + \
                            wandb.config.koopman_alpha * loss_k + \
                            0.01 * loss_count + \
                            1 * loss_dyn + \
                            0.1 * loss_mode

            epoch_metrics_val["loss"] += loss_total.item()
            epoch_metrics_val["ce"] += loss_ce.item()
            epoch_metrics_val["dyn"] += loss_dyn.item() if wandb.config.use_koopman_loss else 0
            epoch_metrics_val["linear"] += loss_k.item() if wandb.config.use_koopman_loss else 0
            epoch_metrics_val["count"] += loss_count.item() if wandb.config.use_koopman_loss else 0
            epoch_metrics_val["mode"] += loss_mode.item() if wandb.config.use_koopman_loss else 0

        # 平均計算 & 履歴保存
        n_val = len(val_loader)
        history["val_loss"].append(epoch_metrics_val["loss"] / n_val)
        history["val_ce"].append(epoch_metrics_val["ce"] / n_val)
        history["val_dyn"].append(epoch_metrics_val["dyn"] / n_val)
        history["val_linear"].append(epoch_metrics_val["linear"] / n_val)
        history["val_count"].append(epoch_metrics_val["count"] / n_val)
        history["val_mode"].append(epoch_metrics_val["mode"] / n_val)

        print(f"Epoch {epoch+1}: Train Loss = {history['train_loss'][-1]:.4f} | Val Loss = {history['val_loss'][-1]:.4f}")
        

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

# --- グラフ描画 (詳細版) ---
try:
    epochs_range = range(1, len(history["train_loss"]) + 1)
    
    # 2x3 のサブプロットを作成
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
    
    # 2. Cross Entropy (Next Token) Loss
    ax = axes[0, 1]
    ax.plot(epochs_range, history["train_ce"], label='Train', marker='.', color='orange')
    ax.plot(epochs_range, history["val_ce"], label='Val', marker='.', color='red')
    ax.set_title('Next Token Prediction (CE Loss)')
    ax.legend()
    ax.grid(True)
    
    # 3. Dynamics Loss (Multi-step)
    ax = axes[0, 2]
    ax.plot(epochs_range, history["train_dyn"], label='Train', marker='.', color='green')
    ax.plot(epochs_range, history["val_dyn"], label='Val', marker='.', color='lime')
    ax.set_title('Multi-step Dynamics (MSE)')
    ax.legend()
    ax.grid(True)
    
    # 4. Linear Loss (Single-step)
    ax = axes[1, 0]
    ax.plot(epochs_range, history["train_linear"], label='Train', marker='.', color='purple')
    ax.plot(epochs_range, history["val_linear"], label='Val', marker='.', color='magenta')
    ax.set_title('Single-step Linear (MSE)')
    ax.legend()
    ax.grid(True)
    
    # 5. Count Reconstruction Loss
    ax = axes[1, 1]
    ax.plot(epochs_range, history["train_count"], label='Train', marker='.', color='brown')
    ax.plot(epochs_range, history["val_count"], label='Val', marker='.', color='pink')
    ax.set_title('Stay Count Reconstruction (MSE)')
    ax.legend()
    ax.grid(True)
    
    # 6. Mode reconstruction Loss
    ax = axes[1, 2]
    # ax.axis('off') # 何も表示しない
    ax.plot(epochs_range, history["train_mode"], label='Train', marker='.', color='olive')
    ax.plot(epochs_range, history["val_mode"], label='Val', marker='.', color='yellowgreen')
    ax.set_title('Mode Reconstruction (MSE)')
    ax.legend()
    ax.grid(True)


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # タイトル分のスペースを空ける
    
    graph_filename = stamp(f"loss_graph_detailed_{run_id}.png")
    plt.savefig(graph_filename)
    plt.close()
    print(f"Detailed loss graph saved at: {graph_filename}")

except Exception as e:
    print(f"Failed to plot detailed loss graph: {e}")