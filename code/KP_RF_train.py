import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
import os
from datetime import datetime
from KP_RF import KoopmanRoutesFormer
# 自作モジュール
from network import Network, expand_adjacency_matrix
from tokenization import Tokenization
import matplotlib.pyplot as plt


# --- 設定周り ---
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
    
    # Ablation
    "use_koopman_loss": False,  
    "koopman_alpha": 0.1,
    
    # ★ Rollout設定
    "rollout_steps": 5,        # 何ステップ先まで自己回帰予測するか
    "rollout_start_prob": 0.5  # 各バッチでRollout lossを計算する確率(計算コスト削減用)
})()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 保存ディレクトリ ---
out_dir = f"/home/mizutani/projects/RF/runs/{run_id}"
os.makedirs(out_dir, exist_ok=True)
def stamp(name): return os.path.join(out_dir, name)

# --- データの準備 ---
# ※パスは適宜環境に合わせてください
trip_arrz = np.load('/home/mizutani/projects/RF/data/input_real_m5long.npz') 
adj_matrix = torch.load('/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt', weights_only=True)

# 距離行列計算 (省略可能だが元のコードに従う)
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

dist_mat_base = compute_shortest_path_distance_matrix(base_adj, directed=False)
expanded_adj = expand_adjacency_matrix(adj_matrix)
dummy_node_features = torch.zeros((len(adj_matrix), 1))
expanded_features = torch.cat([dummy_node_features, dummy_node_features], dim=0)
network = Network(expanded_adj, expanded_features)

trip_arr = trip_arrz['route_arr']
time_arr = trip_arrz['time_arr']
agent_ids_arr = trip_arrz['agent_ids'] if 'agent_ids' in trip_arrz else np.zeros(len(trip_arr), dtype=int)

# 既存のコンテキスト配列もロード(互換性のため)
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

# --- Dataset & DataLoader ---
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, r, t, a, h, tz, e):
        self.r = r; self.t = t; self.a = a
        self.h = h; self.tz = tz; self.e = e
    def __len__(self): return len(self.r)
    def __getitem__(self, idx):
        return self.r[idx], self.t[idx], self.a[idx], self.h[idx], self.tz[idx], self.e[idx]

dataset = MyDataset(route_pt, time_pt, agent_pt, holiday_pt, timezone_pt, event_pt)

num_samples = len(dataset)
train_size = int(num_samples * 0.8)
val_size = num_samples - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

def collate_fn_pad(batch):
    routes, times, agents, holidays, timezones, events = zip(*batch)
    trimmed_data = []
    for i in range(len(routes)):
        r_np = routes[i].cpu().numpy()
        pad_indices = np.where(r_np == 38)[0]
        real_len = pad_indices[0] if len(pad_indices) > 0 else len(r_np)
        trimmed_data.append({
            'r': routes[i][:real_len],
            'h': holidays[i][:real_len],
            'tz': timezones[i][:real_len],
            'e': events[i][:real_len]
        })
    lengths = [len(x['r']) for x in trimmed_data]
    max_len = max(lengths) if lengths else 0
    
    padded_routes = torch.full((len(batch), max_len), 38, dtype=torch.long)
    padded_holidays = torch.zeros((len(batch), max_len), dtype=torch.long)
    padded_timezones = torch.zeros((len(batch), max_len), dtype=torch.long)
    padded_events = torch.zeros((len(batch), max_len), dtype=torch.long)

    for i, item in enumerate(trimmed_data):
        L = len(item['r'])
        padded_routes[i, :L] = item['r']
        padded_holidays[i, :L] = item['h']
        padded_timezones[i, :L] = item['tz']
        padded_events[i, :L] = item['e']

    return (padded_routes, torch.tensor(times), torch.tensor(agents), 
            padded_holidays, padded_timezones, padded_events)

train_loader = DataLoader(train_data, batch_size=wandb.config.batch_size, shuffle=True, collate_fn=collate_fn_pad, drop_last=True)
val_loader = DataLoader(val_data, batch_size=wandb.config.batch_size, shuffle=False, collate_fn=collate_fn_pad, drop_last=True)

# --- ★ Context Manager (New) ---
class ContextManager:
    def __init__(self, base_N=19):
        self.base_N = base_N
        # 定数定義
        self.HOLIDAYS = [20240928, 20240929, 20251122, 20251123]
        self.NIGHT_START = 19
        self.NIGHT_END = 2 # 翌2時まで
        
        # (Date, StartHour, EndHour, TargetNodes)
        # ※ 指示「2024/9/29 9:00~16:00 (14番ノード)」等
        # StartTimeがこの範囲に入っていれば、対象ノードのイベントフラグを立てる
        self.EVENTS = [
            (20240929, 9, 16, [14]),
            (20251122, 10, 19, [2, 11]),
            (20251123, 10, 16, [2])
        ]

    def get_trip_context(self, start_times):
        """
        バッチ内の各トリップの開始時刻から、トリップ全体で固定の属性を判定
        start_times: [Batch] tensor (YYYYMMDDHHMM)
        Returns: 
           is_holiday: [Batch] (0 or 1)
           is_night:   [Batch] (0 or 1)
           active_event_nodes: list of lists (各バッチで有効なイベントノードID)
        """
        batch_size = start_times.size(0)
        is_holiday = torch.zeros(batch_size, dtype=torch.long)
        is_night = torch.zeros(batch_size, dtype=torch.long)
        # イベントはノード依存なので、「このトリップで有効なイベント定義」のリストを返す
        trip_event_defs = [[] for _ in range(batch_size)] 

        st_cpu = start_times.cpu().numpy()
        for i, t_val in enumerate(st_cpu):
            date_val = t_val // 10000
            hour_val = (t_val // 100) % 100
            
            # Holiday
            if date_val in self.HOLIDAYS:
                is_holiday[i] = 1
                
            # Night (9:00~19:00 is Day, 19:00~02:00 is Night)
            # 指示: 「19時を跨いだから変える、はしなくていい」= 開始時刻基準
            if hour_val >= self.NIGHT_START or hour_val < self.NIGHT_END:
                is_night[i] = 1
            else:
                is_night[i] = 0
                
            # Event
            for (ev_date, ev_start, ev_end, ev_nodes) in self.EVENTS:
                if date_val == ev_date:
                    if ev_start <= hour_val < ev_end:
                        trip_event_defs[i].extend(ev_nodes)
                        
        return is_holiday.to(device), is_night.to(device), trip_event_defs

    def get_event_flag(self, token_ids, trip_event_defs):
        """
        現在のトークン(ノード)がイベント対象か判定
        token_ids: [Batch] tensor
        trip_event_defs: list of lists (get_trip_contextの戻り値)
        Returns: [Batch] tensor (0 or 1)
        """
        batch_size = token_ids.size(0)
        event_flags = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        tokens_cpu = token_ids.cpu().numpy()
        
        for i in range(batch_size):
            # トークンIDからノードIDへの変換 (Move: 0~18, Stay: 19~37)
            tk = tokens_cpu[i]
            node_id = -1
            if 0 <= tk < self.base_N:
                node_id = tk
            elif self.base_N <= tk < self.base_N * 2:
                node_id = tk - self.base_N
            
            # ノードIDがそのトリップの有効イベントノードに含まれるか
            if node_id != -1 and node_id in trip_event_defs[i]:
                event_flags[i] = 1
                
        return event_flags

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
    pad_token_id=network.N, # 拡張後のものを定義済みなので38
    dist_mat_base=dist_mat_base,
    base_N=base_N,
    num_agents=num_agents,
    agent_emb_dim=wandb.config.agent_emb_dim,
    max_stay_count=wandb.config.max_stay_count,
    stay_emb_dim=wandb.config.stay_emb_dim,
    holiday_emb_dim=wandb.config.holiday_emb_dim,
    time_zone_emb_dim=wandb.config.time_zone_emb_dim,
    event_emb_dim=wandb.config.event_emb_dim
).to(device)

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

optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
ce_loss_fn = nn.CrossEntropyLoss(ignore_index=network.N) # Pad=38．拡張後のnetwork.N

ctx_manager = ContextManager(base_N=base_N)

# --- Training Loop ---

# ★変更: 各ロスの履歴を保持する辞書を拡張
history = {
    "train_loss": [], "val_loss": [],
    "train_ce": [], "val_ce": [],         # 次トークン予測
    "train_dyn": [], "val_dyn": [],       # 多ステップ予測
    "train_linear": [], "val_linear": [], # 単ステップ線形 (Loss K)
    "train_count": [], "val_count": [],   # 滞在数予測
    "train_mode": [], "val_mode": [],    # モード予測
    "train_rollout": [], "val_rollout": [],    # モード予測
    "train_geo": [], "val_geo": []    # モード予測
}


for epoch in range(wandb.config.epochs):
    # --- Training ---
    model.train()
    epoch_metrics = {
        "loss": 0.0, "ce": 0.0, "dyn": 0.0, "linear": 0.0, "count": 0.0, "mode":0.0, "rollout":0.0, "geo":0.0
    }
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    
    for batch in pbar:
        r_b, t_b, a_b, h_b, tz_b, e_b = batch
        r_b, t_b, a_b = r_b.to(device), t_b.to(device), a_b.to(device)
        h_b, tz_b, e_b = h_b.to(device), tz_b.to(device), e_b.to(device)
        
        # 1. 前処理
        tokenizer = Tokenization(network)
        input_tokens = tokenizer.tokenization(r_b, mode="simple").long().to(device) # [B, T] (starts with <b>)
        target_tokens = tokenizer.tokenization(r_b, mode="next").long().to(device)  # [B, T] (ends with <e>)
        stay_counts = tokenizer.calculate_stay_counts(input_tokens)

        # Context alignment
        B_size, T = input_tokens.shape
        def align_ctx(ctx, target_len):
            out = torch.zeros((B_size, target_len), dtype=torch.long, device=device)
            copy_len = min(ctx.shape[1], target_len - 1)
            if copy_len > 0: out[:, 1 : 1 + copy_len] = ctx[:, :copy_len]
            return out
        
        h_in, tz_in, e_in = align_ctx(h_b, T), align_ctx(tz_b, T), align_ctx(e_b, T)

        # 2. Forward (Transformer + Batch Dynamics)
        logits, z_hat, z_pred_next, u_all = model(
            input_tokens, stay_counts, a_b, h_in, tz_in, e_in
        )
        
        loss_ce = ce_loss_fn(logits.view(-1, vocab_size), target_tokens.view(-1))
        # (★追加) 地理的損失の計算
        # 係数(geo_alpha)は 0.1 ~ 1.0 程度で調整 (距離の値がホップ数で 1~10 程度あるため、CEより大きくなりやすい)
        geo_alpha = 0.1
        loss_geo = model.calc_geo_loss(logits, target_tokens)

        # 初期化
        loss_total = loss_ce
        loss_count = torch.tensor(0.0, device=device)
        loss_dyn = torch.tensor(0.0, device=device)
        loss_k = torch.tensor(0.0, device=device)
        loss_mode = torch.tensor(0.0, device=device)
        loss_rollout = torch.tensor(0.0, device=device)

        # ★ Koopmanモードの時だけ追加Lossを計算
        if wandb.config.use_koopman_loss and wandb.config.rollout_steps > 0:
            valid_mask = (input_tokens != network.N)

            # 2. Count Reconstruction (zの意味付け)
            # 滞在数は「現在の場所」の属性なので、現在の状態 z_hat から予測します (変更なし)
            pred_count = model.count_decoder(z_hat).squeeze(-1)
            loss_count = masked_mse_loss(pred_count, stay_counts.float(), valid_mask)

            # 3. Multi-step Dynamics Loss (再帰予測)
            K_steps = 5
            
            # 再帰予測の起点: Transformerが出力した z_hat
            # 比較対象: 未来の z_hat
            
            # z_hat は [B, T, D]
            # Kステップ先まで予測するため、末尾K個は起点にできない
            current_z = z_hat[:, :-K_steps, :]
            
            for k in range(K_steps):
                # 入力 u を取得 (t+k のタイミングのもの)
                end_idx = -K_steps + k if (-K_steps + k) != 0 else None
                u_curr_step = u_all[:, k : end_idx, :]
                
                # 線形遷移
                term_A = torch.einsum("ij,btj->bti", model.A, current_z)
                term_B = torch.einsum("ij,btj->bti", model.B, u_curr_step)
                pred_z_rec = term_A + term_B # recursive prediction
                
                # 正解データ: (k+1)ステップ先の z_hat
                start_true = k + 1
                end_true = -K_steps + k + 1 if (-K_steps + k + 1) != 0 else None
                true_z_next = z_hat[:, start_true : end_true, :]
                
                # マスク処理
                future_mask = valid_mask[:, start_true : end_true]
                
                decay = 0.8 ** k
                loss_dyn += masked_mse_loss(pred_z_rec, true_z_next, future_mask) * decay
                
                current_z = pred_z_rec

            # 4. Single-step Loss (Loss K)
            # z_pred_next (モデル出力) は「t+1 の予測」
            # z_hat (モデル出力) は「t の状態」
            # よって、z_pred_next[t] と z_hat[t+1] を比較します。
            
            # z_pred_next の最後(予測先が系列外)は比較対象がないので捨てる [:, :-1]
            z_pred_step = z_pred_next[:, :-1, :]
            
            # z_hat の最初(t=0)は予測対象ではないので捨てる [:, 1:]
            z_true_step = z_hat[:, 1:, :] 
            
            # マスクもずらす
            step_mask = valid_mask[:, 1:]
            
            loss_k = masked_mse_loss(z_pred_step, z_true_step, step_mask)
            
            # 5. モード損失
            target_modes = torch.full_like(target_tokens, -100)
            is_move = (target_tokens >= 0) & (target_tokens < network.N)
            target_modes[is_move] = 1
            STAY_OFFSET = network.N
            PAD_TOKEN = network.N * 2
            is_stay = (target_tokens >= STAY_OFFSET) & (target_tokens < PAD_TOKEN)
            target_modes[is_stay] = 0

            # ★変更: モード予測も「次の状態」である z_pred_next から行います
            # target_modes は「次のトークン」の属性なので、これで整合します
            pred_modes = model.mode_classifier(z_pred_next)

            loss_mode = nn.CrossEntropyLoss(ignore_index=-100)(
                pred_modes.view(-1, 2), 
                target_modes.view(-1)
            )

            # 3. ★ Multi-step Rollout Loss (Detached) ★
            # 予測開始地点を決める（系列全体からランダム、あるいは全て）
            # ここでは計算コスト抑制のため、系列の途中からKステップ予測を行う
            # T_seqは系列長。Kステップ先まで正解がある範囲でループ。
            
            K = wandb.config.rollout_steps
            # 予測可能な最大index: T - 1 (last input) -> predict T (last target)
            # z_pred_next[:, t] は input[t] (time t) から input[t+1] (time t+1) への予測
            # input_tokens[t+1] が正解トークン
            
            # Context判定用の静的情報を取得
            trip_hol, trip_night, trip_ev_defs = ctx_manager.get_trip_context(t_b)
            
            # ランダムな開始点を1つ選ぶ (またはストライド)
            # input_tokensの有効長を考慮すべきだが、簡単のためPadding以外で。
            valid_len = (input_tokens != 38).sum(dim=1)
            
            rollout_loss_accum = []
            
            # バッチ内の各サンプルについて処理（ベクトル化は難しいロジックを含むためループかマスク処理）
            # ※ 効率化のため、バッチ全体で「有効なステップ」を一斉に計算する
            
            start_t = np.random.randint(1, max(2, T - K - 1)) # ランダムスタート
            
            # 初期状態: Transformerによる予測値 (勾配は切る)
            z_curr = z_pred_next[:, start_t, :].detach() 
            
            # ロールアウト用の現在トークン (正解ではなく予測を使うのが純粋なRolloutだが、
            # Detached Rolloutでは「予測したトークン」を次ステップの入力にする)
            
            # 初期入力トークン（start_t+1 の予測）
            # ここは z_curr から予測する
            pred_logits_init = model.to_logits(z_curr)
            pred_token = torch.argmax(pred_logits_init, dim=-1) # [B]
            
            # 滞在カウント追跡用: start_t 時点の正解トークンを取得（遷移判定のため）
            prev_token = input_tokens[:, start_t] # [B]
            
            # start_t 時点の正解カウント（ここからインクリメントするか判定）
            prev_stay_count = stay_counts[:, start_t] # [B]

            for k in range(K):
                # ターゲット: start_t + 1 + k
                target_idx = start_t + 1 + k
                if target_idx >= T: break
                
                gt_token = target_tokens[:, target_idx] # 正解
                
                # Loss 1: 予測精度 (CrossEntropy)
                # マスク作成 (Paddingは無視)
                mask = (gt_token != 38)
                if mask.sum() > 0:
                    logits_k = model.to_logits(z_curr)
                    step_loss = ce_loss_fn(logits_k[mask], gt_token[mask])
                    rollout_loss_accum.append(step_loss)
                
                # --- 次ステップへの入力作成 (u_t+1) ---
                # 1. Token: 予測したトークン (pred_token) を使用
                #    ※ ここで pred_token は z_curr から argmax したもの
                #    ※ ループ初回は上で計算済み。2回目以降はループ末尾で更新。
                curr_token_in = pred_token 
                
                # 2. Stay Count Logic
                # "滞在ノード(19-37)かつ前回と同じトークンなら+1、それ以外は1"
                next_stay_count = torch.ones_like(curr_token_in) # Default 1
                
                is_stay_node = (curr_token_in >= base_N) & (curr_token_in < base_N * 2)
                is_same = (curr_token_in == prev_token)
                
                # 条件を満たす場所だけカウントアップ
                inc_mask = is_stay_node & is_same
                next_stay_count[inc_mask] = prev_stay_count[inc_mask] + 1
                
                # キャップ
                next_stay_count = torch.clamp(next_stay_count, max=wandb.config.max_stay_count)
                
                # 3. Context Logic
                # Holiday, Night: トリップ固定
                # Event: トリップ開始時刻が条件合致 & 現在ノードが対象
                curr_event_flag = ctx_manager.get_event_flag(curr_token_in, trip_ev_defs)
                
                # 入力埋め込み作成
                u_next = model.get_single_step_input(
                    token_id=curr_token_in,
                    stay_count=next_stay_count,
                    agent_id=a_b,
                    holiday=trip_hol,
                    timezone=trip_night,
                    event=curr_event_flag
                )
                
                # ダイナミクス遷移 (勾配はA, Bに流れるが、u_nextには流れない)
                z_next, logits_next = model.forward_step(z_curr, u_next)
                
                # 次ループへの更新
                z_curr = z_next
                prev_token = curr_token_in
                prev_stay_count = next_stay_count
                pred_token = torch.argmax(logits_next, dim=-1).detach() # 次の予測トークン

            if rollout_loss_accum:
                loss_rollout = torch.stack(rollout_loss_accum).mean()

            loss_total = loss_ce + \
                        wandb.config.koopman_alpha * loss_k + \
                        0.1 * loss_mode + \
                        0.01 * loss_count + \
                        wandb.config.koopman_alpha * loss_rollout
                        # geo_alpha * loss_geo
                        # 1 * loss_dyn + \

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
        epoch_metrics["rollout"] += loss_rollout.item() if wandb.config.use_koopman_loss else 0
        epoch_metrics["geo"] += loss_geo.item() if wandb.config.use_koopman_loss else 0

        pbar.set_postfix(loss=loss_total.item())

    # 平均計算 & 履歴保存
    n_batches = len(train_loader)
    history["train_loss"].append(epoch_metrics["loss"] / n_batches)
    history["train_ce"].append(epoch_metrics["ce"] / n_batches)
    history["train_dyn"].append(epoch_metrics["dyn"] / n_batches)
    history["train_linear"].append(epoch_metrics["linear"] / n_batches)
    history["train_count"].append(epoch_metrics["count"] / n_batches)
    history["train_mode"].append(epoch_metrics["mode"] / n_batches)
    history["train_rollout"].append(epoch_metrics["rollout"] / n_batches)
    history["train_geo"].append(epoch_metrics["geo"] / n_batches)


    # =========================================================
    #  Validation Loop (Trainと同じロジック + Rollout)
    # =========================================================
    model.eval()
    epoch_metrics_val = {
        "loss": 0.0, "ce": 0.0, "dyn": 0.0, "linear": 0.0, 
        "count": 0.0, "mode": 0.0, "rollout": 0.0, "geo":0.0
    }    

    with torch.no_grad():
        for batch in val_loader:
            r_b, t_b, a_b, h_b, tz_b, e_b = batch
            r_b, t_b, a_b = r_b.to(device), t_b.to(device), a_b.to(device)
            h_b, tz_b, e_b = h_b.to(device), tz_b.to(device), e_b.to(device)

            # 1. 前処理 (Trainと同じ)
            tokenizer = Tokenization(network)
            input_tokens = tokenizer.tokenization(r_b, mode="simple").long().to(device)
            target_tokens = tokenizer.tokenization(r_b, mode="next").long().to(device)
            stay_counts = tokenizer.calculate_stay_counts(input_tokens)

            B_size, T = input_tokens.shape
            
            # Context Alignment
            def align_ctx(ctx, target_len):
                out = torch.zeros((B_size, target_len), dtype=torch.long, device=device)
                copy_len = min(ctx.shape[1], target_len - 1)
                if copy_len > 0: out[:, 1 : 1 + copy_len] = ctx[:, :copy_len]
                return out
                
            h_in, tz_in, e_in = align_ctx(h_b, T), align_ctx(tz_b, T), align_ctx(e_b, T)

            # 2. Forward
            logits, z_hat, z_pred_next, u_all = model(
                tokens=input_tokens, 
                stay_counts=stay_counts, 
                agent_ids=a_b, 
                holidays=h_in, 
                time_zones=tz_in, 
                events=e_in
            )

            # 基本Loss (CE)
            loss_ce = ce_loss_fn(logits.view(-1, vocab_size), target_tokens.view(-1))

            loss_geo = model.calc_geo_loss(logits, target_tokens)
            
            # 各種Loss初期化
            loss_dyn = torch.tensor(0.0, device=device)
            loss_k = torch.tensor(0.0, device=device)
            loss_count = torch.tensor(0.0, device=device)
            loss_mode = torch.tensor(0.0, device=device)
            loss_rollout = torch.tensor(0.0, device=device)

            if wandb.config.use_koopman_loss:
                valid_mask = (input_tokens != network.N)
                
                # --- (A) Count Reconstruction ---
                pred_count = model.count_decoder(z_hat).squeeze(-1)
                loss_count = masked_mse_loss(pred_count, stay_counts.float(), valid_mask)

                # --- (B) Multi-step Dynamics (Z-space MSE) ---
                K_steps = 5
                current_z = z_hat[:, :-K_steps, :]
                for k in range(K_steps):
                    end_idx = -K_steps + k if (-K_steps + k) != 0 else None
                    u_curr_step = u_all[:, k : end_idx, :]
                    
                    term_A = torch.einsum("ij,btj->bti", model.A, current_z)
                    term_B = torch.einsum("ij,btj->bti", model.B, u_curr_step)
                    pred_z_rec = term_A + term_B
                    
                    start_true = k + 1
                    end_true = -K_steps + k + 1 if (-K_steps + k + 1) != 0 else None
                    true_z_next = z_hat[:, start_true : end_true, :]
                    
                    future_mask = valid_mask[:, start_true : end_true]
                    loss_dyn += masked_mse_loss(pred_z_rec, true_z_next, future_mask) * (0.8 ** k)
                    current_z = pred_z_rec

                # --- (C) Single-step Linear (Loss K) ---
                z_pred_step = z_pred_next[:, :-1, :]
                z_true_step = z_hat[:, 1:, :]
                next_step_mask = valid_mask[:, 1:]
                loss_k = masked_mse_loss(z_pred_step, z_true_step, next_step_mask)
            
                # --- (D) Mode Loss ---
                target_modes = torch.full_like(target_tokens, -100)
                is_move = (target_tokens >= 0) & (target_tokens < network.N)
                target_modes[is_move] = 1
                STAY_OFFSET = network.N
                PAD_TOKEN = network.N * 2
                is_stay = (target_tokens >= STAY_OFFSET) & (target_tokens < PAD_TOKEN)
                target_modes[is_stay] = 0
                
                pred_modes = model.mode_classifier(z_pred_next)
                loss_mode = nn.CrossEntropyLoss(ignore_index=-100)(
                    pred_modes.view(-1, 2), target_modes.view(-1)
                )

                # --- (E) Detached Rollout Loss (New!) ---
                # Trainと同じロジックで評価
                if wandb.config.rollout_steps > 0:
                    K_roll = wandb.config.rollout_steps
                    trip_hol, trip_night, trip_ev_defs = ctx_manager.get_trip_context(t_b)
                    
                    # 評価時はランダムではなく「系列の真ん中あたり」や「固定点」から始めると安定しますが
                    # ここではTrain条件と合わせるため同様にランダム、または複数回試行が理想です。
                    # 簡易的にTrainと同じランダムスタートを採用します。
                    start_t = np.random.randint(1, max(2, T - K_roll - 1))
                    
                    z_curr = z_pred_next[:, start_t, :] # detach不要(no_grad内なので)
                    pred_logits_init = model.to_logits(z_curr)
                    pred_token = torch.argmax(pred_logits_init, dim=-1)
                    
                    prev_token = input_tokens[:, start_t]
                    prev_stay_count = stay_counts[:, start_t]
                    
                    rollout_loss_accum = []
                    
                    for k in range(K_roll):
                        target_idx = start_t + 1 + k
                        if target_idx >= T: break
                        gt_token = target_tokens[:, target_idx]
                        
                        # Loss計算
                        mask = (gt_token != 38)
                        if mask.sum() > 0:
                            logits_k = model.to_logits(z_curr)
                            step_loss = ce_loss_fn(logits_k[mask], gt_token[mask])
                            rollout_loss_accum.append(step_loss)
                        
                        # 次入力作成
                        curr_token_in = pred_token
                        next_stay_count = torch.ones_like(curr_token_in)
                        is_stay_node = (curr_token_in >= base_N) & (curr_token_in < base_N * 2)
                        is_same = (curr_token_in == prev_token)
                        inc_mask = is_stay_node & is_same
                        next_stay_count[inc_mask] = prev_stay_count[inc_mask] + 1
                        next_stay_count = torch.clamp(next_stay_count, max=wandb.config.max_stay_count)
                        
                        curr_event_flag = ctx_manager.get_event_flag(curr_token_in, trip_ev_defs)
                        u_next = model.get_single_step_input(
                            curr_token_in, next_stay_count, a_b, trip_hol, trip_night, curr_event_flag
                        )
                        
                        z_next, logits_next = model.forward_step(z_curr, u_next)
                        
                        z_curr = z_next
                        prev_token = curr_token_in
                        prev_stay_count = next_stay_count
                        pred_token = torch.argmax(logits_next, dim=-1)

                    if rollout_loss_accum:
                        loss_rollout = torch.stack(rollout_loss_accum).mean()

            # Total Loss (Trainと重みを合わせる)
            loss_total = loss_ce + \
                        wandb.config.koopman_alpha * loss_k + \
                        wandb.config.koopman_alpha * loss_rollout + \
                        0.01 * loss_count + \
                        0.1 * loss_mode
                        # geo_alpha * loss_geo
                        # 1 * loss_dyn + \

            epoch_metrics_val["loss"] += loss_total.item()
            epoch_metrics_val["ce"] += loss_ce.item()
            epoch_metrics_val["dyn"] += loss_dyn.item() if wandb.config.use_koopman_loss else 0
            epoch_metrics_val["linear"] += loss_k.item() if wandb.config.use_koopman_loss else 0
            epoch_metrics_val["count"] += loss_count.item() if wandb.config.use_koopman_loss else 0
            epoch_metrics_val["mode"] += loss_mode.item() if wandb.config.use_koopman_loss else 0
            epoch_metrics_val["rollout"] += loss_rollout.item() if wandb.config.use_koopman_loss else 0
            epoch_metrics_val["geo"] += loss_geo.item() if wandb.config.use_koopman_loss else 0

    # 平均計算 & 履歴保存
    n_val = len(val_loader)
    history["val_loss"].append(epoch_metrics_val["loss"] / n_val)
    history["val_ce"].append(epoch_metrics_val["ce"] / n_val)
    history["val_dyn"].append(epoch_metrics_val["dyn"] / n_val)
    history["val_linear"].append(epoch_metrics_val["linear"] / n_val)
    history["val_count"].append(epoch_metrics_val["count"] / n_val)
    history["val_mode"].append(epoch_metrics_val["mode"] / n_val)
    history["val_rollout"].append(epoch_metrics_val["rollout"] / n_val)
    history["val_geo"].append(epoch_metrics_val["geo"] / n_val)

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

    # ax = axes[0, 1]
    # ax.plot(epochs_range, history["train_geo"], label='Train', marker='.', color='orange')
    # ax.plot(epochs_range, history["val_geo"], label='Val', marker='.', color='red')
    # ax.set_title('Geo CE Loss')
    # ax.legend()
    # ax.grid(True)
    
    # 3. Dynamics Loss (Multi-step)
    ax = axes[0, 2]
    ax.plot(epochs_range, history["train_rollout"], label='Train', marker='.', color='green')
    ax.plot(epochs_range, history["val_rollout"], label='Val', marker='.', color='lime')
    ax.set_title('Multi-step rollout Dynamics (MSE)')
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