import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 既存の便利クラスはそのまま利用
class EmbeddingWithFeatures(nn.Module):
    def __init__(self, vocab_size, token_dim, feature_dim=None, feature_emb_dim=None, dropout=0.1):
        super(EmbeddingWithFeatures, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, token_dim, padding_idx=38) # padding_idx指定推奨
        
        # 特徴量を埋め込み次元に変換する層
        if feature_dim and feature_emb_dim:
            self.feature_projection = nn.Linear(feature_dim, feature_emb_dim)
            self.use_features = True
        else:
            self.feature_projection = None
            self.use_features = False
        self.dropout = nn.Dropout(dropout)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, tokens, features=None):
        tokens = tokens.to(self.device)
        token_emb = self.token_embedding(tokens)
        
        if self.use_features and features is not None:
            features = features.to(self.device)
            feature_emb = self.feature_projection(features)
            # トークン埋め込みと特徴量埋め込みを結合
            emb = torch.cat((token_emb, feature_emb), dim=-1)
        else:
            emb = token_emb
        return self.dropout(emb)

class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, 1, dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [Batch, Seq, Dim] -> pe: [Seq, 1, Dim] -> transpose to add
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


#0105ノード埋め込み変更版

class KoopmanRoutesFormer(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        token_emb_dim,    
        d_model,          
        nhead, 
        num_layers, 
        d_ff, 
        z_dim,            
        pad_token_id=38,
        # --- Delta-distance (to plaza) bias ---
        # dist_mat_base: [base_N, base_N] shortest-path distance matrix on the *base* graph (0..base_N-1).
        # If provided, the model will add an additive bias to logits that depends on the change in
        # distance-to-plaza between the current token and each candidate next token.
        dist_mat_base: torch.Tensor | None = None,
        # base_N: number of base nodes (e.g., 19). If None, inferred as pad_token_id//2.
        base_N: int | None = None,
        # Whether to apply delta-distance bias only to MOVE tokens (0..base_N-1). Recommended: True.
        delta_bias_move_only: bool = True,
        # Whether to treat the base graph as directed when computing distances (distance matrix must match).
        # (This flag is informational; computation happens outside this class.)
        dist_is_directed: bool = False,
        # ★追加引数
        num_agents=1,        # 全エージェント数 (IDの最大値+1)
        agent_emb_dim=16,    # エージェントIDの埋め込み次元
        max_stay_count=500,  # 滞在カウントの最大値
        stay_emb_dim=16,      # 滞在カウントの埋め込み次元

        # --- ★New Context Embeddings★ ---
        holiday_emb_dim=4,
        time_zone_emb_dim=4,
        event_emb_dim=4
    ):
        super().__init__()
        
        # 1. 埋め込み層の定義
        self.token_embedding = nn.Embedding(vocab_size, token_emb_dim, padding_idx=pad_token_id)
        
        # ★追加: ユーザーIDと滞在カウントの埋め込み
        self.agent_embedding = nn.Embedding(num_agents, agent_emb_dim)
        self.stay_embedding = nn.Embedding(max_stay_count + 1, stay_emb_dim, padding_idx=0) # 0をパディング扱いとする想定

        # ★0120追加: 広場フラグ埋め込み (0:広場じゃない, 1:広場である)
        # self.plaza_embedding = nn.Embedding(2, plaza_emb_dim)

        # 2. New Context Embeddings
        # 0:False, 1:True
        self.holiday_embedding = nn.Embedding(2, holiday_emb_dim)
        # 0:Day, 1:Night
        self.time_zone_embedding = nn.Embedding(2, time_zone_emb_dim)
        # 0:NoEvent, 1:EventActive
        self.event_embedding = nn.Embedding(2, event_emb_dim)

        # Total dimension for concatenation
        total_input_dim = (token_emb_dim + agent_emb_dim + stay_emb_dim + 
                           holiday_emb_dim + time_zone_emb_dim + event_emb_dim)
        
        self.pos_encoder = PositionalEncoding(d_model)

        # ★射影層の入力次元を変更 (total_input_dim -> d_model)
        self.input_proj = nn.Linear(total_input_dim, d_model)

        # 2. Transformer Block (変更なし)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_ff, 
            batch_first=True
        )
        self.transformer_block = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. 射影層 (変更なし)
        self.to_z = nn.Linear(d_model, z_dim)
        self.to_logits = nn.Linear(z_dim, vocab_size) 

        # --- Delta-distance bias components ---
        self.base_N = int(base_N) if base_N is not None else int(pad_token_id) // 2
        self.delta_bias_move_only = bool(delta_bias_move_only)
        self.dist_is_directed = bool(dist_is_directed)

        if dist_mat_base is not None:
            # store distance matrix as a buffer (moves with .to(device), not trained)
            if not isinstance(dist_mat_base, torch.Tensor):
                dist_mat_base = torch.tensor(dist_mat_base)
            self.register_buffer("dist_mat_base", dist_mat_base.long())

            # candidate token -> base node id (0..base_N-1). Special tokens map to 0 but will be masked out.
            candidate_ids = torch.arange(vocab_size, dtype=torch.long)
            pad_id = pad_token_id
            is_node_token = candidate_ids < pad_id  # 0..(2*base_N-1)
            candidate_base = torch.zeros_like(candidate_ids)
            candidate_base[is_node_token] = candidate_ids[is_node_token] % self.base_N
            self.register_buffer("candidate_base_id", candidate_base)

            # candidate token is a MOVE token (0..base_N-1)
            candidate_is_move = (candidate_ids < self.base_N)
            self.register_buffer("candidate_is_move", candidate_is_move)

            # Gate g_t (context-dependent strength of plaza attraction), scalar per time step
            self.delta_gate = nn.Linear(z_dim, 1)

            # 3 bins: toward (<0), same (=0), away (>0)
            # ★修正: 重みを [2, 3] に変更
            # 0行目: 2024年用 (Toward, Same, Away)
            # 1行目: 2025年用 (Toward, Same, Away)
            self.delta_bin_weight = nn.Parameter(torch.zeros(2, 3))

        else:
            # Disable delta bias if no distance matrix is given
            self.dist_mat_base = None
            self.candidate_base_id = None
            self.candidate_is_move = None
            self.delta_gate = None
            self.delta_bin_weight = None
        
        # 4. Koopman Dynamics
        # ★B行列の入力次元を「合計次元」に変更 (u_t は結合ベクトルになるため)
        self.A = nn.Parameter(torch.randn(z_dim, z_dim) * 0.05)
        self.B = nn.Parameter(torch.randn(z_dim, total_input_dim) * 0.05) 

        # 5. zからの滞在カウント復元強制
        self.count_decoder = nn.Linear(z_dim, 1)

        # 6. zからの移動/滞在
        self.mode_classifier = nn.Linear(z_dim, 2) # [Stay, Move]の2値分類

        # ★★★ 広場判定用の定数を登録 (GPU計算用) ★★★
        # Query Periods: 
        # 1. 2025-11-22 10:00 ~ 19:00
        # 2. 2025-11-23 10:00 ~ 17:00
        # これらを「202511221000」のような整数形式で保持します
        # self.plaza_periods = [
        #     (202511221000, 202511221900),
        #     (202511231000, 202511231700)
        # ]

# ★★★ ヘルパー関数: 時刻テンソルを「分単位の通算時間」に変換 ★★★
    def _to_linear_minutes(self, t_tensor):
        """
        YYYYMMDDHHMM 形式の整数テンソルを、分単位の連続値に変換する。
        （月をまたぐ計算は簡易化のため11月固定と仮定しますが、日は考慮します）
        """
        # 日、時、分を抽出
        day = (t_tensor // 10000) % 100
        hour = (t_tensor // 100) % 100
        minute = t_tensor % 100
        
        # 1日=1440分, 1時間=60分
        # 基準はとりあえず 0日0時0分 からの経過分とします
        total_minutes = day * 1440 + hour * 60 + minute
        return total_minutes


    # ★★★ ヘルパー関数: 広場開催中かどうかの厳密判定 ★★★
    def check_plaza_active(self, start_time_tensor, duration_steps):
        """
        start_time_tensor: [Batch]  (各バッチの開始時刻 YYYYMMDDHHMM)
        duration_steps: int         (系列長 = 経過分数)
        
        Returns:
            is_active: [Batch, 1, 1] (True/False mask, broadcastable)
        """
        batch_size = start_time_tensor.size(0)
        device = start_time_tensor.device
        
        # 1. トリップの開始・終了時刻を「通算分」に変換
        trip_start_mins = self._to_linear_minutes(start_time_tensor) # [Batch]
        trip_end_mins = trip_start_mins + duration_steps            # [Batch]
        
        # 結果を格納するマスク（初期値False）
        active_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # 2. 定義された各期間との重複チェック
        for (p_start_int, p_end_int) in self.plaza_periods:
            # 期間の開始・終了も「通算分」に変換
            # 定数ですが、broadcastのためにtensor化
            p_start_mins = self._to_linear_minutes(torch.tensor(p_start_int, device=device))
            p_end_mins   = self._to_linear_minutes(torch.tensor(p_end_int, device=device))
            
            # 重複判定ロジック:
            # (TripStart < PeriodEnd) AND (TripEnd > PeriodStart)
            # ※ 「少しでも被ればOK」なので、接するだけ(Start==End)は除外するなら不等号、含めるなら等号付き
            # ここでは「時間幅を持つ」前提で厳密な不等号を使います
            is_overlap = (trip_start_mins < p_end_mins) & (trip_end_mins > p_start_mins)
            
            # どれか一つの期間にでも被ればOK (OR演算)
            active_mask = active_mask | is_overlap
            
        return active_mask.view(batch_size, 1, 1) # 放送用にreshape


    def forward(
        self, 
        tokens, 
        stay_counts, 
        agent_ids, 
        holidays,      # [Batch, Seq]
        time_zones,    # [Batch, Seq]
        events,        # [Batch, Seq]
        time_tensor=None, 
        return_debug=False
    ):
        """
        Modified forward:
        Structure: Inputs -> Transformer -> z_t -> [Dynamics: Az + Bu] -> z_{t+1} -> Logits
        """
        batch_size, seq_len = tokens.size()
        device = tokens.device

        # --- 1. 各埋め込みの取得 (u_t の作成) ---
        token_vec = self.token_embedding(tokens)        # [B, T, token_dim]
        stay_vec = self.stay_embedding(stay_counts)     # [B, T, stay_dim]
        
        # AgentIDは系列全体で共通なので拡張
        agent_vec = self.agent_embedding(agent_ids)     # [B, agent_dim]
        agent_vec = agent_vec.unsqueeze(1).expand(-1, seq_len, -1) # [B, T, agent_dim]

        # Contexts
        holiday_vec = self.holiday_embedding(holidays)     # [B, T, holiday_dim]
        timezone_vec = self.time_zone_embedding(time_zones) # [B, T, timezone_dim]
        event_vec = self.event_embedding(events)           # [B, T, event_dim]

        # ★結合 (u_t): 現在時刻の入力全セット
        u_all = torch.cat([
            token_vec, stay_vec, agent_vec, 
            holiday_vec, timezone_vec, event_vec
        ], dim=-1)        

        # --- 2. Transformerへの入力 ---
        # u_all を射影して Transformer に通す
        x = self.input_proj(u_all) # [B, T, d_model]
        x = self.pos_encoder(x)
        
        # マスク作成
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        pad_mask = (tokens == self.token_embedding.padding_idx).to(x.device)
        
        # --- 3. 状態推定 (h_t -> z_t) ---
        h = self.transformer_block(
            src=x, 
            mask=causal_mask, 
            src_key_padding_mask=pad_mask
        )
        z_hat = self.to_z(h) # [B, T, z_dim] これは「現在の状態 z_t」

        # --- 4. Koopman Dynamics (z_t -> z_{t+1}) ---
        # ★ここが変更点: 系列全体に対して一括でダイナミクスを適用
        # z_{t+1} = A * z_t + B * u_t
        # output shape: [B, T, z_dim]
        # ※この z_pred_next の t番目の要素は、時刻 t+1 の状態を予測したもの
        z_pred_next = (
            torch.einsum("ij,btj->bti", self.A, z_hat) + 
            torch.einsum("ij,btj->bti", self.B, u_all)
        )

        # --- 5. Logitsの計算 ---
        # ★変更: Transformerの生出力(z_hat)ではなく、
        # ダイナミクスで予測した次ステップの状態(z_pred_next)から次ノードを予測する
        logits = self.to_logits(z_pred_next) # [B, T, vocab_size]

        if return_debug:
            # debug info (必要なら適宜修正)
            debug_info = {}
            return logits, z_hat, z_pred_next, u_all, debug_info
        else:
            return logits, z_hat, z_pred_next, u_all

    def set_plaza_dist(self, full_dist_mat: torch.Tensor, plaza_id: int):
        """
        分析・シナリオ用に、特定のノード(plaza_id)を広場とした場合の距離行列をセットする。
        full_dist_mat: [N, N] の全ペア最短距離行列
        plaza_id: 広場とみなすノードID (0 ~ base_N-1)
        """
        if self.dist_mat_base is None:
            return # 距離バイアス機能がないモデルなら無視

        # 現在のバッファと同じデバイス・型にする
        device = self.dist_mat_base.device
        dtype = self.dist_mat_base.dtype
        
        # 本来は dist_mat_base は [N, N] 全体を持っていますが、
        # モデル内では self.dist_mat_base[curr, plaza] のように参照しています。
        # シナリオ分析では「広場IDが変わる」＝「参照する列が変わる」だけなので、
        # 単純に dist_mat_base を外部から与えられた新しい行列で上書きします。
        
        self.dist_mat_base.data = full_dist_mat.to(device=device, dtype=dtype)
        
        # もし `forward` 内で plaza_base_ids を引数で受け取る仕様にしている場合、
        # このメソッドは「距離行列そのもの」を更新するため、整合性が取れます。
        # 呼び出し側は model(..., plaza_base_ids=[plaza_id]) と呼ぶ必要がありますが、
        # 提示のスクリプトは `set_plaza_dist` で行列の状態を変える前提のようなので、
        # これでOKです。 