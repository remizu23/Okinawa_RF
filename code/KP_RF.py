import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EmbeddingWithFeatures(nn.Module):
    def __init__(self, vocab_size, token_dim, feature_dim=None, feature_emb_dim=None, dropout=0.1):
        super(EmbeddingWithFeatures, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, token_dim, padding_idx=38)
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
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

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
        dist_mat_base: torch.Tensor | None = None,
        base_N: int | None = None,
        delta_bias_move_only: bool = True,
        dist_is_directed: bool = False,
        num_agents=1,        
        agent_emb_dim=16,    
        max_stay_count=500,  
        stay_emb_dim=16,
        holiday_emb_dim=4,
        time_zone_emb_dim=4,
        event_emb_dim=4
    ):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, token_emb_dim, padding_idx=pad_token_id)
        self.agent_embedding = nn.Embedding(num_agents, agent_emb_dim)
        self.stay_embedding = nn.Embedding(max_stay_count + 1, stay_emb_dim, padding_idx=0)

        self.holiday_embedding = nn.Embedding(2, holiday_emb_dim)
        self.time_zone_embedding = nn.Embedding(2, time_zone_emb_dim)
        self.event_embedding = nn.Embedding(2, event_emb_dim)

        total_input_dim = (token_emb_dim + agent_emb_dim + stay_emb_dim + 
                           holiday_emb_dim + time_zone_emb_dim + event_emb_dim)
        
        self.pos_encoder = PositionalEncoding(d_model)
        self.input_proj = nn.Linear(total_input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_ff, 
            batch_first=True
        )
        self.transformer_block = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.to_z = nn.Linear(d_model, z_dim)
        self.to_logits = nn.Linear(z_dim, vocab_size) 

        self.base_N = int(base_N) if base_N is not None else int(pad_token_id) // 2
        self.pad_token_id = pad_token_id

        # Koopman Dynamics
        self.A = nn.Parameter(torch.randn(z_dim, z_dim) * 0.05)
        self.B = nn.Parameter(torch.randn(z_dim, total_input_dim) * 0.05) 

        self.count_decoder = nn.Linear(z_dim, 1)
        self.mode_classifier = nn.Linear(z_dim, 2)


        if dist_mat_base is not None:
            # 距離行列をバッファとして登録
            if not isinstance(dist_mat_base, torch.Tensor):
                dist_mat_base = torch.tensor(dist_mat_base)
            self.register_buffer("dist_mat_base", dist_mat_base.long())

            # ★ここが重要: candidate_base_id の作成と登録
            candidate_ids = torch.arange(vocab_size, dtype=torch.long)
            is_node_token = candidate_ids < pad_token_id 
            candidate_base = torch.zeros_like(candidate_ids)
            # base_N で割った余りがベースノードID (例: 0->0, 19->0)
            candidate_base[is_node_token] = candidate_ids[is_node_token] % self.base_N
            
            self.register_buffer("candidate_base_id", candidate_base) # <--- これが必要です！

            # (以下略)
            self.delta_gate = nn.Linear(z_dim, 1)
            self.delta_bin_weight = nn.Parameter(torch.zeros(2, 3))

        else:
            self.dist_mat_base = None
            self.candidate_base_id = None # <--- ない場合はNoneを入れる
            self.delta_gate = None
            self.delta_bin_weight = None

    def forward(
        self, 
        tokens, 
        stay_counts, 
        agent_ids, 
        holidays,
        time_zones,
        events,
        time_tensor=None
    ):
        batch_size, seq_len = tokens.size()

        token_vec = self.token_embedding(tokens)
        stay_vec = self.stay_embedding(stay_counts)
        
        agent_vec = self.agent_embedding(agent_ids)
        agent_vec = agent_vec.unsqueeze(1).expand(-1, seq_len, -1)

        holiday_vec = self.holiday_embedding(holidays)
        timezone_vec = self.time_zone_embedding(time_zones)
        event_vec = self.event_embedding(events)

        u_all = torch.cat([
            token_vec, stay_vec, agent_vec, 
            holiday_vec, timezone_vec, event_vec
        ], dim=-1)        

        x = self.input_proj(u_all)
        x = self.pos_encoder(x)
        
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        pad_mask = (tokens == self.token_embedding.padding_idx).to(x.device)
        
        h = self.transformer_block(
            src=x, 
            mask=causal_mask, 
            src_key_padding_mask=pad_mask
        )
        z_hat = self.to_z(h) # [B, T, z_dim]

        # Batch Dynamics
        z_pred_next = (
            torch.einsum("ij,btj->bti", self.A, z_hat) + 
            torch.einsum("ij,btj->bti", self.B, u_all)
        )

        logits = self.to_logits(z_pred_next) 

        return logits, z_hat, z_pred_next, u_all

    # =========================================================================
    # ★ New Methods for Multi-step Rollout
    # =========================================================================
    
    def get_single_step_input(self, token_id, stay_count, agent_id, holiday, timezone, event):
        """
        単一ステップ用の入力埋め込み u(t) を作成するヘルパー関数
        Returns: [Batch, total_input_dim]
        """
        # 各IDは [Batch] の1次元テンソルを想定
        
        token_vec = self.token_embedding(token_id)      # [B, dim]
        stay_vec  = self.stay_embedding(stay_count)     # [B, dim]
        agent_vec = self.agent_embedding(agent_id)      # [B, dim]
        
        hol_vec   = self.holiday_embedding(holiday)     # [B, dim]
        tz_vec    = self.time_zone_embedding(timezone)  # [B, dim]
        evt_vec   = self.event_embedding(event)         # [B, dim]
        
        u_t = torch.cat([
            token_vec, stay_vec, agent_vec, 
            hol_vec, tz_vec, evt_vec
        ], dim=-1)
        
        return u_t

    def forward_step(self, z_curr, u_curr):
        """
        z(t+1) = A * z(t) + B * u(t)
        Returns: z_next, logits
        """
        # z_curr: [B, z_dim], u_curr: [B, input_dim]
        # A: [z_dim, z_dim], B: [z_dim, input_dim]
        
        term_A = torch.matmul(z_curr, self.A.t()) # [B, z_dim]
        term_B = torch.matmul(u_curr, self.B.t()) # [B, z_dim]
        
        z_next = term_A + term_B
        logits = self.to_logits(z_next)
        
        return z_next, logits

    # KP_RF.py の KoopmanRoutesFormer クラス内に追加/修正
    def calc_geo_loss(self, logits, targets):
        """
        地理的距離損失 (Expected Distance Loss)
        logits: [Batch, Seq, Vocab]
        targets: [Batch, Seq]
        """
        # 1. 確率分布の計算
        probs = F.softmax(logits, dim=-1) # [B, T, V]
        
        # ★ここが修正点: self.candidate_base_id が存在するかチェック
        if not hasattr(self, 'candidate_base_id') or self.candidate_base_id is None:
            return torch.tensor(0.0, device=logits.device)

        # 2. ターゲット(正解)に対応する「ベースノードID」を取得
        # self.candidate_base_id: TokenID -> BaseNodeID のマッピング
        target_base_nodes = self.candidate_base_id[targets] # [B, T]
        
        # 3. 距離行列の参照
        # self.dist_mat_base: [BaseN, BaseN]
        dists_to_nodes = self.dist_mat_base[target_base_nodes] 
        
        # 4. 全トークンへの距離に拡張
        dists_to_tokens = dists_to_nodes[:, :, self.candidate_base_id] # [B, T, V]
        
        # 5. 期待距離の計算
        expected_dist = torch.sum(probs * dists_to_tokens, dim=-1)
        
        # 6. マスク処理 (Move/Stayトークンのみ有効)
        valid_mask = (targets < self.base_N * 2) 
        
        loss = (expected_dist * valid_mask.float()).sum() / (valid_mask.sum() + 1e-6)
        
        return loss

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