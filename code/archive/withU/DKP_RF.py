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
        event_emb_dim=4,
        use_aux_loss=False,  # 補助損失を使うかどうか
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
        self.vocab_size = vocab_size
        self.z_dim = z_dim
        # self.u_dim = z_dim
        self.max_stay_count = max_stay_count
        self.use_aux_loss = use_aux_loss

        # u(t)側（制御入力）用：新規に追加（別パラメータ）
        self.holiday_embedding_u = nn.Embedding(2, holiday_emb_dim)
        self.time_zone_embedding_u = nn.Embedding(2, time_zone_emb_dim)
        self.event_embedding_u = nn.Embedding(2, event_emb_dim)

        # Koopman Dynamics (B行列は削除)
        self.A = nn.Parameter(torch.randn(z_dim, z_dim) * 0.05)

        # # 追加：制御入力 u(t) 用のノード埋め込み（Transformerの token_embedding と分離）
        # self.control_embedding = nn.Embedding(self.base_N*2, self.u_dim)

        # # 追加：B行列
        # self.B = nn.Parameter(torch.randn(self.z_dim, self.u_dim) * 0.005)

        # ---- ここから差し替え ----
        # token由来の制御埋め込み次元（いままでと同じ z_dim にしておくのが最小変更）
        self.u_tok_dim = z_dim

        # u(t) 全体は「token_u + (holiday/timezone/event)」を結合した次元
        self.u_dim = self.u_tok_dim + holiday_emb_dim + time_zone_emb_dim + event_emb_dim

        # 制御入力 u(t) 用：トークン種別（move/stayノード）埋め込み
        self.control_token_embedding = nn.Embedding(self.base_N * 2, self.u_tok_dim)

        # B行列（z_dim × u_dim に拡張）
        self.B = nn.Parameter(torch.randn(self.z_dim, self.u_dim) * 0.005)
        # ---- ここまで差し替え ----

        # スケジュールサンプリング
        self.p_tf = 1.0


        # 補助損失用のヘッド
        if use_aux_loss:
            self.count_decoder = nn.Linear(z_dim, max_stay_count + 1)  # カウント予測
            self.mode_classifier = nn.Linear(z_dim, 2)  # Move(0)/Stay(1)分類

        if dist_mat_base is not None:
            if not isinstance(dist_mat_base, torch.Tensor):
                dist_mat_base = torch.tensor(dist_mat_base)
            self.register_buffer("dist_mat_base", dist_mat_base.long())

            candidate_ids = torch.arange(vocab_size, dtype=torch.long)
            is_node_token = candidate_ids < pad_token_id 
            candidate_base = torch.zeros_like(candidate_ids)
            candidate_base[is_node_token] = candidate_ids[is_node_token] % self.base_N
            
            self.register_buffer("candidate_base_id", candidate_base)

            self.delta_gate = nn.Linear(z_dim, 1)
            self.delta_bin_weight = nn.Parameter(torch.zeros(2, 3))

        else:
            self.dist_mat_base = None
            self.candidate_base_id = None
            self.delta_gate = None
            self.delta_bin_weight = None

    def encode_prefix(
        self, 
        prefix_tokens, 
        prefix_stay_counts, 
        prefix_agent_ids, 
        prefix_holidays,
        prefix_time_zones,
        prefix_events,
        prefix_mask=None
    ):
        """
        Prefix を Transformer でエンコードして初期潜在状態 z_0 を作成
        
        Args:
            prefix_tokens: [B, Lp]
            prefix_stay_counts: [B, Lp]
            prefix_agent_ids: [B] または [B, Lp]
            prefix_holidays: [B, Lp]
            prefix_time_zones: [B, Lp]
            prefix_events: [B, Lp]
            prefix_mask: [B, Lp] (True=パディング位置)
        
        Returns:
            z_0: [B, z_dim]
            h_last: [B, d_model] (補助損失用)
        """
        batch_size, seq_len = prefix_tokens.size()

        # 埋め込み
        token_vec = self.token_embedding(prefix_tokens)
        stay_vec = self.stay_embedding(prefix_stay_counts)
        
        # agent_ids の形状処理
        if prefix_agent_ids.dim() == 1:
            agent_vec = self.agent_embedding(prefix_agent_ids)
            agent_vec = agent_vec.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            agent_vec = self.agent_embedding(prefix_agent_ids)

        holiday_vec = self.holiday_embedding(prefix_holidays)
        timezone_vec = self.time_zone_embedding(prefix_time_zones)
        event_vec = self.event_embedding(prefix_events)

        u_all = torch.cat([
            token_vec, stay_vec, agent_vec, 
            holiday_vec, timezone_vec, event_vec
        ], dim=-1)        

        x = self.input_proj(u_all)
        x = self.pos_encoder(x)
        
        # Causal mask と padding mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        if prefix_mask is None:
            pad_mask = (prefix_tokens == self.pad_token_id).to(x.device)
        else:
            pad_mask = prefix_mask.to(x.device)
        
        h = self.transformer_block(
            src=x, 
            mask=causal_mask, 
            src_key_padding_mask=pad_mask
        )  # [B, Lp, d_model]
        
        # 最後の有効位置を取得
        # pad_mask が True の位置は無効
        valid_lengths = (~pad_mask).sum(dim=1)  # [B]
        last_indices = (valid_lengths - 1).clamp(min=0)  # [B]
        
        # 各バッチの最後の有効位置の hidden state を取得
        h_last = h[torch.arange(batch_size), last_indices, :]  # [B, d_model]
        
        # z_0 に投影
        z_0 = self.to_z(h_last)  # [B, z_dim]
        
        return z_0, h_last

    def forward_rollout(
        self,
        prefix_tokens,
        prefix_stay_counts,
        prefix_agent_ids,
        prefix_holidays,
        prefix_time_zones,
        prefix_events,
        K,  # ロールアウトステップ数
        future_tokens=None,  # 学習時の正解データ [B, K]
        prefix_mask=None,
        prefix_times=None,
    ):
        """
        Koopman 自律ロールアウト (B項なし)
        
        Args:
            prefix_*: Prefix入力
            K: ロールアウトステップ数
            future_tokens: 学習時の正解 [B, K]
            prefix_mask: [B, Lp]
        
        Returns:
            dict with:
                - pred_logits: [B, K, vocab]
                - z_0: [B, z_dim]
                - aux_losses: 補助損失の辞書 (use_aux_loss=True時)
        """
        # 1. Prefix から z_0 を作成
        z_0, h_last = self.encode_prefix(
            prefix_tokens, 
            prefix_stay_counts, 
            prefix_agent_ids,
            prefix_holidays,
            prefix_time_zones,
            prefix_events,
            prefix_mask
        )
        
        # 2. 補助損失の計算（学習時のみ）
        aux_losses = {}
        if self.use_aux_loss and self.training and future_tokens is not None:
            # Prefixの最後のトークンの特性を復元
            batch_size = prefix_tokens.size(0)
            if prefix_mask is None:
                valid_mask = (prefix_tokens != self.pad_token_id)
            else:
                valid_mask = ~prefix_mask
            
            valid_lengths = valid_mask.sum(dim=1)
            last_indices = (valid_lengths - 1).clamp(min=0)
            
            last_tokens = prefix_tokens[torch.arange(batch_size), last_indices]
            last_stay_counts = prefix_stay_counts[torch.arange(batch_size), last_indices]
            
            # Move(0) / Stay(1) の判定
            last_mode = (last_tokens >= self.base_N).long()
            
            # 予測
            count_logits = self.count_decoder(z_0)  # [B, max_stay_count+1]
            mode_logits = self.mode_classifier(z_0)  # [B, 2]
            
            # 損失計算
            aux_losses['count'] = F.cross_entropy(count_logits, last_stay_counts)
            aux_losses['mode'] = F.cross_entropy(mode_logits, last_mode)
        
        # 3. Koopman 自律ロールアウト
        z = z_0
        logits_list = []
        z_list = [z_0]  # ★追加：z_0 から保持
        u_list = []
        Bu_list = []

        # 初期の「現在トークン」＝ prefix の最後の有効トークン
        batch_size = prefix_tokens.size(0)
        if prefix_mask is None:
            pad_mask = (prefix_tokens == self.pad_token_id)
        else:
            pad_mask = prefix_mask
        valid_lengths = (~pad_mask).sum(dim=1)            # [B]
        last_indices = (valid_lengths - 1).clamp(min=0)   # [B]
        cur_tok = prefix_tokens[torch.arange(batch_size), last_indices]  # [B]

        # 念のため：padが紛れたら0に落とす（学習データが正しければ基本起きない）
        cur_tok = cur_tok.clamp(min=0, max=self.base_N * 2 - 1)

        # prefix最後の条件を固定して使う（future_* は参照しない）
        fixed_h = prefix_holidays[torch.arange(batch_size), last_indices]     # [B]
        fixed_tz = prefix_time_zones[torch.arange(batch_size), last_indices]  # [B]
        fixed_e = prefix_events[torch.arange(batch_size), last_indices]       # [B]

        # ---------------------------
        # prefix時刻（系列代表時刻）で条件を一回だけ判定
        # ---------------------------
        if prefix_times is None:
            raise ValueError("prefix_times is required for context determination")

        # prefix_times: [B]  (例: 202409290900 のような int)
        t_int = prefix_times.long()

        date_int = t_int // 10000
        hour = (t_int // 100) % 100

        # 休日判定（configと同じ）
        HOLIDAYS = {20240928, 20240929, 20251122, 20251123}
        fixed_h = torch.isin(date_int, torch.tensor(list(HOLIDAYS), device=t_int.device)).long()

        # 夜判定（19:00-02:00）
        night_start = 19
        night_end = 2
        fixed_tz = ((hour >= night_start) | (hour < night_end)).long()

        # イベント対象ノード集合（prefix時刻で決定）
        # events: (date, start_hour, end_hour, [nodes])
        EVENTS = [
            (20240929, 9, 16, [14]),
            (20251122, 10, 19, [2, 11]),
            (20251123, 10, 16, [2]),
        ]

        # event_node_mask[b, n] = 1 なら、その系列(b)のイベント対象ノード n
        event_node_mask = torch.zeros((t_int.size(0), self.base_N), device=t_int.device, dtype=torch.bool)
        for ev_date, ev_start, ev_end, ev_nodes in EVENTS:
            cond = (date_int == ev_date) & (ev_start <= hour) & (hour < ev_end)
            if cond.any():
                node_idx = torch.tensor(ev_nodes, device=t_int.device, dtype=torch.long)
                event_node_mask[cond] = event_node_mask[cond].clone()
                event_node_mask[cond][:, node_idx] = True

        for k in range(K):
            # --- (1) そのステップの条件を決める（学習時は future_* を優先） ---
            # cur_tok -> node_id（move/stay どちらでも同じ node_id）
            # cur_tok は既に 0..2*base_N-1 に clamp 済みの想定 :contentReference[oaicite:5]{index=5}
            node_id = cur_tok % self.base_N  # [B]

            # event_flag: イベント対象ノードなら1、そうでなければ0
            batch_idx = torch.arange(batch_size, device=cur_tok.device)
            e_k = event_node_mask[batch_idx, node_id].long()  # [B]

            h_k = fixed_h      # [B] すべてのトークンに同じ
            tz_k = fixed_tz    # [B] すべてのトークンに同じ

            # --- (2) u(t) を concat で構築 ---
            u_tok = self.control_token_embedding(cur_tok)  # [B, u_tok_dim]
            u_h  = self.holiday_embedding_u(h_k)
            u_tz = self.time_zone_embedding_u(tz_k)
            u_e  = self.event_embedding_u(e_k)
              # [B, event_emb_dim]

            u = torch.cat([u_tok, u_h, u_tz, u_e], dim=-1) # [B, u_dim]
            u_list.append(u)

            Bu = u @ self.B.T                               # [B, z_dim]
            Bu_list.append(Bu)

            z = z @ self.A.T + 0.1 * Bu
            z_list.append(z)

            logits = self.to_logits(z)
            logits_list.append(logits)

            # --- 次トークン（既存ロジックそのまま） ---
            if self.training and (future_tokens is not None):
                if torch.rand(1, device=logits.device).item() < float(self.p_tf):
                    cur_tok = future_tokens[:, k]
                else:
                    cur_tok = torch.argmax(logits, dim=-1)
            else:
                cur_tok = torch.argmax(logits, dim=-1)

            cur_tok = cur_tok.clamp(min=0, max=self.base_N * 2 - 1)



        pred_logits = torch.stack(logits_list, dim=1)  # [B, K, vocab]
        
        return {
            'pred_logits': pred_logits,
            'z_0': z_0,
            "u_traj": torch.stack(u_list, dim=1),  # ★追加: [B, K, u_dim]
            "Bu_traj": torch.stack(Bu_list, dim=1),  # ★追加: [B, K, z_dim]
            "z_traj": torch.stack(z_list, dim=1),  # ★追加: [B, K+1, z_dim]
            'aux_losses': aux_losses,
        }

    def calc_geo_loss(self, logits, targets):
        """
        地理的距離損失 (Expected Distance Loss)
        logits: [Batch, Seq, Vocab]
        targets: [Batch, Seq]
        """
        # 確率分布の計算
        probs = F.softmax(logits, dim=-1)  # [B, T, V]
        
        if not hasattr(self, 'candidate_base_id') or self.candidate_base_id is None:
            return torch.tensor(0.0, device=logits.device)

        # ターゲット(正解)に対応する「ベースノードID」を取得
        target_base_nodes = self.candidate_base_id[targets]  # [B, T]
        
        # 距離行列の参照
        dists_to_nodes = self.dist_mat_base[target_base_nodes] 
        
        # 全トークンへの距離に拡張
        dists_to_tokens = dists_to_nodes[:, :, self.candidate_base_id]  # [B, T, V]
        
        # 期待距離の計算
        expected_dist = torch.sum(probs * dists_to_tokens, dim=-1)
        
        # マスク処理 (Move/Stayトークンのみ有効)
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
            return

        device = self.dist_mat_base.device
        dtype = self.dist_mat_base.dtype
        
        self.dist_mat_base.data = full_dist_mat.to(device=device, dtype=dtype)