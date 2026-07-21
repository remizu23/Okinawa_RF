import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from context_modes import (
    ContextAblationConfig,
    compute_u,
)

class EmbeddingWithFeatures(nn.Module): # Koopmanでは不使用
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
        encoder_type: str = "transformer",
        max_prefix_len: int = 8,
        context_ablation: ContextAblationConfig | None = None,
        stay_u_threshold: int = 3,
        context_config: dict | None = None,
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
        self.encoder_type = str(encoder_type).lower()
        self.max_prefix_len = int(max_prefix_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            batch_first=True
        )
        self.transformer_block = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lstm_encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=max(1, int(num_layers)),
            batch_first=True,
            dropout=0.0 if int(num_layers) <= 1 else 0.1,
        )
        self.mlp_flat_proj = nn.Linear(d_model * self.max_prefix_len, z_dim)
        self.to_z = nn.Linear(d_model, z_dim)
        self.to_logits = nn.Linear(z_dim, vocab_size) 

        self.base_N = int(base_N) if base_N is not None else int(pad_token_id) // 2
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.z_dim = z_dim
        self.max_stay_count = max_stay_count
        self.use_aux_loss = use_aux_loss
        if self.encoder_type not in {"transformer", "lstm", "mlp_flat"}:
            raise ValueError(f"Unsupported encoder_type: {self.encoder_type}")

        self.context_ablation = context_ablation or ContextAblationConfig()
        self.stay_u_threshold = int(stay_u_threshold)
        self.context_config = context_config
        self.uses_dual_A = self.context_ablation.uses_dual_A

        # Koopman Dynamics (B行列は削除)
        if self.uses_dual_A:
            self.A0 = nn.Parameter(torch.randn(z_dim, z_dim) * 0.05)
            self.delta_A = nn.Parameter(torch.randn(z_dim, z_dim) * 0.05)
        else:
            self.A = nn.Parameter(torch.randn(z_dim, z_dim) * 0.05)

        # u shuffle control (training only)
        self._u_perm: torch.Tensor | None = None
        self._random_u_labels: bool = False

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

    def set_u_permutation(self, u_perm: torch.Tensor | None) -> None:
        """Set epoch-fixed u permutation for random-u-label control (training)."""
        self._u_perm = u_perm

    def set_random_u_labels(self, enabled: bool) -> None:
        self._random_u_labels = bool(enabled)

    def _embed_context_feature(self, name: str, indices: torch.Tensor, embedding: nn.Embedding):
        vec = embedding(indices)
        if self.context_ablation.should_zero_embed(name):
            return torch.zeros_like(vec)
        return vec

    def compute_u(
        self,
        prefix_stay_counts: torch.Tensor,
        prefix_times: torch.Tensor | None = None,
        prefix_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return compute_u(
            self.context_ablation,
            prefix_stay_counts,
            prefix_times=prefix_times,
            prefix_mask=prefix_mask,
            context_config=self.context_config,
        )

    def get_A_for_u(self, u: torch.Tensor) -> torch.Tensor:
        """Return effective A matrices [B, z_dim, z_dim]."""
        if not self.uses_dual_A:
            return self.A.unsqueeze(0).expand(u.shape[0], -1, -1)
        u_col = u.view(-1, 1, 1)
        return self.A0.unsqueeze(0) + u_col * self.delta_A.unsqueeze(0)

    def _apply_koopman_step(self, z: torch.Tensor, A_eff: torch.Tensor) -> torch.Tensor:
        return torch.bmm(z.unsqueeze(1), A_eff.transpose(1, 2)).squeeze(1)

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
        stay_vec = self._embed_context_feature("stay", prefix_stay_counts, self.stay_embedding)

        # agent_ids の形状処理
        if prefix_agent_ids.dim() == 1:
            agent_vec = self.agent_embedding(prefix_agent_ids)
            agent_vec = agent_vec.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            agent_vec = self.agent_embedding(prefix_agent_ids)

        holiday_vec = self.holiday_embedding(prefix_holidays)
        timezone_vec = self._embed_context_feature(
            "timezone", prefix_time_zones, self.time_zone_embedding
        )
        event_vec = self._embed_context_feature("event", prefix_events, self.event_embedding)

        u_all = torch.cat([
            token_vec, stay_vec, agent_vec, 
            holiday_vec, timezone_vec, event_vec
        ], dim=-1)        

        x = self.input_proj(u_all)  # [B, Lp, d_model]

        if prefix_mask is None:
            pad_mask = (prefix_tokens == self.pad_token_id).to(x.device)
        else:
            pad_mask = prefix_mask.to(x.device)

        valid_lengths = (~pad_mask).sum(dim=1)  # [B]
        last_indices = (valid_lengths - 1).clamp(min=0)  # [B]

        if self.encoder_type == "transformer":
            x_enc = self.pos_encoder(x)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
            h = self.transformer_block(
                src=x_enc,
                mask=causal_mask,
                src_key_padding_mask=pad_mask
            )  # [B, Lp, d_model]
            h_last = h[torch.arange(batch_size), last_indices, :]  # [B, d_model]
            z_0 = self.to_z(h_last)  # [B, z_dim]

        elif self.encoder_type == "lstm":
            x_masked = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            h_seq, _ = self.lstm_encoder(x_masked)  # [B, Lp, d_model]
            h_last = h_seq[torch.arange(batch_size), last_indices, :]
            z_0 = self.to_z(h_last)

        else:  # mlp_flat
            # 順序を強制しない単純flatten: [B, Lp, d_model] -> [B, max_prefix_len*d_model]
            x_masked = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            if seq_len >= self.max_prefix_len:
                x_fixed = x_masked[:, :self.max_prefix_len, :]
            else:
                pad_len = self.max_prefix_len - seq_len
                x_fixed = F.pad(x_masked, (0, 0, 0, pad_len), mode="constant", value=0.0)
            flat = x_fixed.reshape(batch_size, self.max_prefix_len * x_fixed.size(-1))
            z_0 = self.mlp_flat_proj(flat)

            # 補助用途に使える代表ベクトル（学習損失の主対象ではない）
            denom = valid_lengths.clamp(min=1).unsqueeze(-1).to(x.dtype)
            h_last = x_masked.sum(dim=1) / denom
        
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
        u_override=None,
        sample_indices=None,
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

        # u for conditional A (M2)
        u = self.compute_u(prefix_stay_counts, prefix_times, prefix_mask)
        if u_override is not None:
            u = u_override.to(u.device, dtype=u.dtype)
        elif (
            self.training
            and self._random_u_labels
            and self._u_perm is not None
            and sample_indices is not None
        ):
            u = self._u_perm[sample_indices.to(self._u_perm.device)].to(u.device)

        A_eff = self.get_A_for_u(u)  # [B, z_dim, z_dim]

        # 3. Koopman 自律ロールアウト
        z = z_0
        logits_list = []
        z_list = [z_0]  # ★追加：z_0 から保持

        for _k in range(K):
            z = self._apply_koopman_step(z, A_eff)
            z_list.append(z)

            logits = self.to_logits(z)  # [B, vocab]
            logits_list.append(logits)

        pred_logits = torch.stack(logits_list, dim=1)  # [B, K, vocab]

        return {
            'pred_logits': pred_logits,
            'z_0': z_0,
            "z_traj": torch.stack(z_list, dim=1),  # ★追加: [B, K+1, z_dim]
            'aux_losses': aux_losses,
            'u': u,
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