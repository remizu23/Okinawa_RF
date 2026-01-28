"""
Transformer Ablation Model
純粋な自己回帰Transformer（Koopman演算子なし）

KP_RF_modifiedとの比較用Baseline
- 同じ入力形式（token, stay_count, agent, holiday, timezone, event）
- TransformerでPrefixをエンコード
- 自己回帰で1ステップずつ予測
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class TransformerAblation(nn.Module):
    """
    純粋な自己回帰Transformer（Ablation版）
    
    KP_RFとの違い：
    - Koopman演算子（A行列）なし
    - 毎ステップTransformerを通して予測
    - 自律ロールアウトなし
    """
    
    def __init__(
        self,
        vocab_size,
        token_emb_dim,
        d_model,
        nhead,
        num_layers,
        d_ff,
        pad_token_id=38,
        dist_mat_base: torch.Tensor | None = None,
        base_N: int | None = None,
        num_agents=1,
        agent_emb_dim=16,
        max_stay_count=500,
        stay_emb_dim=16,
        holiday_emb_dim=4,
        time_zone_emb_dim=4,
        event_emb_dim=4,
    ):
        super().__init__()
        
        # Embeddings（KP_RFと同じ）
        self.token_embedding = nn.Embedding(vocab_size, token_emb_dim, padding_idx=pad_token_id)
        self.agent_embedding = nn.Embedding(num_agents, agent_emb_dim)
        self.stay_embedding = nn.Embedding(max_stay_count + 1, stay_emb_dim, padding_idx=0)
        self.holiday_embedding = nn.Embedding(2, holiday_emb_dim)
        self.time_zone_embedding = nn.Embedding(2, time_zone_emb_dim)
        self.event_embedding = nn.Embedding(2, event_emb_dim)
        
        total_input_dim = (token_emb_dim + agent_emb_dim + stay_emb_dim + 
                           holiday_emb_dim + time_zone_emb_dim + event_emb_dim)
        
        self.input_proj = nn.Linear(total_input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head
        self.to_logits = nn.Linear(d_model, vocab_size)
        
        # Config
        self.base_N = int(base_N) if base_N is not None else int(pad_token_id) // 2
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_stay_count = max_stay_count
        
        # 地理的距離損失用（オプション）
        if dist_mat_base is not None:
            if not isinstance(dist_mat_base, torch.Tensor):
                dist_mat_base = torch.tensor(dist_mat_base)
            self.register_buffer("dist_mat_base", dist_mat_base.long())
            
            candidate_ids = torch.arange(vocab_size, dtype=torch.long)
            is_node_token = candidate_ids < pad_token_id
            candidate_base = torch.zeros_like(candidate_ids)
            candidate_base[is_node_token] = candidate_ids[is_node_token] % self.base_N
            
            self.register_buffer("candidate_base_id", candidate_base)
        else:
            self.dist_mat_base = None
            self.candidate_base_id = None
    
    def _embed_sequence(
        self,
        tokens,           # [B, T]
        stay_counts,      # [B, T]
        agent_ids,        # [B] or [B, T]
        holidays,         # [B, T]
        time_zones,       # [B, T]
        events,           # [B, T]
    ):
        """
        系列を埋め込みベクトルに変換
        """
        batch_size, seq_len = tokens.size()
        
        # Token埋め込み
        token_vec = self.token_embedding(tokens)
        stay_vec = self.stay_embedding(stay_counts)
        
        # Agent埋め込み
        if agent_ids.dim() == 1:
            agent_vec = self.agent_embedding(agent_ids)
            agent_vec = agent_vec.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            agent_vec = self.agent_embedding(agent_ids)
        
        holiday_vec = self.holiday_embedding(holidays)
        timezone_vec = self.time_zone_embedding(time_zones)
        event_vec = self.event_embedding(events)
        
        # 結合
        u_all = torch.cat([
            token_vec, stay_vec, agent_vec,
            holiday_vec, timezone_vec, event_vec
        ], dim=-1)
        
        return u_all
    
    def forward(
        self,
        tokens,           # [B, T]
        stay_counts,      # [B, T]
        agent_ids,        # [B] or [B, T]
        holidays,         # [B, T]
        time_zones,       # [B, T]
        events,           # [B, T]
        mask=None,        # [B, T] (True=padding)
    ):
        """
        順伝播（学習時）
        
        Returns:
            logits: [B, T, vocab_size] - 次トークン予測
        """
        batch_size, seq_len = tokens.size()
        
        # 埋め込み
        x = self._embed_sequence(tokens, stay_counts, agent_ids, holidays, time_zones, events)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        # Causal mask（自己回帰用）
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Padding mask
        if mask is None:
            pad_mask = (tokens == self.pad_token_id).to(x.device)
        else:
            pad_mask = mask.to(x.device)
        
        # Transformer
        h = self.transformer(
            src=x,
            mask=causal_mask,
            src_key_padding_mask=pad_mask
        )  # [B, T, d_model]
        
        # 次トークン予測
        logits = self.to_logits(h)  # [B, T, vocab_size]
        
        return logits
    
    def forward_with_prefix_future(
        self,
        prefix_tokens,
        prefix_stay_counts,
        prefix_agent_ids,
        prefix_holidays,
        prefix_time_zones,
        prefix_events,
        future_tokens,
        future_stay_counts,
        future_holidays,
        future_time_zones,
        future_events,
        prefix_mask=None,
        future_mask=None,
    ):
        """
        Prefix + Future を結合してforwardパス（学習時）
        
        Returns:
            dict with:
                - logits: [B, T_total, vocab_size]
                - prefix_len: int
        """
        # Prefix と Future を結合
        tokens = torch.cat([prefix_tokens, future_tokens], dim=1)
        stay_counts = torch.cat([prefix_stay_counts, future_stay_counts], dim=1)
        holidays = torch.cat([prefix_holidays, future_holidays], dim=1)
        time_zones = torch.cat([prefix_time_zones, future_time_zones], dim=1)
        events = torch.cat([prefix_events, future_events], dim=1)
        
        if prefix_mask is not None and future_mask is not None:
            mask = torch.cat([prefix_mask, future_mask], dim=1)
        else:
            mask = None
        
        # Forward
        logits = self.forward(
            tokens, stay_counts, prefix_agent_ids,
            holidays, time_zones, events, mask
        )
        
        return {
            'logits': logits,
            'prefix_len': prefix_tokens.size(1),
        }
    
    def calc_geo_loss(self, logits, targets):
        """
        地理的距離損失（KP_RFと同じ）
        """
        if not hasattr(self, 'candidate_base_id') or self.candidate_base_id is None:
            return torch.tensor(0.0, device=logits.device)
        
        probs = F.softmax(logits, dim=-1)  # [B, T, V]
        
        target_base_nodes = self.candidate_base_id[targets]  # [B, T]
        dists_to_nodes = self.dist_mat_base[target_base_nodes]
        dists_to_tokens = dists_to_nodes[:, :, self.candidate_base_id]  # [B, T, V]
        
        expected_dist = torch.sum(probs * dists_to_tokens, dim=-1)
        
        valid_mask = (targets < self.base_N * 2)
        loss = (expected_dist * valid_mask.float()).sum() / (valid_mask.sum() + 1e-6)
        
        return loss
    
    def generate_next_token(
        self,
        current_sequence,      # [B, T]
        current_stay_counts,   # [B, T]
        agent_ids,             # [B]
        holidays,              # [B, T]
        time_zones,            # [B, T]
        events,                # [B, T]
    ):
        """
        1ステップ予測（自己回帰生成用）
        
        Returns:
            logits: [B, vocab_size]
        """
        logits = self.forward(
            current_sequence,
            current_stay_counts,
            agent_ids,
            holidays,
            time_zones,
            events,
        )  # [B, T, vocab_size]
        
        # 最後のステップの予測を取得
        next_logits = logits[:, -1, :]  # [B, vocab_size]
        
        return next_logits


if __name__ == "__main__":
    # 簡単な動作確認
    print("=== Transformer Ablation Model Test ===")
    
    model = TransformerAblation(
        vocab_size=39,
        token_emb_dim=32,
        d_model=64,
        nhead=4,
        num_layers=2,
        d_ff=128,
    )
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ダミーデータ
    batch_size = 4
    seq_len = 10
    
    tokens = torch.randint(0, 38, (batch_size, seq_len))
    stay_counts = torch.randint(0, 5, (batch_size, seq_len))
    agent_ids = torch.zeros(batch_size, dtype=torch.long)
    holidays = torch.randint(0, 2, (batch_size, seq_len))
    time_zones = torch.randint(0, 2, (batch_size, seq_len))
    events = torch.randint(0, 2, (batch_size, seq_len))
    
    # Forward
    logits = model(tokens, stay_counts, agent_ids, holidays, time_zones, events)
    print(f"Output logits shape: {logits.shape}")  # Should be [4, 10, 39]
    
    # 1ステップ生成
    next_logits = model.generate_next_token(
        tokens, stay_counts, agent_ids, holidays, time_zones, events
    )
    print(f"Next token logits shape: {next_logits.shape}")  # Should be [4, 39]
    
    print("✓ Test passed!")