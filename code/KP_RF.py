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
        # ★追加引数
        num_agents=1,        # 全エージェント数 (IDの最大値+1)
        agent_emb_dim=16,    # エージェントIDの埋め込み次元
        max_stay_count=500,  # 滞在カウントの最大値
        stay_emb_dim=16      # 滞在カウントの埋め込み次元
    ):
        super().__init__()
        
        # 1. 埋め込み層の定義
        self.token_embedding = nn.Embedding(vocab_size, token_emb_dim, padding_idx=pad_token_id)
        
        # ★追加: ユーザーIDと滞在カウントの埋め込み
        self.agent_embedding = nn.Embedding(num_agents, agent_emb_dim)
        self.stay_embedding = nn.Embedding(max_stay_count + 1, stay_emb_dim, padding_idx=0) # 0をパディング扱いとする想定

        # ★入力ベクトルの合計次元 (Concatするため)
        total_input_dim = token_emb_dim + agent_emb_dim + stay_emb_dim
        
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
        
        # 4. Koopman Dynamics
        # ★B行列の入力次元を「合計次元」に変更 (u_t は結合ベクトルになるため)
        self.A = nn.Parameter(torch.randn(z_dim, z_dim) * 0.05)
        self.B = nn.Parameter(torch.randn(z_dim, total_input_dim) * 0.05) 

        # 5. zからの滞在カウント復元強制
        self.count_decoder = nn.Linear(z_dim, 1)

        # 6. zからの移動/滞在
        self.mode_classifier = nn.Linear(z_dim, 2) # [Stay, Move]の2値分類

        # ★追加: 特別な埋め込み層
        # 年埋め込み: 2025年用 (バイナリ的なものなので1つあればいいが、汎用的にするならEmbedding)
        # ここでは単純に「2025年フラグが立った時に足すベクトル」として定義
        # token_vec に足すため、次元は token_emb_dim に合わせる
        self.year_2025_embedding = nn.Parameter(torch.randn(1, 1, token_emb_dim) * 0.02)

        # 広場埋め込み: ノード2番用
        self.plaza_embedding = nn.Parameter(torch.randn(1, 1, token_emb_dim) * 0.02)        
        
    def forward(self, tokens, stay_counts, agent_ids, time_tensor=None): # ★引数 time_tensor 追加
        """
        tokens: [Batch, Seq]
        stay_counts: [Batch, Seq]
        agent_ids: [Batch]
        time_tensor: [Batch] (YYYYMMDDHHmm形式の整数) -> 追加
        """
        batch_size, seq_len = tokens.size()

        # --- 1. 各埋め込みの取得 ---
        token_vec = self.token_embedding(tokens)        # [B, T, token_dim]
        stay_vec = self.stay_embedding(stay_counts)     # [B, T, stay_dim]
        
        # ★★★ ここで条件付き埋め込みを加算する (u_all結合の前！) ★★★
        if time_tensor is not None:
            # A. 年の判定 (2025年かどうか)
            # time_tensor: [B] -> [B, 1, 1] に拡張
            years = time_tensor // 100000000
            is_2025 = (years == 2025).view(batch_size, 1, 1)
            
            # token_vec に 2025年埋め込みを加算
            # is_2025 が True のバッチだけ加算される
            token_vec = token_vec + self.year_2025_embedding * is_2025.float()

            # B. 広場ノードの判定 (2025年 かつ Node=2 or 21)
            # ノードID 2 (移動) と 21 (滞在: 2+50-31) を対象とする
            # tokens: [B, T] -> [B, T, 1] に拡張
            target_mask = (tokens == 2) | (tokens == 21)
            target_mask = target_mask.unsqueeze(-1) # [B, T, 1]
            
            # 条件: 「2025年」かつ「対象ノード」
            condition_plaza = is_2025.float() * target_mask.float()
            
            # token_vec に 広場埋め込みを加算
            token_vec = token_vec + self.plaza_embedding * condition_plaza
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

        # AgentIDは系列全体で共通なので拡張する
        agent_vec = self.agent_embedding(agent_ids)     # [B, agent_dim]
        agent_vec = agent_vec.unsqueeze(1).expand(-1, seq_len, -1) # [B, T, agent_dim]

        # ★結合 (Concatenate)
        # token_vec が既に修正されているため、u_all にも年/広場の情報が含まれる！
        u_all = torch.cat([token_vec, stay_vec, agent_vec], dim=-1) # [B, T, total_dim]
        
        # Koopman用の入力 u_curr (最後の時刻を除く)
        u_curr = u_all[:, :-1, :]

        # --- 2. Transformerへの入力 ---
        # u_all が修正済みなので、ここも自動的に反映される
        x = self.input_proj(u_all) # 次元圧縮 [B, T, d_model]
        x = self.pos_encoder(x)
        
        # マスク作成 (変更なし)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        pad_mask = (tokens == self.token_embedding.padding_idx).to(x.device)

        # --- 3. 状態推定 (h_t) ---
        h = self.transformer_block(
            src=x, 
            mask=causal_mask, 
            src_key_padding_mask=pad_mask
        )
        
        # --- 4. 潜在変数 (z_t) ---
        z_hat = self.to_z(h)
        z_curr = z_hat[:, :-1, :]

        # --- 5. Koopman Dynamics ---
        # z_{t+1} = A * z_t + B * u_t
        # u_curr (u_all) にも修正が反映されているので、Koopman予測も「広場がある前提」で行われる
        z_pred_next = (
            torch.einsum("ij,btj->bti", self.A, z_curr) + 
            torch.einsum("ij,btj->bti", self.B, u_curr)
        )

        logits = self.to_logits(z_hat)
        
        # 修正された u_all が返る
        return logits, z_hat, z_pred_next, u_all