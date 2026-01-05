import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 既存の便利クラスはそのまま利用
class EmbeddingWithFeatures(nn.Module):
    def __init__(self, vocab_size, token_dim, feature_dim=None, feature_emb_dim=None, dropout=0.1):
        super(EmbeddingWithFeatures, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, token_dim, padding_idx=19) # padding_idx指定推奨
        
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

# --- ここが新しいモデル（ノード埋め込みがKP潜在情報を更新） ---
class KoopmanRoutesFormer(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        token_emb_dim,    # これが u_t の次元になります
        d_model,          
        nhead, 
        num_layers, 
        d_ff, 
        z_dim,            
        pad_token_id=19
    ):
        super().__init__()
        # 特徴量(feature_dim)の引数は不要になります
        
        # 1. シンプルな埋め込み層 (特徴量との結合なし)
        self.token_embedding = nn.Embedding(vocab_size, token_emb_dim, padding_idx=pad_token_id)
        self.pos_encoder = PositionalEncoding(d_model)

        # d_model と token_emb_dim が違う場合の調整用 (必要なら)
        # 今回は token_emb_dim = d_model と仮定するか、あるいは射影層を噛ませます
        if token_emb_dim != d_model:
            self.input_proj = nn.Linear(token_emb_dim, d_model)
        else:
            self.input_proj = nn.Identity()

        # 2. Transformer Block (ここを修正！)
        # Decoder-onlyモデルを作る場合でも、実装上は TransformerEncoderLayer を使います。
        # maskを与えることで、実質的にDecoderとして振る舞わせます。
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_ff, 
            batch_first=True
        )
        # self.decoder ではなく self.transformer_block とします
        self.transformer_block = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. 射影層
        self.to_z = nn.Linear(d_model, z_dim)
        self.to_logits = nn.Linear(z_dim, vocab_size) 
        
        # 4. Koopman Dynamics
        # u_t が「ノード埋め込み(token_emb_dim)」になるため、B行列の入力次元を変更

        # self.A = nn.Parameter(torch.eye(z_dim) + torch.randn(z_dim, z_dim) * 0.01) #単位行列+ノイズ に変更
        self.A = nn.Parameter(torch.randn(z_dim, z_dim) * 0.05)
        self.B = nn.Parameter(torch.randn(z_dim, token_emb_dim) * 0.05) 

    def forward(self, tokens):
        """
        tokens: [Batch, Seq] のみを受け取る (featuresは不要)
        """
        # --- 1. 埋め込みベクトルの取得 ---
        # これが「そのノード自体の意味表現」であり、今回の u_t です
        token_emb = self.token_embedding(tokens) # [Batch, Seq, token_emb_dim]
        u_curr = token_emb[:, :-1, :]            # 時刻 t の埋め込み

        # --- 2. Transformerへの入力 ---
        x = self.input_proj(token_emb) # 次元合わせ
        x = self.pos_encoder(x)
        
        # マスク作成など (省略、前と同じ)
        seq_len = tokens.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        pad_mask = (tokens == 19).to(x.device)

        # --- 3. 状態推定 (h_t) ---
        # ここも修正： memory引数は不要になります。
        # src_key_padding_mask と mask を渡すことで GPT的な動作になります。
        h = self.transformer_block(
            src=x, 
            mask=causal_mask, 
            src_key_padding_mask=pad_mask
        )
        # --- 4. 潜在変数 (z_t) ---
        z_hat = self.to_z(h)
        z_curr = z_hat[:, :-1, :]

        # --- 5. Koopman Dynamics ---
        # z_{t+1} = A * z_t + B * u_t (u_t はノード埋め込み)
        z_pred_next = (
            torch.einsum("ij,btj->bti", self.A, z_curr) + 
            torch.einsum("ij,btj->bti", self.B, u_curr)
        )

        logits = self.to_logits(z_hat)
        return logits, z_hat, z_pred_next