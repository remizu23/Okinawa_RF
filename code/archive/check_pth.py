import torch

# 1. ファイルパスを指定
pth_path = "/home/mizutani/projects/RF/runs/20260127_014201/model_weights_20260127_014201.pth"  # ここを実際のファイルパスに変更

# 2. ロードする
# ※ map_location='cpu' を指定すると、GPUで学習したモデルもCPUだけで開けます
checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)

print("=== Keys in checkpoint ===")
print(checkpoint.keys())
# 出力例: dict_keys(['model_state_dict', 'optimizer_state_dict', 'config', 'history', 'split_sequences'])

# -----------------------------------------
# 3. 各データの参照方法
# -----------------------------------------

# ■ Config（学習時の設定）を見る
if 'config' in checkpoint:
    config = checkpoint['config']
    print("\n=== Config ===")
    print(config)
    
    # 特定の値を取り出す例
    z_dim = config.get('z_dim', 'Not Found')
    print(f"z_dim: {z_dim}")

# ■ History（Lossの推移）を見る
if 'history' in checkpoint:
    history = checkpoint['history']
    print("\n=== History Keys ===")
    print(history.keys())
    
    # Lossの最後の値を確認する例
    if 'train_loss' in history and len(history['train_loss']) > 0:
        print(f"Final Train Loss: {history['train_loss'][-1]}")
        print(f"Final Val Loss: {history['val_loss'][-1]}")
    
    # グラフを描画したい場合（matplotlibが必要）
    # import matplotlib.pyplot as plt
    # plt.plot(history['train_loss'], label='Train')
    # plt.plot(history['val_loss'], label='Val')
    # plt.legend()
    # plt.show()

# ■ Split Sequences（データ分割インデックス）を見る
if 'split_sequences' in checkpoint:
    splits = checkpoint['split_sequences']
    print("\n=== Split Info ===")
    print(f"Train indices count: {len(splits['train'])}")
    print(f"Val indices count: {len(splits['val'])}")
    print(f"Test indices count: {len(splits['test'])}")
    # 必要ならここからインデックスを取り出して再利用できます
    # train_indices = splits['train']

# ■ Model Weights（重みパラメータ）
# 中身は巨大なテンソル辞書なので、形状だけ確認するのが無難です
if 'model_state_dict' in checkpoint:
    print("\n=== Model State Dict ===")
    for key, tensor in list(checkpoint['model_state_dict'].items())[:3]: # 最初3つだけ表示
        print(f"{key}: {tensor.shape}")