"""
共通データ分割スクリプト

元の系列レベルで Train/Val/Test に分割し、
KoopmanとAblationの両方で使用する共通インデックスを作成します。

使い方：
    python create_common_split.py

出力：
    /home/mizutani/projects/RF/data/common_split_indices_m5.npz
    - train_sequences: Train用の系列インデックス
    - val_sequences: Val用の系列インデックス
    - test_sequences: Test用の系列インデックス
"""

import numpy as np
import os

# ========================================
# 設定
# ========================================
CONFIG = {
    "data_path": "/home/mizutani/projects/RF/data/input_real_m5long.npz",
    "output_path": "/home/mizutani/projects/RF/data/common_split_indices_m5long.npz",
    
    # 分割比率
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    
    # 乱数シード（再現性のため）
    "random_seed": 42,
}

# ========================================
# メイン処理
# ========================================

print("="*60)
print("Common Data Split - Sequence Level")
print("="*60)

# データロード
print(f"\nLoading data from: {CONFIG['data_path']}")
trip_arrz = np.load(CONFIG['data_path'])
route_arr = trip_arrz['route_arr']
num_sequences = len(route_arr)

print(f"Total sequences: {num_sequences}")

# 分割
print("\n=== Splitting ===")
indices = np.arange(num_sequences)
np.random.seed(CONFIG['random_seed'])
np.random.shuffle(indices)

train_end = int(num_sequences * CONFIG['train_ratio'])
val_end = train_end + int(num_sequences * CONFIG['val_ratio'])

train_seq_indices = indices[:train_end]
val_seq_indices = indices[train_end:val_end]
test_seq_indices = indices[val_end:]

print(f"Train sequences: {len(train_seq_indices)} ({len(train_seq_indices)/num_sequences*100:.1f}%)")
print(f"Val sequences:   {len(val_seq_indices)} ({len(val_seq_indices)/num_sequences*100:.1f}%)")
print(f"Test sequences:  {len(test_seq_indices)} ({len(test_seq_indices)/num_sequences*100:.1f}%)")

# 保存
print(f"\n=== Saving ===")
os.makedirs(os.path.dirname(CONFIG['output_path']), exist_ok=True)

np.savez(
    CONFIG['output_path'],
    train_sequences=train_seq_indices,
    val_sequences=val_seq_indices,
    test_sequences=test_seq_indices,
    config={
        'train_ratio': CONFIG['train_ratio'],
        'val_ratio': CONFIG['val_ratio'],
        'test_ratio': CONFIG['test_ratio'],
        'random_seed': CONFIG['random_seed'],
        'source_file': CONFIG['data_path'],
    }
)

print(f"Saved to: {CONFIG['output_path']}")

# 検証
print("\n=== Verification ===")
loaded = np.load(CONFIG['output_path'])
print(f"Train sequences: {len(loaded['train_sequences'])}")
print(f"Val sequences:   {len(loaded['val_sequences'])}")
print(f"Test sequences:  {len(loaded['test_sequences'])}")

# 重複チェック
all_indices = np.concatenate([
    loaded['train_sequences'],
    loaded['val_sequences'],
    loaded['test_sequences']
])
unique_indices = np.unique(all_indices)

assert len(all_indices) == len(unique_indices), "ERROR: Duplicate indices found!"
assert len(unique_indices) == num_sequences, "ERROR: Some sequences are missing!"

print("✓ No duplicates")
print("✓ All sequences accounted for")

print("\n" + "="*60)
print("Split creation complete!")
print("="*60)
print(f"\nUse this file in training scripts:")
print(f"  split_data = np.load('{CONFIG['output_path']}')")
print(f"  train_seq = split_data['train_sequences']")
print(f"  val_seq = split_data['val_sequences']")
print(f"  test_seq = split_data['test_sequences']")