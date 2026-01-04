import torch
import pandas as pd
import numpy as np

# ==========================================
# 設定
# ==========================================
# 読み込むファイルパス
input_pt_path = '/mnt/okinawa/9月BLEデータ/route_input/network/adjacency_matrix.pt'

# 保存するCSVファイル名
output_matrix_csv = "adj_matrix_grid.csv"      # ① 行列形式（0と1の羅列）
output_edgelist_csv = "adj_matrix_edges.csv"   # ② 接続リスト（どこからどこへ）

def inspect_and_save_adj(file_path):
    print(f"Loading: {file_path}")
    
    # .ptファイルを読み込み (GPUで保存されていてもCPUで読めるように設定)
    try:
        adj_tensor = torch.load(file_path, map_location=torch.device('cpu'), weights_only=True)
    except Exception as e:
        # weights_only=Trueで失敗する場合のフォールバック
        adj_tensor = torch.load(file_path, map_location=torch.device('cpu'))

    # NumPy配列に変換
    adj_np = adj_tensor.numpy()
    
    print(f"\n=== Basic Info ===")
    print(f"Shape: {adj_np.shape}")
    print(f"Type: {adj_np.dtype}")
    print(f"Min value: {adj_np.min()}")
    print(f"Max value: {adj_np.max()}")
    
    # ------------------------------------------
    # ① 行列形式 (Grid) で保存
    # ------------------------------------------
    df_matrix = pd.DataFrame(adj_np)
    df_matrix.to_csv(output_matrix_csv, index=True, header=True)
    print(f"\n[Saved] Matrix format -> {output_matrix_csv}")
    
    # ------------------------------------------
    # ② 接続リスト (Edge List) で保存
    # ------------------------------------------
    # 値が 0 ではない（繋がっている）要素のインデックスを取得
    # 行番号が From, 列番号が To
    rows, cols = np.where(adj_np > 0)
    values = adj_np[rows, cols]
    
    df_edges = pd.DataFrame({
        'From_Node': rows,
        'To_Node': cols,
        'Weight': values
    })
    
    df_edges.to_csv(output_edgelist_csv, index=False)
    print(f"[Saved] Edge List format -> {output_edgelist_csv}")
    
    # ------------------------------------------
    # コンソール表示 (プレビュー)
    # ------------------------------------------
    print("\n=== Edge List Preview (Top 10 connections) ===")
    print(df_edges.head(10))
    
    print(f"\nTotal connections (edges): {len(df_edges)}")
    
    # ネットワークの密度（全組み合わせのうち、繋がっている割合）
    density = len(df_edges) / (adj_np.shape[0] * adj_np.shape[1])
    print(f"Graph Density: {density:.4f}")

if __name__ == "__main__":
    inspect_and_save_adj(input_pt_path)