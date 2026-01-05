import pandas as pd
import numpy as np

def create_stay_extended_data(input_df):
    """
    既存のデータフレームに対し、以下の変換を行う関数
    1. 同じトークンが連続する場合、2個目以降を +19 する (滞在状態化)
       例: [0, 0, 0, 1] -> [0, 19, 19, 1]
    2. Paddingトークンを 19 から 38 に変更する
       注: Paddingの連続は「滞在」とはみなさず、すべて 38 にする
    """
    # データのコピー（ID列を除外してnumpy配列化）
    # mac_index列が0列目にある想定
    ids = input_df.iloc[:, 0].values
    data = input_df.iloc[:, 1:].values.astype(int)
    
    # ---------------------------------------------------------
    # ロジック実装
    # ---------------------------------------------------------
    
    # 1. 前の時点のデータを取得（右に1つずらす）
    # axis=1 (横方向) にシフト
    data_shifted = np.roll(data, 1, axis=1)
    # 先頭列は比較対象がないため、ありえない値(-1)を入れておく
    data_shifted[:, 0] = -999
    
    # 2. マスク作成
    # (A) 元がPadding(19)である場所
    mask_old_pad = (data == 19)
    
    # (B) 「滞在」である場所
    # 条件: 「現在の値 == 1つ前の値」 かつ 「Paddingではない」
    # ※ Padding(19)の連続を 19+19=38, 38+19=57... としないため
    mask_stay = (data == data_shifted) & (~mask_old_pad)
    
    # 3. 値の置換
    # 先に滞在オフセットを適用 (+19)
    # 例: ノード0の滞在 -> 0 + 19 = 19
    # 例: ノード18の滞在 -> 18 + 19 = 37
    data[mask_stay] += 19
    
    # 次にPaddingを置換 (19 -> 38)
    # ※元々19だった場所はすべて38にする
    # (滞在計算で生成された19(ノード0の滞在)は、mask_old_padがFalseなので影響受けない)
    data[mask_old_pad] = 38
    
    # ---------------------------------------------------------
    # データフレーム再構築
    # ---------------------------------------------------------
    df_extended = pd.DataFrame(data, columns=input_df.columns[1:])
    df_extended.insert(0, 'mac_index', ids)
    
    return df_extended

# =========================================================
# 実行ブロック
# =========================================================
if __name__ == "__main__":
    # 1. データの読み込み
    input_file = "synthetic_input_v2.csv"
    
    try:
        print(f"Loading {input_file}...")
        df_original = pd.read_csv(input_file)
        
        # 2. 変換実行
        print("Processing stay extension...")
        df_new = create_stay_extended_data(df_original)
        
        # 3. 保存
        output_file = "synthetic_input_v3.csv"
        df_new.to_csv(output_file, index=False)
        
        # 4. 結果確認
        print(f"\nSaved to {output_file}")
        print("-" * 50)
        
       
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Please run the previous code to generate it first.")