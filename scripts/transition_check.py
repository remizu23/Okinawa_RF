import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# ==========================================
# 設定
# ==========================================
# 解析したいファイル（推論結果 または 正解データ）
input_csv_path = "/home/mizutani/projects/RF/runs/20251216_230505/result_ordinary_20251216_230505.csv" 
# input_csv_path = "teacher_xxxx.csv"

# 出力ファイル名
output_csv_path = "transition_counts.csv"

# 無視するトークン（パディングや特殊トークン）
# ネットワークのノード数 N=19 と仮定した場合の設定です。環境に合わせて変更してください。
PAD_TOKEN = 19
START_TOKEN = 21 # <b>
END_TOKEN = 20   # <e>
MASK_TOKEN = 22  # <m>

IGNORE_TOKENS = {PAD_TOKEN, START_TOKEN, END_TOKEN, MASK_TOKEN}

# 同じ場所への滞在（A -> A）をカウントに含めるか？
# True: 移動のみ（A -> B）をカウント / False: 滞在（A -> A）もカウント
EXCLUDE_SELF_LOOPS = False  

def analyze_transitions(file_path):
    print(f"Loading: {file_path}")
    df = pd.read_csv(file_path)
    
    transitions = []

    # 1行（1ルート）ずつ処理
    for _, row in df.iterrows():
        seq = row.values
        
        # 時系列順にペアを見ていく (t, t+1)
        for i in range(len(seq) - 1):
            from_node = int(seq[i])
            to_node = int(seq[i+1])
            
            # 特殊トークンが含まれていたらスキップ
            if from_node in IGNORE_TOKENS or to_node in IGNORE_TOKENS:
                continue
            
            # 自己ループ（滞在）を除外する場合
            if EXCLUDE_SELF_LOOPS and (from_node == to_node):
                continue
            
            transitions.append((from_node, to_node))

    # 集計
    counts = Counter(transitions)
    
    # DataFrame化
    result_df = pd.DataFrame(counts.items(), columns=['Transition', 'Count'])
    result_df['From'] = result_df['Transition'].apply(lambda x: x[0])
    result_df['To'] = result_df['Transition'].apply(lambda x: x[1])
    result_df = result_df[['From', 'To', 'Count']]
    
    # カウントの多い順にソート
    result_df = result_df.sort_values(by='Count', ascending=False)
    
    return result_df

def plot_heatmap(df_counts):
    """
    遷移回数をヒートマップとして表示・保存する
    """
    if df_counts.empty:
        print("有効な遷移データがありません。")
        return

    # ピボットテーブル作成 (行: From, 列: To, 値: Count)
    pivot_df = df_counts.pivot(index='From', columns='To', values='Count').fillna(0)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_df, annot=True, fmt='g', cmap='viridis', cbar_kws={'label': 'Transition Count'})
    plt.title('Node Transition Heatmap')
    plt.xlabel('To Node')
    plt.ylabel('From Node')
    
    save_img = "transition_heatmap.png"
    plt.savefig(save_img)
    print(f"ヒートマップを保存しました: {save_img}")
    plt.show()

if __name__ == "__main__":
    # 解析実行
    df_result = analyze_transitions(input_csv_path)
    
    if not df_result.empty:
        # CSV保存
        df_result.to_csv(output_csv_path, index=False)
        print(f"集計結果をCSV保存しました: {output_csv_path}")
        
        # 上位10件を表示
        print("\n=== Top 10 Transitions ===")
        print(df_result.head(10))
        
        # ヒートマップ描画
        plot_heatmap(df_result)
    else:
        print("集計対象となる移動が見つかりませんでした。IGNORE_TOKENSの設定を確認してください。")