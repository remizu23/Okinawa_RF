import pandas as pd
import networkx as nx
import numpy as np
import random
import io
import matplotlib.pyplot as plt
import os

# =========================================================
# 1. マップとグラフの構築
# =========================================================
csv_data = """
,0,1,2,3,4,5,6,7,8,9,10,11,13,14,16,18
0,1,1,1,0,1,0,0,0,0,0,0,1,0,0,0,0
1,1,1,1,1,1,1,0,0,0,1,0,0,0,0,0,0
2,1,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0
3,0,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0
4,1,1,0,0,1,1,0,0,1,1,0,1,0,0,0,0
5,0,0,0,1,1,1,1,0,0,1,1,0,0,0,0,0
6,0,0,0,1,0,1,1,1,0,0,0,0,0,0,0,0
7,0,0,1,1,0,0,1,1,0,0,0,0,1,1,0,0
8,0,0,0,0,1,0,0,0,1,1,0,1,0,0,0,0
9,0,1,0,0,1,0,0,0,1,1,1,1,0,0,0,0
10,0,0,0,0,0,1,1,0,0,1,1,0,1,0,0,0
11,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0
13,0,0,0,0,0,0,0,1,0,0,1,0,1,1,0,0
14,0,0,0,0,0,0,0,1,0,0,0,0,1,1,1,0
16,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1
18,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1
"""
df_adj = pd.read_csv(io.StringIO(csv_data), index_col=0)
df_adj.columns = df_adj.columns.astype(int)
G = nx.from_pandas_adjacency(df_adj)
G.remove_edges_from(nx.selfloop_edges(G))

# エリア定義
AREA_SHOP1 = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11]
AREA_SHOP2 = [7, 13, 14, 16, 18]
PAD_TOKEN = 19
MAX_STEPS = 60  # 今回は60ステップとします（長い場合はパディング）

# =========================================================
# 2. 新・エージェントロジック (状態遷移型)
# =========================================================
class AgentV2:
    def __init__(self, agent_id, graph, behavior_type):
        self.id = agent_id
        self.graph = graph
        self.type = behavior_type
        self.trajectory = []
        self.finished = False
        
        # --- 状態定義 ---
        # state: 'WALK', 'STAY', 'FINISHED'
        
        if self.type == 'through':
            # 通過型: Start -> Goal (最短経路)
            self.start_node = random.choice([0, 3, 7])
            self.goal_node = random.choice([11, 16, 18])
            self.current_node = self.start_node
            
            self.state = 'WALK'
            self.target = self.goal_node
            self.path_queue = [] # 計算済み経路バッファ
            
        elif self.type == 'stopover':
            # 立ち寄り型: Start -> Shop (Stay) -> Start
            self.start_node = random.choice([0, 7, 11, 18])
            self.shop_node = random.choice([2, 4, 8, 10])
            self.current_node = self.start_node
            
            self.state = 'WALK'
            self.target = self.shop_node
            self.phase = 'GO_TO_SHOP' # GO_TO_SHOP -> SHOPPING -> GO_HOME
            self.path_queue = []
            
        elif self.type == 'wander':
            # 回遊型: Shop -> Shop (Stay) -> Shop...
            self.current_node = random.choice(AREA_SHOP1)
            self.state = 'STAY' # 最初はいきなり滞在から入ることにする
            self.stay_counter = random.randint(3, 5)
            self.target = None
            self.path_queue = []

    def get_shortest_path_step(self, target):
        """ターゲットに向かう最短経路の次の一歩を返す"""
        try:
            path = nx.shortest_path(self.graph, self.current_node, target)
            if len(path) > 1:
                return path[1] # 次のノード
            else:
                return self.current_node # 既に到着
        except nx.NetworkXNoPath:
            return self.current_node # 動けない

    def step(self):
        # 既に終了している場合
        if self.finished:
            return PAD_TOKEN
        
        # 軌跡記録
        self.trajectory.append(self.current_node)

        # --- A. 滞在中の処理 ---
        if self.state == 'STAY':
            self.stay_counter -= 1
            if self.stay_counter <= 0:
                # 滞在終了 -> 次の行動へ
                if self.type == 'stopover':
                    # 買い物終わったら帰宅
                    self.phase = 'GO_HOME'
                    self.state = 'WALK'
                    self.target = self.start_node
                elif self.type == 'wander':
                    # 次の店へ
                    self.state = 'WALK'
                    self.target = random.choice(AREA_SHOP1 + AREA_SHOP2)
                    # 同じ場所を選んでしまったら再抽選
                    while self.target == self.current_node:
                        self.target = random.choice(AREA_SHOP1 + AREA_SHOP2)
            return self.current_node

        # --- B. 移動中の処理 ---
        if self.state == 'WALK':
            # 目的地に着いているか確認
            if self.current_node == self.target:
                # 到着時の処理
                if self.type == 'through':
                    self.finished = True # 通過完了
                    self.state = 'FINISHED'
                
                elif self.type == 'stopover':
                    if self.phase == 'GO_TO_SHOP':
                        self.state = 'STAY'
                        self.stay_counter = random.randint(5, 10) # ★必ず5ステップ以上滞在
                    elif self.phase == 'GO_HOME':
                        self.finished = True # 帰宅完了
                        self.state = 'FINISHED'
                
                elif self.type == 'wander':
                    self.state = 'STAY'
                    self.stay_counter = random.randint(5, 15) # 回遊は長めに滞在
                
                return self.current_node

            # まだ着いていないなら移動
            next_node = self.get_shortest_path_step(self.target)
            self.current_node = next_node
            return self.current_node
        
        return PAD_TOKEN

# =========================================================
# 3. データ生成と検証プロット
# =========================================================
def generate_and_verify_data(num_agents=100, max_steps=60, check_plot=True):
    all_data = []
    
    # 3タイプの比率
    types = ['through'] * 30 + ['stopover'] * 30 + ['wander'] * 40
    
    # プロット用のサンプル収集リスト
    samples = {'through': [], 'stopover': [], 'wander': []}
    
    print(f"Generating {num_agents} agents...")
    
    for i in range(num_agents):
        b_type = random.choice(types)
        agent = AgentV2(i, G, b_type)
        
        # 1エージェント分の全ステップ実行
        full_seq = []
        for _ in range(max_steps):
            node = agent.step()
            full_seq.append(node)
            
        # 完了したエージェントの処理
        # パディング(19)を除いた有効長さを計算
        valid_len = len([x for x in full_seq if x != PAD_TOKEN])
        
        # あまりに短すぎるデータ（開始即終了など）はノイズになるので除外しても良いが、
        # 今回はthroughなどは短くて当然なのでそのまま採用
        
        # Chunking処理（学習データ用）
        # ★ここ重要: 「真の経路」を細切れにする処理
        # Throughのような短い経路の場合、チャンクが1つしかできないこともある
        
        t = 0
        chunk_id = 0
        # 有効区間のみをループ対象にする
        valid_seq = full_seq[:valid_len]
        
        # 開始時間を少しバラす（全員0スタートだと不自然なので）
        start_offset = random.randint(0, 10)
        
        current_t = 0
        while current_t < len(valid_seq):
            chunk_len = random.randint(10, 20)
            end_t = min(current_t + chunk_len, len(valid_seq))
            
            segment = valid_seq[current_t : end_t]
            
            # ID生成
            # Through_0001_0 という形式にしておく
            readable_id = f"{b_type.capitalize()}_{i:04d}_{chunk_id}"
            
            # global_start_t: シミュレーション世界での絶対時刻
            global_start_t = start_offset + current_t
            
            all_data.append({
                'mac_index': readable_id,
                'start_t': global_start_t,
                'segment': segment
            })
            
            # 次のチャンクへ（少し隙間を空ける）
            current_t = end_t + random.randint(1, 5)
            chunk_id += 1

        # プロット用にサンプル保存（最初の3人だけ）
        if len(samples[b_type]) < 3:
            samples[b_type].append(full_seq)

    # --- チェック機構: データの可視化 ---
    if check_plot:
        print("\n=== Data Verification: Plotting Samples ===")
        fig, axes = plt.subplots(3, 3, figsize=(15, 10))
        fig.suptitle(f"Generated Trajectories (Max {max_steps} steps)", fontsize=16)
        
        for r, (b_type, seqs) in enumerate(samples.items()):
            for c, seq in enumerate(seqs):
                ax = axes[r, c]
                # パディングはプロットしない
                valid_seq = [x for x in seq if x != PAD_TOKEN]
                ax.plot(valid_seq, marker='o', markersize=4, linestyle='-', alpha=0.7)
                ax.set_title(f"{b_type} (len={len(valid_seq)})")
                ax.set_ylim(-1, 20) # ノード数に合わせて
                ax.grid(True, alpha=0.3)
                if r == 2: ax.set_xlabel("Time Step")
                if c == 0: ax.set_ylabel("Node ID")
        
        plt.tight_layout()
        plt.savefig("data_verification.png")
        print("Saved verification plot to 'data_verification.png'. Please check this file first.")

    # --- データフレーム化（ワイドフォーマット） ---
    output_rows = []
    for row in all_data:
        mac = row['mac_index']
        start_t = row['start_t']
        segment = row['segment']
        
        # 全期間パディング
        sequence = [PAD_TOKEN] * 180 # 学習モデルの入力長に合わせる(180)
        
        # 埋め込み
        for local_t, node in enumerate(segment):
            if start_t + local_t < 180:
                sequence[start_t + local_t] = node
        
        output_rows.append([mac] + sequence)
        
    df_wide = pd.DataFrame(output_rows, columns=['mac_index'] + list(range(180)))
    return df_wide

# =========================================================
# 実行
# =========================================================
if __name__ == "__main__":
    # シード固定
    random.seed(42)
    np.random.seed(42)
    
    # 1. データ生成とチェック
    # 学習用なので数は多めに生成 (例: 2000エージェント)
    df_result = generate_and_verify_data(num_agents=2000, max_steps=60, check_plot=True)
    
    # 2. 保存
    save_path = "synthetic_input_v2.csv"
    df_result.to_csv(save_path, index=False)
    print(f"\nSaved generated data to {save_path}")
    print(f"Total Rows: {len(df_result)}")
    
    # 確認表示
    print(df_result.head())