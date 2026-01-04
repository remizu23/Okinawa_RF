import pandas as pd
import networkx as nx
import numpy as np
import random
import io

# ---------------------------------------------------------
# 1. マップとグラフの構築
# ---------------------------------------------------------
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

# データ読み込み
df_adj = pd.read_csv(io.StringIO(csv_data), index_col=0)
df_adj.columns = df_adj.columns.astype(int)

# グラフ作成（自己ループ除去）
G = nx.from_pandas_adjacency(df_adj)
G.remove_edges_from(nx.selfloop_edges(G))

# エリア定義
AREA_SHOP1 = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11]
AREA_SHOP2 = [7, 13, 14, 16, 18]

# パディングトークン定義
PADDING_TOKEN = 19
MAX_STEPS = 180  # 列数（0〜179）

# ---------------------------------------------------------
# 2. エージェントクラス定義
# ---------------------------------------------------------
class Agent:
    def __init__(self, agent_id, graph, behavior_type):
        self.id = agent_id
        self.graph = graph
        self.type = behavior_type
        self.trajectory = [] 
        
        # --- 初期化 ---
        if self.type == 'through': # 通過型
            self.current_node = random.choice([0, 3])
            self.target_node = random.choice([16, 18])
            self.stay_prob_base = 0.05
            
        elif self.type == 'stopover': # 立ち寄り型
            self.current_node = random.choice([0, 3])
            self.target_node = random.choice([2, 4, 8]) 
            self.stay_prob_base = 0.2
            self.has_stopped = False

        elif self.type == 'wander': # 回遊型
            self.current_node = random.choice(AREA_SHOP1)
            self.target_node = random.choice(AREA_SHOP2 + AREA_SHOP1)
            self.stay_prob_base = 0.15

        self.mode = 'move'
        self.stay_counter = 0

    def step(self):
        self.trajectory.append(self.current_node)

        # A. 滞在中の処理
        if self.mode == 'stay':
            self.stay_counter -= 1
            if self.stay_counter <= 0:
                self.mode = 'move'
                if self.type == 'stopover' and not self.has_stopped:
                    self.has_stopped = True
                    self.target_node = 0
            return

        # B. 滞在判定
        current_prob = self.stay_prob_base
        if self.current_node in [2, 7, 10]:
            current_prob += 0.3
        
        if random.random() < current_prob:
            self.mode = 'stay'
            self.stay_counter = random.randint(3, 8)
            return

        # C. 次の移動先決定
        neighbors = list(self.graph.neighbors(self.current_node))
        if not neighbors: return

        try:
            dists = [nx.shortest_path_length(self.graph, n, self.target_node) for n in neighbors]
            curr_dist = nx.shortest_path_length(self.graph, self.current_node, self.target_node)
        except nx.NetworkXNoPath:
            dists = [100] * len(neighbors)
            curr_dist = 100

        weights = []
        for d in dists:
            if d < curr_dist: weights.append(10.0)
            elif d == curr_dist: weights.append(1.0)
            else: weights.append(0.1)

        total_w = sum(weights)
        probs = [w/total_w for w in weights]
        next_node = np.random.choice(neighbors, p=probs)
        self.current_node = next_node

        # D. ターゲット到達時の処理
        if self.current_node == self.target_node:
            if self.type == 'wander':
                self.target_node = random.choice(AREA_SHOP1 + AREA_SHOP2)
            elif self.type == 'through':
                self.mode = 'stay' 
                self.stay_counter = 10

# ---------------------------------------------------------
# 3. データ生成とフォーマット
# ---------------------------------------------------------
def generate_large_scale_data(num_agents=2000, max_steps=180):
    """
    1エージェントあたり平均5-6個のチャンク(行)が生成されるため、
    num_agents=2000 で 約10,000〜12,000行のデータになります。
    """
    all_rows = []
    
    # 行動タイプの割合
    types = ['through'] * 30 + ['stopover'] * 30 + ['wander'] * 40
    
    print(f"Simulating {num_agents} agents...")
    
    for i in range(num_agents):
        b_type = random.choice(types)
        agent = Agent(i, G, b_type)
        
        # 1. シミュレーション実行
        for _ in range(max_steps):
            agent.step()
            
        full_traj = agent.trajectory
        
        # 2. 分断処理 (Chunking) & 分析用ID生成
        t = 0
        chunk_id = 0
        
        # 開始時間をランダムにずらす（全期間にエージェントを分散させる）
        start_offset = random.randint(0, max_steps - 20)
        
        while t < len(full_traj):
            # 10~30ステップのランダムな長さで切る
            chunk_len = random.randint(10, 30)
            
            # シミュレーション上の時刻
            global_start_t = start_offset + t
            global_end_t = min(global_start_t + chunk_len, max_steps)
            
            # 範囲外なら終了
            if global_start_t >= max_steps:
                break
            
            # 該当区間の軌跡を取得
            segment_len = global_end_t - global_start_t
            if segment_len <= 0: break
            
            segment = full_traj[t : t + segment_len]
            
            # ★分析用IDの生成 (Hashではなく可読性のあるID)
            # Format: Type_OriginalAgentID_ChunkIndex
            # 例: Through_0123_0, Stopover_0056_2
            readable_id = f"{b_type.capitalize()}_{i:05d}_{chunk_id}"
            
            # 1行分のデータを作成
            row_data = {
                'mac_index': readable_id,
                'start_t': global_start_t,
                'segment': segment
            }
            all_rows.append(row_data)
            
            # 次のチャンクへ（少し隙間を開ける＝欠損期間を作る）
            t += chunk_len + random.randint(0, 10) 
            chunk_id += 1

    # 3. ピボット処理（横持ち変換・パディング）
    print(f"Formatting {len(all_rows)} sequences...")
    output_data = []

    for row in all_rows:
        mac = row['mac_index']
        start_t = row['start_t']
        segment = row['segment']
        
        # 全期間をPadding Token (19) で初期化
        sequence = [PADDING_TOKEN] * max_steps
        
        # 該当期間に軌跡を埋め込む
        for local_t, node in enumerate(segment):
            if start_t + local_t < max_steps:
                sequence[start_t + local_t] = node
        
        # 結果リストに追加
        output_data.append([mac] + sequence)

    # DataFrame作成
    col_names = ['mac_index'] + [i for i in range(max_steps)]
    df_wide = pd.DataFrame(output_data, columns=col_names)
    
    return df_wide

# ---------------------------------------------------------
# 実行
# ---------------------------------------------------------
random.seed(42)
np.random.seed(42)

# エージェント数を2000に設定 -> 約10,000行以上のデータを生成
df_result = generate_large_scale_data(num_agents=2000, max_steps=MAX_STEPS)

# 結果確認
print(f"Generated Data Shape: {df_result.shape}")
print("\n=== Sample Data (First 5 rows) ===")
print(df_result.iloc[:5, :20]) # 最初の20列だけ表示

# IDの内訳確認（分析用）
print("\n=== Data Distribution by Type ===")
# IDの先頭文字（Through/Stopover/Wander）で集計
type_counts = df_result['mac_index'].apply(lambda x: x.split('_')[0]).value_counts()
print(type_counts)

# CSV保存
output_path = 'synthetic_input.csv'
df_result.to_csv(output_path, index=False)
print(f"\nSaved to {output_path}")