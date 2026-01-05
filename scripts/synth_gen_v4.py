import pandas as pd
import networkx as nx
import numpy as np
import random
import io
import matplotlib.pyplot as plt
import os

# =========================================================
# 1. マップ構築と基本設定
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

# --- 定数定義 ---
# ノード分類
AREA_SHOP = [1, 3, 4, 5, 6, 8, 9, 10, 13, 14, 16] # 滞在可能な店
ENTRY_EXIT_NODES = [0, 2, 7, 11, 18] # 出入り口

# トークン関連
NUM_NODES = 19
STAY_OFFSET = 19   # 滞在2ステップ目以降は +19
PAD_TOKEN = 38     # パディングトークン
MAX_STEPS = 60     # 系列長上限

# --- 満足度パラメータ設定 (β) ---
# ノードごとの「単位時間あたりの満足度」を固定
# 商店街ノードには高めの値を、通路には0を設定
NODE_BETAS = {i: 0.0 for i in range(NUM_NODES)}

# ランダムシード固定で毎回同じβになるように設定
rng = np.random.default_rng(42)
for node in AREA_SHOP:
    # 0.5 ~ 1.5 の間でランダムな効用係数を付与
    NODE_BETAS[node] = round(rng.uniform(0.5, 1.5), 2)

print("=== Node Beta Values (Satisfaction per step) ===")
print({k: v for k, v in NODE_BETAS.items() if v > 0})


# =========================================================
# 2. 活動欲求型エージェント (Activity Need Agent)
# =========================================================
class AgentV3:
    def __init__(self, agent_id, graph, behavior_type):
        self.id = agent_id
        self.graph = graph
        self.type = behavior_type # 'stopover' or 'wander' ('through'は除外)
        
        self.history_nodes = []     # 生のノード履歴
        self.history_tokens = []    # トークン化された履歴
        self.finished = False
        
        # --- 活動欲求パラメータ (Satisfaction Params) ---
        # 閾値 N*: これを超えたら帰宅する
        # 系列長60に収めるため、移動時間を考慮して閾値を調整
        # 例: β平均1.0 × 滞在30ステップ = 30.0 くらいが目安
        self.satisfaction_threshold = rng.uniform(20.0, 35.0) 
        self.current_satisfaction = 0.0
        
        # --- 初期位置 ---
        self.start_node = random.choice(ENTRY_EXIT_NODES)
        self.current_node = self.start_node
        self.history_nodes.append(self.current_node)
        self.history_tokens.append(self.current_node) # 最初はそのまま
        
        # --- 状態管理 ---
        self.state = 'WALK' 
        self.target = self._choose_next_shop() # 最初の店を決める
        self.local_sat_goal = 0.0 # その店で稼ぐ満足度の目標

    def _choose_next_shop(self):
        """次の目的地の店を選ぶ（現在の満足度に応じて）"""
        # まだ閾値に達していない -> 店に行く
        candidates = [n for n in AREA_SHOP if n != self.current_node]
        if not candidates:
            return self.start_node # 候補がなければ帰る
        return random.choice(candidates)

    def _choose_exit(self):
        """帰宅するための出口を選ぶ"""
        return self.start_node # 今回は入ってきた場所に戻る設定

    def get_shortest_path_step(self, target):
        try:
            path = nx.shortest_path(self.graph, self.current_node, target)
            if len(path) > 1:
                return path[1]
            return self.current_node
        except nx.NetworkXNoPath:
            return self.current_node

    def step(self):
        if self.finished:
            return PAD_TOKEN
        
        # --- 1. 滞在中 (STAY) の処理 ---
        if self.state == 'STAY':
            # 満足度獲得 (線形ダイナミクス: z_t+1 = z_t + beta)
            beta = NODE_BETAS.get(self.current_node, 0)
            self.current_satisfaction += beta
            
            # トークン生成: 滞在継続中は (NodeID + 19)
            token = self.current_node + STAY_OFFSET
            self.history_nodes.append(self.current_node)
            self.history_tokens.append(token)
            
            # 離脱判定
            # A. 全体の満足度が閾値を超えた -> 帰宅モードへ
            if self.current_satisfaction >= self.satisfaction_threshold:
                self.state = 'WALK'
                self.target = self._choose_exit()
                # 帰宅モードフラグ（デバッグ用）
                self.phase = 'GO_HOME'
            
            # B. その店での局所目標を達成した -> 次の店へ（回遊）
            elif self.current_satisfaction >= self.local_sat_goal:
                self.state = 'WALK'
                self.target = self._choose_next_shop()
                self.phase = 'WANDER'
            
            return token

        # --- 2. 移動中 (WALK) の処理 ---
        elif self.state == 'WALK':
            # 目的地に到着しているか？
            if self.current_node == self.target:
                # 帰宅目的地だった場合 -> 終了
                if self.current_satisfaction >= self.satisfaction_threshold:
                    self.finished = True
                    self.state = 'FINISHED'
                    return PAD_TOKEN
                
                # 店だった場合 -> 滞在開始
                else:
                    self.state = 'STAY'
                    # この店でどれくらい満足度を稼ぐか決める (局所目標)
                    # 全体の残りの何割か、あるいはランダムな固定値
                    remaining = self.satisfaction_threshold - self.current_satisfaction
                    # その店で稼ぐ量をランダムに決定 (例: 残りの20%~50%)
                    gain = remaining * rng.uniform(0.3, 0.6)
                    # 最低でも1.0くらいは稼ぐ
                    gain = max(gain, 1.0)
                    self.local_sat_goal = self.current_satisfaction + gain
                    
                    # 滞在初手は通常のノードID（移動してきた扱い）
                    # ただし「到着した瞬間」は前のstepで記録済みなので、
                    # ここでは「滞在1ステップ目」として処理するが、
                    # ユーザー要望: [3, 2, 1, 1, 1] -> [3, 2, 1, 20, 20]
                    # 到着したステップ(移動完了) = 1
                    # 次のステップ(滞在継続) = 20
                    # よって、ここではstayロジックを1回回すのと同義だが、
                    # 状態遷移した直後のこのターンは「店にいる」状態の最初のトークン
                    
                    # ★ロジック調整:
                    # 到着した時点で、そのステップの出力は「到着ノードID」であるべき。
                    # しかし「移動」ロジックの中で `current_node` が更新される。
                    # 今回のstep関数は「次の状態」を出力するもの。
                    
                    # 到着したので、ここでは「滞在開始」として処理するが、
                    # トークンはまだ `current_node` (Stayオフセットなし) とするのか？
                    # 要望: 「5ステップ滞在するとき [1, 20, 20, 20, 20]」
                    # つまり最初の1回は生ID、その後+19。
                    
                    # ここでは「移動して到着した」のではなく「到着済み」の状態からの処理
                    # つまりこのターンは「滞在1回目」に相当する。
                    
                    # 満足度計算（滞在1回目）
                    beta = NODE_BETAS.get(self.current_node, 0)
                    self.current_satisfaction += beta
                    
                    # トークンは「生ID」 (滞在初手)
                    token = self.current_node
                    self.history_nodes.append(self.current_node)
                    self.history_tokens.append(token)
                    
                    return token

            # まだ着いていないなら移動
            next_node = self.get_shortest_path_step(self.target)
            self.current_node = next_node
            
            # トークン: 移動中は生ID
            self.history_nodes.append(self.current_node)
            self.history_tokens.append(self.current_node)
            
            return self.current_node
        
        return PAD_TOKEN

# =========================================================
# 3. データ生成マネージャ
# =========================================================
def generate_synthetic_data(num_agents=1000, fragmentation_ratio=0.3):
    """
    params:
      fragmentation_ratio: 断片化（Chunking）を行うエージェントの割合
                           (0.3なら30%のエージェントは断片化、70%は1本の系列)
    """
    all_data = []
    
    # Throughを除外、StopoverとWanderのみ
    types = ['stopover', 'wander']
    
    print(f"Generating {num_agents} agents with Activity Need Model...")
    print(f"Fragmentation Ratio: {fragmentation_ratio*100}%")
    
    # 統計用
    lengths = []
    
    for i in range(num_agents):
        b_type = random.choice(types)
        agent = AgentV3(i, G, b_type)
        
        # シミュレーション実行 (MAX_STEPSまで)
        # 最初の位置はinitで入っているので、残りステップ分回す
        for _ in range(MAX_STEPS - 1):
            token = agent.step()
            if token == PAD_TOKEN and agent.finished:
                break
        
        # パディング埋め (MAX_STEPSになるまで)
        full_tokens = agent.history_tokens
        valid_len = len(full_tokens)
        lengths.append(valid_len)
        
        padded_tokens = full_tokens + [PAD_TOKEN] * (MAX_STEPS - valid_len)
        padded_tokens = padded_tokens[:MAX_STEPS] # 念のためカット
        
        # --- 断片化の判定 ---
        is_fragmented = (random.random() < fragmentation_ratio)
        
        if is_fragmented:
            # --- 断片化あり (既存ロジックに近い) ---
            # 有効長さを適当な長さ(10~20)で切り出す
            t = 0
            chunk_id = 0
            start_offset = random.randint(0, 5) # 開始時刻の揺らぎ
            
            while t < valid_len:
                chunk_len = random.randint(10, 20)
                end_t = min(t + chunk_len, valid_len)
                segment = full_tokens[t : end_t]
                
                # 短すぎるゴミは捨てる
                if len(segment) > 2:
                    # パディングして保存
                    seg_padded = segment + [PAD_TOKEN] * (MAX_STEPS - len(segment))
                    
                    readable_id = f"{b_type[0].upper()}_Frag_{i:04d}_{chunk_id}"
                    all_data.append({
                        'mac_index': readable_id,
                        'sequence': seg_padded
                    })
                    chunk_id += 1
                
                t = end_t # 隙間なくつなぐか、少し空けるかは任意だが今回は連続で切る
        else:
            # --- 断片化なし (1本の完全な系列) ---
            readable_id = f"{b_type[0].upper()}_Full_{i:04d}"
            all_data.append({
                'mac_index': readable_id,
                'sequence': padded_tokens
            })

    # --- データフレーム化 ---
    # カラム名: mac_index, 0, 1, 2, ... 59
    columns = ['mac_index'] + list(range(MAX_STEPS))
    output_rows = []
    for d in all_data:
        row = [d['mac_index']] + d['sequence']
        output_rows.append(row)
        
    df_result = pd.DataFrame(output_rows, columns=columns)
    
    print(f"Average Route Length: {np.mean(lengths):.2f} steps")
    print(f"Max Route Length: {np.max(lengths)} steps")
    
    return df_result

# =========================================================
# 実行ブロック
# =========================================================
if __name__ == "__main__":
    # 1. 生成
    # データ数多め、断片化率30%
    df_out = generate_synthetic_data(num_agents=4000, fragmentation_ratio=0)
    
    # 2. 保存
    save_path = "synthetic_input_v4.csv"
    df_out.to_csv(save_path, index=False)
    
    print(f"\nSaved to {save_path}")
    print(f"Total Rows: {len(df_out)}")
    print(df_out.head())
    
    # 3. 簡易検証プロット（最初の3つだけ）
    # トークンIDのままプロットすると滞在が跳ねて見えるので確認しやすい
    print("\nPlotting sample trajectories (Token IDs)...")
    plt.figure(figsize=(12, 4))
    for idx in range(min(3, len(df_out))):
        seq = df_out.iloc[idx, 1:].values
        valid_seq = [x for x in seq if x != PAD_TOKEN]
        plt.plot(valid_seq, marker='o', label=df_out.iloc[idx, 0])
    
    plt.title("Sample Token Trajectories (Jump to >19 means Staying)")
    plt.xlabel("Step")
    plt.ylabel("Token ID")
    plt.legend()
    plt.grid(True)
    plt.savefig("check_trajectory_v4.png")
