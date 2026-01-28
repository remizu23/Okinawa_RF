"""
Prism-RL Path Generator

Prism-constrained Recursive Logit モデルによる経路生成
学習済みパラメータを用いて O→D の最尤経路を生成

"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import networkx as nx


class PrismRLGenerator:
    """
    Prism-RL経路生成クラス
    
    choice stage t ごとに価値関数 V^d(t, a) を計算し、
    各ステップで最も確率が高い行動を選択（最尤経路）
    """
    
    def __init__(self, 
                 link_df: pd.DataFrame,
                 node_df: pd.DataFrame,
                 params: Dict[str, float],
                 tau: float = 3.0,
                 J: int = 3):
        """
        Args:
            link_df: リンクデータ (link_id, O, D, length, park_ave, ...)
            node_df: ノードデータ (node_id, x, y)
            params: 推定済みパラメータ {'beta_length': -0.0861, ...}
            tau: Detour rate (論文の τ)
            J: 最小 choice stage 制約
        """
        self.link_df = link_df
        self.node_df = node_df
        self.params = params
        self.tau = tau
        self.J = J
        
        # ネットワーク構築
        self._build_network()
        
        # 最短距離行列を事前計算
        self._compute_shortest_paths()
        
        print(f"Prism-RL Generator initialized:")
        print(f"  Nodes: {len(self.nodes)}")
        print(f"  Links: {len(self.link_df)}")
        print(f"  Parameters: {params}")
        print(f"  Detour rate τ: {tau}")
        print(f"  Min choice stage J: {J}")
    
    def _build_network(self):
        """ネットワーク構築"""
        # ノードリスト
        self.nodes = sorted(self.node_df['node_id'].unique())
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        
        # リンクの隣接リスト: node -> [(next_node, link_data), ...]
        self.adjacency = {node: [] for node in self.nodes}
        
        for _, row in self.link_df.iterrows():
            O, D = row['O'], row['D']
            self.adjacency[O].append((D, row))
        
        # NetworkX グラフ（最短距離計算用）
        self.G = nx.Graph()
        for node in self.nodes:
            self.G.add_node(node)
        
        for _, row in self.link_df.iterrows():
            O, D = row['O'], row['D']
            # 自己ループは最短距離計算では無視
            if O != D:
                self.G.add_edge(O, D)
    
    def _compute_shortest_paths(self):
        """全ノード間の最短距離を計算"""
        print("Computing shortest path distances...")
        
        self.d_min = {}
        for node in self.nodes:
            lengths = nx.single_source_shortest_path_length(self.G, node)
            self.d_min[node] = lengths
        
        print("  Done.")
    
    def _calc_utility(self, link_data: pd.Series) -> float:
        """
        リンクの効用を計算
        
        Args:
            link_data: リンクの属性データ
        
        Returns:
            utility: 効用値
        """
        utility = 0.0
        
        for var_name, beta in self.params.items():
            # beta_length -> length
            var_col = var_name.replace('beta_', '')
            
            if var_col in link_data:
                utility += beta * link_data[var_col]
        
        return utility
    
    def _get_T_d(self, origin: int, destination: int) -> int:
        """
        choice stage constraint T_d を計算
        
        T_d = max(τ * d_min(o, d), J)
        
        Args:
            origin: 起点ノード
            destination: 終点ノード
        
        Returns:
            T_d: choice stage 制約
        """
        d_min_od = self.d_min[origin].get(destination, 0)
        T_d = max(int(self.tau * d_min_od), self.J)
        return T_d
    
    def _check_prism_constraint(self, t: int, node: int, destination: int, T_d: int) -> bool:
        """
        Prism制約をチェック
        
        Δ_d(t, a) = 1 if d_min(a, d) <= T_d - t else 0
        
        Args:
            t: 現在の choice stage
            node: 現在のノード
            destination: 目的地
            T_d: choice stage 制約
        
        Returns:
            True if prism内, False otherwise
        """
        d_min_nd = self.d_min[node].get(destination, float('inf'))
        return d_min_nd <= (T_d - t)
    
    def _compute_value_function(self, destination: int, T_d: int) -> Dict[Tuple[int, int], float]:
        """
        価値関数 V^d(t, a) を計算（後ろ向き計算）
        
        V^d(t, a) = log Σ_{a' ∈ A(a)} I_d(t, a'|a) * exp(v(a'|a) + V^d(t+1, a'))
        
        Args:
            destination: 目的地ノード
            T_d: choice stage 制約
        
        Returns:
            V: {(t, node): value} の辞書
        """
        V = {}
        
        # 初期化: V(T_d, d) = 0
        V[(T_d, destination)] = 0.0
        
        # 後ろ向き計算: t = T_d-1, ..., 0
        for t in range(T_d - 1, -1, -1):
            for node in self.nodes:
                # Prism制約チェック
                if not self._check_prism_constraint(t, node, destination, T_d):
                    continue
                
                # 目的地に到達している場合
                if node == destination:
                    V[(t, node)] = 0.0
                    continue
                
                # 次ステップへの遷移を計算
                sum_exp = 0.0
                
                for next_node, link_data in self.adjacency[node]:
                    # 次状態のPrism制約チェック
                    if not self._check_prism_constraint(t + 1, next_node, destination, T_d):
                        continue
                    
                    # 効用計算
                    utility = self._calc_utility(link_data)
                    
                    # 次状態の価値関数
                    V_next = V.get((t + 1, next_node), 0.0)
                    
                    # exp(v + V)
                    sum_exp += np.exp(utility + V_next)
                
                # V(t, a) = log(Σ exp(...))
                if sum_exp > 0:
                    V[(t, node)] = np.log(sum_exp)
                else:
                    # 遷移先がない場合（到達不可能）
                    V[(t, node)] = -np.inf
        
        return V
    
    def generate_path(self, 
                     origin: int, 
                     destination: int,
                     max_steps: Optional[int] = None) -> List[int]:
        """
        O→D の最尤経路を生成
        
        各ステップで最も確率が高い行動を選択（Greedy decoding）
        
        Args:
            origin: 起点ノード
            destination: 終点ノード
            max_steps: 最大ステップ数（Noneなら到達まで）
        
        Returns:
            path: ノードのリスト [origin, ..., destination]
        """
        # choice stage 制約
        T_d = self._get_T_d(origin, destination)
        
        # 価値関数を計算
        V = self._compute_value_function(destination, T_d)
        
        # 経路生成（前向き）
        path = [origin]
        current_node = origin
        t = 0
        
        while current_node != destination:
            # 最大ステップ数チェック
            if max_steps is not None and len(path) - 1 >= max_steps:
                break
            
            # choice stage 制約チェック
            if t >= T_d:
                break
            
            # 次の行動を選択
            best_next = None
            best_prob = -np.inf
            
            for next_node, link_data in self.adjacency[current_node]:
                # Prism制約チェック
                if not self._check_prism_constraint(t + 1, next_node, destination, T_d):
                    continue
                
                # 遷移確率を計算
                utility = self._calc_utility(link_data)
                V_next = V.get((t + 1, next_node), -np.inf)
                
                # log P = v + V_next - V_current
                log_prob = utility + V_next - V.get((t, current_node), 0.0)
                
                if log_prob > best_prob:
                    best_prob = log_prob
                    best_next = next_node
            
            # 次ノードが見つからない場合（到達不可能）
            if best_next is None:
                break
            
            # 経路に追加
            path.append(best_next)
            current_node = best_next
            t += 1
        
        return path


def load_prism_rl_params(csv_path: str) -> Dict[str, float]:
    """
    推定済みパラメータをCSVから読み込み
    
    Args:
        csv_path: PrismRL_est_xxx.csv のパス
    
    Returns:
        params: {'beta_length': -0.0861, ...}
    """
    df = pd.read_csv(csv_path, index_col=0)
    
    # beta_* の列を抽出
    params = {}
    for col in df.columns:
        if col.startswith('beta_'):
            params[col] = df[col].iloc[0]
    
    return params


if __name__ == "__main__":
    # テスト
    print("="*60)
    print("Prism-RL Generator Test")
    print("="*60)
    
    # データ読み込み
    link_df = pd.read_csv('link.csv')
    node_df = pd.read_csv('node.csv')
    
    # パラメータ読み込み
    params = load_prism_rl_params('PrismRL_est_test_20260128T2339.csv')
    
    # Generator初期化
    generator = PrismRLGenerator(link_df, node_df, params, tau=3.0, J=3)
    
    # テスト経路生成
    origin = 0
    destination = 10
    
    print(f"\nGenerating path from {origin} to {destination}...")
    path = generator.generate_path(origin, destination)
    print(f"Generated path: {path}")
    print(f"Path length: {len(path) - 1} steps")