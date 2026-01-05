import torch
import pandas as pd
import numpy as np
import utils.logger as log


class Network:
    def __init__(self, adj_matrix, node_features):
        log.logger.info('Network init')
        self.adj_matrix = adj_matrix
        self.node_features = node_features
        # self.df_penalty = df_penalty

        self.N = len(self.adj_matrix)
        self.node_id_list = [i for i in range(self.N)]

        #効用関数に入れる変数の行列を作成する
        # 説明変数4つの行列を挿入する
        self.feature_mat_0 = self._node_feature_matrix(feature_num = 0) 
        #self.feature_mat_1 = self._node_feature_matrix(feature_num = 1)

        

    def _node_feature_matrix(self, feature_num):
        part_feature_mat = np.zeros((self.N, self.N))
        feature_vec = self.node_features[:, feature_num]
        for j in range(self.N): # 移動先がアーケードあるか
            part_feature_mat[j, :] = feature_vec 

        feature_mat = np.tile(part_feature_mat, (1, 1)).reshape((self.N, self.N))
        return feature_mat
    
def expand_adjacency_matrix(original_adj_matrix):
    """
    隣接行列を「移動」と「滞在」用に2倍に拡張する関数
    """
    N = original_adj_matrix.shape[0]
    device = original_adj_matrix.device # デバイスを合わせる
    
    # 1. Move -> Move (元の接続関係)
    block_mm = original_adj_matrix.clone()
    
    # 2. Move -> Stay (自分自身の滞在IDへ遷移)
    block_ms = torch.eye(N, device=device)
    
    # 3. Stay -> Move (滞在から隣のノードへ移動再開)
    block_sm = original_adj_matrix.clone()
    
    # 4. Stay -> Stay (滞在継続)
    block_ss = torch.eye(N, device=device)
    
    # 結合
    top_row = torch.cat((block_mm, block_ms), dim=1)
    bottom_row = torch.cat((block_sm, block_ss), dim=1)
    expanded_adj = torch.cat((top_row, bottom_row), dim=0)
    
    return expanded_adj