import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
from network import Network
#ルートはタイムステップで作るので，ルートデータはdata_num * Tの配列
#1タイムステップで滞在するか別のリンクに移動することしかできないと仮定する←ここの仮定は実データでは変更するかも

class Tokenization:
    def __init__(self, network):
        self.network = network
        self.num_nodes = network.N
        self.SPECIAL_TOKENS = {
            "<p>": self.num_nodes,  # パディングトークン
            "<e>": self.num_nodes + 1,  # 終了トークン
            "<b>": self.num_nodes + 2,  # 開始トークン
            "<m>": self.num_nodes + 3,  # 非隣接ノードトークン
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.to(self.device)

    def tokenization(self, route, mode):
        """
        route: [Batch, Seq] (固定長, 右側パディング済み)
        """
        self.route = route
        batch_size, seq_len = route.shape
        tokens = route.clone().to(self.device)
        
        # 入力データのパディングマスク (38の部分がTrue)
        # ※前提: 入力データにおいてパディングは '38' または '0' などで埋められている
        # ここではnetwork.N (38) がパディングIDと仮定
        is_padding = (tokens == self.SPECIAL_TOKENS["<p>"])

        # パディング用の列を作成
        pad_col = torch.full((batch_size, 1), self.SPECIAL_TOKENS["<p>"], device=self.device)

        if mode == "simple":
            # 入力: [start, ..., end, pad, pad]
            # 出力: [<b>, start, ..., end, pad, pad] (長さは+1されるが、末尾のpadを一つ消して長さを保つか、伸ばすか)
            # ここではシンプルに「頭に<b>をつける」処理にする
            
            # 1. 左にPadカラムを追加
            tokens = torch.cat((pad_col, tokens), dim=1) # [Batch, Seq+1]
            
            # 2. 先頭(index 0)を <b> に変える
            # 元データが左詰めなら、これで [<b>, node1, node2, ..., pad] になる
            tokens[:, 0] = self.SPECIAL_TOKENS["<b>"]
            
            # ※もし元データ自体にパディングがなくフルに入っている場合、系列長が1増える
            # 学習時はこれでも問題ない

        elif mode == "next":
            # 入力: [start, ..., end, pad, pad]
            # 出力: [start, ..., end, <e>, pad] (右にずらして<e>を入れる)
            
            # 1. 右にPadカラムを追加
            tokens = torch.cat((tokens, pad_col), dim=1) # [Batch, Seq+1]
            
            # 2. 有効データの末尾の「次」に <e> を入れる
            # is_padding: [Batch, Seq]
            # 各行で「最初のパディング位置」を探す
            # パディングがない場合は seq_len の位置
            
            # 論理反転して、右から累積和をとるなどの方法もあるが、
            # シンプルに「パディングでないものの個数」をカウントしてインデックスにする
            valid_lengths = (~is_padding).sum(dim=1) # [Batch]
            
            # <e> を挿入する位置
            eos_indices = valid_lengths
            
            # scatter_ で <e> を埋め込む
            # tokens: [Batch, Seq+1]
            # dim=1 の eos_indices の位置に <e> を入れる
            tokens.scatter_(1, eos_indices.unsqueeze(1), self.SPECIAL_TOKENS["<e>"])
            
            # 3. simpleモードと長さを合わせるために先頭を削るかどうか？
            # TransformerのTeacher Forcingでは:
            # Input: [<b>, A, B, C]
            # Target: [A, B, C, <e>]
            # というペアを作るのが一般的。
            # 上記simpleモードは [<b>, A, B, C, pad] (長さSeq+1)
            # このnextモードは [A, B, C, <e>, pad] (長さSeq+1) になっているはず
            # ただし、元tokensの先頭がAなので、そのまま使うとずれない。
            
            # inputとtargetで系列長を合わせる必要があるため、
            # simple: cat(pad, tokens) -> set 0 to <b>
            # next:   cat(tokens, pad) -> set tail to <e>
            # これで長さは共に Seq+1 になる。
            
            pass # 処理完了

        return tokens.long()


    def mask(self, mask_rate):
        mask_token_id = self.SPECIAL_TOKENS["<m>"]
        token_sequences = self.route.clone().to(self.device)
        batch_size, seq_len = token_sequences.shape

        # マスクを適用する位置を計算（1列目と最後の列を除外）
        mask_tokens = torch.rand(batch_size, seq_len) < mask_rate
        mask_tokens[:, 0] = False  # 1列目はマスクしない
        mask_tokens[:, -1] = False  # 最後の列はマスクしない

        # マスクトークンを適用
        token_sequences[mask_tokens] = mask_token_id

        num_data = len(self.route)
        new_column = torch.full((num_data, 1), self.SPECIAL_TOKENS["<b>"], device=self.device) 
        new_column2 = torch.full((num_data, 1), self.SPECIAL_TOKENS["<e>"], device=self.device) 
        token_sequences = torch.cat((token_sequences, new_column2), dim=1)
        token_sequences = torch.cat((new_column, token_sequences), dim=1)

        return token_sequences
    
    def make_feature_mat(self, token_sequences):
        token_sequences = token_sequences.long().to(self.device)
        batch_size, seq_len = token_sequences.shape
        node_features = self.network.node_features.to(self.device)
        feature_dim =  node_features.shape[1]
        special_token_features = torch.zeros((4, feature_dim), device=self.device)
        total_node_features = torch.cat((node_features, special_token_features), dim=0)
        feature_mat = torch.zeros((batch_size, seq_len, feature_dim), device=self.device)
        feature_mat = total_node_features[token_sequences]
        return feature_mat
    
    def make_VAE_input(self, token_sequences, time_index, img_dic):
        # 時間ごとの特徴量を格納
        feature_list = [img_dic[idx.item()].to(self.device) for idx in time_index]
        combined_feature_mat = torch.stack(feature_list, dim=0)
        combined_feature_mat = torch.nn.functional.pad(combined_feature_mat, (0, 0, 1, 1, 0, 1, 0, 0), mode='constant', value=0)
        

        #シーケンスの形状
        batch_size, seq_len = token_sequences.shape
        ble_to_camera = torch.tensor([
        3, 4, 6, 6, 6, 6, 6, 6, 1, 2, 0, 6, 6, 6, 6, 6, 5, 6, 6, 6, 6, 6, 6
        ], device=self.device)
        # トークンシーケンスに基づきカメラインデックスを取得
        camera_indices = ble_to_camera[token_sequences]

        #print(f'camera_indices: {camera_indices.shape}')
        #print(f'combined_feature_mat: {combined_feature_mat.shape}')
        #print("combined_feature_mat.shape:", combined_feature_mat.shape)
        #print("インデックスの範囲: ", torch.min(camera_indices), torch.max(camera_indices))  # 使用しているインデックスを確認


        feature_mat = combined_feature_mat[
        torch.arange(batch_size, device=self.device).unsqueeze(1),  # バッチ次元
        camera_indices,                                            # カメラインデックス
        torch.arange(seq_len, device=self.device).unsqueeze(0)     # 時間次元
        ]

        return feature_mat
    def make_VAE_input_sim(self, token_sequences, time_index, img_dic):
        # 時間ごとの特徴量を格納
        feature_list = [img_dic[idx.item()].to(self.device) for idx in time_index]
        combined_feature_mat = torch.stack(feature_list, dim=0)
        combined_feature_mat = torch.nn.functional.pad(combined_feature_mat, (0, 0, 1, 1, 0, 1, 0, 0), mode='constant', value=0)
        

        #シーケンスの形状
        batch_size, seq_len = token_sequences.shape
        ble_to_camera = torch.tensor([
        3, 4, 6, 6, 6, 0, 0, 6, 1, 2, 0, 6, 6, 6, 6, 6, 5, 6, 6, 6, 6, 6, 6
        ], device=self.device)
        # トークンシーケンスに基づきカメラインデックスを取得
        camera_indices = ble_to_camera[token_sequences]

        #print(f'camera_indices: {camera_indices.shape}')
        #print(f'combined_feature_mat: {combined_feature_mat.shape}')
        #print("combined_feature_mat.shape:", combined_feature_mat.shape)
        #print("インデックスの範囲: ", torch.min(camera_indices), torch.max(camera_indices))  # 使用しているインデックスを確認


        feature_mat = combined_feature_mat[
        torch.arange(batch_size, device=self.device).unsqueeze(1),  # バッチ次元
        camera_indices,                                            # カメラインデックス
        torch.arange(seq_len, device=self.device).unsqueeze(0)     # 時間次元
        ]

        return feature_mat

    #滞在カウント埋め込み

    def calculate_stay_counts(self, tokens):
        """
        トークン列から滞在カウントテンソルを作成する (CPU処理版)
        """
        batch_size, seq_len = tokens.shape
        # CPUで計算してGPUに戻す
        tokens_cpu = tokens.cpu()
        counts_cpu = torch.zeros_like(tokens_cpu)
        
        special_ids = [v for k, v in self.SPECIAL_TOKENS.items()]

        for b in range(batch_size):
            current_val = -1
            counter = 0
            for t in range(seq_len):
                val = tokens_cpu[b, t].item()
                
                # 特殊トークンはカウント0
                if val in special_ids:
                    counter = 0
                    current_val = -1
                    counts_cpu[b, t] = 0
                    continue
                
                if val == current_val:
                    counter += 1
                else:
                    counter = 1
                    current_val = val
                
                counts_cpu[b, t] = counter
        
        return counts_cpu.to(self.device)

'''
    def make_VAE_input(self, token_sequences, time_index, img_dic):
        # time_index の各要素に対応する特徴量を格納するリスト
        feature_list = []

        # time_index の各要素についてループ処理
        for idx in time_index:
            # 該当する時間の特徴量が格納されたデータを取得
            idx_value = idx.item()  # tensor -> 整数
            time_feature_mat = img_dic[idx_value].to(self.device)
            #print(time_feature_mat.size())
            if torch.isnan(time_feature_mat).any() or torch.isinf(time_feature_mat).any():
                print(f"NaN or Inf detected in time_feature_mat for {idx_value}")
                break
            #time_feature_mat = time_feature_mat[:, :, :2]

            # サイズを合わせるために，time_feature_mat の前後をゼロ埋め
            feature_dim = time_feature_mat.size(2)
            padding_mat = torch.zeros((time_feature_mat.size(0), 1, feature_dim), device=self.device)
            time_feature_mat = torch.cat((padding_mat, time_feature_mat, padding_mat), dim=1)

            # データ欠損対応ようのパディング
            padding_mat2 = torch.zeros((1, time_feature_mat.size(1), feature_dim), device=self.device)
            time_feature_mat = torch.cat((time_feature_mat, padding_mat2), dim=0)

            # リストに追加
            feature_list.append(time_feature_mat)

        # time_index に対応するすべての特徴量を結合
        # 各 time_feature_mat が同じ形状であると仮定して結合
        combined_feature_mat = torch.stack(feature_list, dim=0)
        #print(combined_feature_mat.size())

        batch_size, seq_len = token_sequences.shape
        feature_mat = torch.zeros(batch_size, seq_len, combined_feature_mat.size(3), device=self.device)
        #print(feature_mat.size())

        # カメラの行と BLE の行の対応
        ble_to_camera_dic = {
            0: 3, 1: 4, 2: 6, 3: 6, 4: 6, 5: 6, 6: 6, 7: 6, 8: 1, 9: 2, 10: 0,
            11: 6, 12: 6, 13: 6, 14: 6, 15: 6, 16: 5, 17: 6, 18: 6, 19: 6, 20: 6, 21: 6, 22: 6
        }

        # トークンシーケンスに基づいて特徴行列を埋める
        for i in range(batch_size):
            for j in range(seq_len):
                feature_mat[i, j, :] = combined_feature_mat[i, ble_to_camera_dic[token_sequences[i, j].item()], j, :]

        return feature_mat
'''