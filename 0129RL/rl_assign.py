# recursive logit (not time-structured) 

import pandas as pd
import numpy as np
import scipy
import os 
import time
from scipy.optimize import minimize

# base_path = "/Users/matsunagatakahiro/Desktop/jrres/okinawa_routechoice"
base_path = "/Users/matsunagatakahiro/Desktop/jrres/okinawa_mizutani"
# data loading 
# df_link = pd.read_csv(os.path.join(base_path, "data/input/ped_link.csv"))


df_link = pd.read_csv(os.path.join(base_path, "link_with_latlon.csv")) ## NEW!! 作成したリンクデータを読み込む
df_node = pd.read_csv(os.path.join(base_path, "node.csv"))
df_nw = pd.read_csv(os.path.join(base_path, 'nodebased_matrix.csv'))

# 今回choiceデータは使用しない
# df_choice = pd.read_csv(os.path.join(base_path, "data/input/1122_16-18_choice_replaced_sorted_filtered_300users.csv"))
## NEW!! 作成したdemandデータを読み込む "o" "d" "demand"の3列からなるcsvファイルを想定　ファイル名とかは適当です
df_demand = pd.read_csv("/Users/matsunagatakahiro/Desktop/jrres/okinawa_routechoice/data/input/demand_kari2.csv")
# df_demand = {}

# parameters ## 自分で推定したパラメータを順番に入れてください 今は適当に5つ入れています．
# 0.29506855  0.29506855  1.26310241 -0.0383981  -0.39769479 -0.21402674 -1.6483896   1.38917
true_param = np.array([0.29506855,  0.29506855,  1.26310241, -0.0383981,  -0.39769479, -0.21402674, -1.6483896,   1.38917]) # length, width, cross, elev, sidew
# true_param = np.array([-10, -0.01287876, -0.00615289])#, -0.5, 0.3, -0.02, 0.4]) # length, width, cross, elev, sidew
x_dim = len(true_param)

# preparations  
L = len(df_link)
N = len(df_node)
d_list = [19]# df_choice["d"].unique()
D = len(d_list)
# D = 1
# df_list = [df_choice] #[df_choice[df_choice['d'] == d_list[d]].reset_index(drop=True) for d in range(D)]
linkid_list = sorted(df_link['link_id'].unique()) # 接続行列に使用
nodeid_list = sorted(df_node['node_id'].unique())
abs_id = 19
nodeid_list.append(abs_id) # 吸収ノードを追加

print(f'nodeid_list: {nodeid_list}')
nodeid_list = [int(nid) for nid in nodeid_list] # nodeidをint型に変換
print(f'nodeid_list (int): {nodeid_list}')
# pair list 
node_index_dict = {node_id: index for index, node_id in enumerate(nodeid_list)}
pair_list = [
    (node_index_dict[df_nw.loc[i, 'k']], node_index_dict[df_nw.loc[i, 'a']])
    for i in range(len(df_nw))
]

# matrix
Ilist = []
I = np.zeros((N+1, N+1)) # stay is not allowed（not time-structured）
for i in range(len(df_nw)):
    kn = df_nw.loc[i, 'k'] # node id
    an = df_nw.loc[i, 'a']
    k = nodeid_list.index(kn) # node index
    a = nodeid_list.index(an)
    I[k, a] = 1

# 吸収リンク
I[:, -1] = 1
I[-1, :] = 0
I[-1, -1] = 1

# print(f'接続行列I:\n{I}')

# value function 
V0 = np.full((N+1, D), -1)
z0 = np.exp(V0)
Alist = []
for k in range(N+1):
    k_id = nodeid_list[k]
    a_ixs = np.where(I[k, :]==1)[0].tolist()
    a_ids = [nodeid_list[item] for item in a_ixs]
    Alist.append(a_ids)

# Alist.append([]) # 吸収ノードはどこ/とも接続しない

for d in range(D):
    d_id = d_list[d]
    d_ix = nodeid_list.index(d_id)
    # Id = I.copy()
    # for k in range(N+1):
    #     alist = Alist[k] # kと接続するnodeのnodeid
    #     if d_id in alist:
    #         Id[k, :] = 0 # このk=dからはどこにも接続しない
    #         Id[k, d_ix] = 1
    # # Id[d_ix, :] = 0 # 終端ノードからはどこにも接続しない

    # Ilist.append(Id)
    V0[d_ix, d] = 0 # 価値関数は目的地ごとに異なる
    z0[d_ix, d] = 1

V = V0
z = z0
# print(f'初期価値関数V0:\n{V0}')
# 変数行列
# node k からnode a を繋ぐlink ka の説明変数を入れていく
length_mat = np.zeros((N+1, N+1))
capacity_mat = np.zeros((N+1, N+1))
parkave_mat = np.zeros((N+1, N+1))
firstst_mat = np.zeros((N+1, N+1))
gatest_mat = np.zeros((N+1, N+1))
nationalrd_mat = np.zeros((N+1, N+1))
parking_mat = np.zeros((N+1, N+1))
plazastay_mat = np.zeros((N+1, N+1))

for i in range(len(df_nw)):
    kn = int(df_nw.loc[i, 'k']) # これはid
    an = int(df_nw.loc[i, 'a'])
    k = nodeid_list.index(kn) # node knのindex
    a = nodeid_list.index(an)
    kan = None ## 以降でif文がpassされる時，kanは未定義のまま参照されることになる．これだとダメなので最初に定義しておく．
    if not df_link[(df_link['O'] == kn) & (df_link['D'] == an)].empty:
        kan = df_link[(df_link['O'] == kn) & (df_link['D'] == an)]['link_id'].iloc[0] # リンクデータは無向リンクで考えているので双方向を考える
    if not df_link[(df_link['O'] == an) & (df_link['D'] == kn)].empty:
        kan = df_link[(df_link['O'] == an) & (df_link['D'] == kn)]['link_id'].iloc[0]
    if not kan == None:
        ka = linkid_list.index(kan) # linkidがkanのリンクのindex
        length_mat[k, a] = (df_link.loc[ka, 'length']) # リンク長
        capacity_mat[k, a] = (df_link.loc[ka, 'capacity']) # リンク容量
        parkave_mat[k, a] = (df_link.loc[ka, 'park_ave']) # 公園前通過リンクダミー
        firstst_mat[k, a] = (df_link.loc[ka, 'first_st']) # 一丁目通過リンクダミー
        gatest_mat[k, a] = (df_link.loc[ka, 'gate_st']) # ゲート通過リンクダミー
        nationalrd_mat[k, a] = (df_link.loc[ka, 'national_rd']) # 国道通過リンクダミー
        parking_mat[k, a] = (df_link.loc[ka, 'parking']) # 駐車場通過リンクダミー
        plazastay_mat[k, a] = (df_link.loc[ka, 'plaza_stay']) # プラザ通過リンクダミー
        # width_mat[k, a] = (df_link.loc[ka, 'width']) # リンク幅
        # cross_mat[k, a] = (df_link.loc[ka, 'cross_dummy']) # 横断歩道ダミー
        # elev_mat[k, a] = (df_link.loc[ka, 'elev_delta']) # 標高差
        # sidew_mat[k, a] = (df_link.loc[ka, 'sideway']) # 歩道ダミー


# instant utility matrix
def Mset(x): 
    inst = np.zeros((N+1, N+1))
    inst = np.exp(length_mat * x[0]
                + capacity_mat/1000 * x[1]
                + parkave_mat * x[2]
                + firstst_mat * x[3]
                + gatest_mat * x[4]
                + nationalrd_mat * x[5]
                + parking_mat * x[6]
                + plazastay_mat * x[7]
                # + width_mat/10 * x[1]
                # + sidew_mat * x[2]
                )
    return inst

def Vset(x): # ガンベル分布のスケールパラメタmu=1を仮定
    z = np.ones((N+1, D))
    V = np.zeros((N+1, D))
    for d in range(D):
        d_ix = nodeid_list.index(d_list[d]) # index
        z[d_ix, d] = 1
        M = np.zeros((N+1, N+1))
        B = np.zeros((N+1, 1))
        B[d_ix, 0] = 1
        Id = I
        M = Id * Mset(x) # instant utility matrix
        M[d_ix, :] = 0
        zi = z[:, d].reshape(N+1, 1) # dを目的地とする時のk→aの期待効用
        zi = (np.linalg.pinv(np.eye(N+1) - M)) @ B # ベルマン方程式を解くための逆行列計算→パラメタの組み合わせが悪いとここでダメになる
        zi = ((zi) <= 0) * 1 + ((zi) >= 0) * zi # 数値微分で負に落ちることでlogの中身が負になるのを回避
        z[:, d] = zi.ravel()
        V[:, d] = (np.log(zi)).ravel()

    return V


def assignment(x, df_link):
    res_all = np.zeros((0, 5)) # columns = ['userid', 'k', 'a', 'o', 'd']
    V = Vset(x)
    usercount = 0
    df_link_flow = df_link.copy()
    df_link_flow['flow'] = 0
    for d in range(D): # 確率がDごとに異なるのでDのループ
        d_id = d_list[d]
        d_ix = nodeid_list.index(d_id)
        z = np.exp(V[:, d]).reshape(N+1, 1)
        ZD = np.tile(z, (1, N+1))
        ZD = ZD.T

        # M = I * np.exp(length_mat/100 * x[0] +
        #           width_mat/10 * x[1] +
        #           # green_mat * x[2] +
        #           arcade_mat * x[2])
        M = I * np.exp(length_mat * x[0]
                + capacity_mat/1000 * x[1]
                + parkave_mat * x[2]
                + firstst_mat * x[3]
                + gatest_mat * x[4]
                + nationalrd_mat * x[5]
                + parking_mat * x[6]
                + plazastay_mat * x[7]
                # + width_mat/10 * x[1]
                # + sidew_mat * x[2]
                )
        M[d_ix, :] = 0

        Mz = (M @ z != 0) * (M @ z) + (M @ z == 0) * 1
        MZ = np.tile(Mz, (1, N+1))
        p = (M * ZD) / MZ

        # 目的地がdのODペアを抽出（df_demandから）
        df_demand_d = df_demand[df_demand['d'] == d_list[d]].reset_index(drop=True)
        OD_d = len(df_demand_d)
        for od in range(OD_d): # 目的地dに向かうODペアごとに実施
            o_id = df_demand_d.loc[od, "o"]

            o_ix = nodeid_list.index(int(o_id))
            for i in range(df_demand_d.loc[od, "demand"]): # 個人ごとに確率配分(個人の数だけシミュレーション)
                usercount += 1
                current_node = o_ix
                while not current_node == d_ix:
                    alist = Alist[current_node] # current_nodeと接続するnodeのnodeid
                    a_ixs = [nodeid_list.index(a_id) for a_id in alist] # current_nodeと接続するnodeのindex
                    print(f'psize={p.shape}, N={N+1}')
                    # p[current_node, :] を取り出す
                    prob_row = p[current_node, :].ravel()

                    # 合計が 1 になるように再正規化（念のため）
                    sum_p = np.sum(prob_row)

                    if sum_p > 0:
                        # 浮動小数点の誤差を防ぐため、合計で割り直して確実に 1 にする
                        prob_row = prob_row / sum_p
                        next_node = np.random.choice(range(N+1), p=prob_row)
                    else:
                        # 合計が0（どこにも行けない）場合の処理
                        # 例：目的地に到達した、あるいはネットワーク上の孤立点
                        print(f"Node {current_node} は行き止まりです。")
                        break
                    # next_node = np.random.choice([i for i in range(N)], p=p[current_node, :].ravel())
                    # increment flow on link current_node -> next_node
                    kn = nodeid_list[current_node]
                    an = nodeid_list[next_node]
                    kan = None
                    if not df_link_flow[(df_link_flow['O'] == kn) & (df_link_flow['D'] == an)].empty:
                        kan = df_link_flow[(df_link_flow['O'] == kn) & (df_link_flow['D'] == an)]['link_id'].iloc[0]
                    if not df_link_flow[(df_link_flow['O'] == an) & (df_link_flow['D'] == kn)].empty:
                        kan = df_link_flow[(df_link_flow['O'] == an) & (df_link_flow['D'] == kn)]['link_id'].iloc[0]
                    if not kan == None:
                        ka = linkid_list.index(kan)
                        df_link_flow.loc[ka, 'flow'] += 1
                    current_node = next_node

                    res_indivi = np.array([usercount, kn, an, o_id, d_id]) # userid, k, a, o, d
                    res_all = np.vstack([res_all, res_indivi])

        df_route_assigned = pd.DataFrame(res_all, columns=['userid', 'k', 'a', 'o', 'd'])
        print('route assignament finishied')
        # リンクフローの集計に加え，，一応個人ごとの経路データも返すようにしてます
        return df_link_flow, df_route_assigned 

## 配分実行部分
df_link_flow, df_route_assigned = assignment(true_param, df_link)
# csvにして保存
df_link_flow.to_csv(os.path.join(base_path, "linkflow_output.csv"), index=False)
df_route_assigned.to_csv(os.path.join(base_path, "route_assigned_output.csv"), index=False)