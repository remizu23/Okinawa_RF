# recursive logit (not time-structured) 

import pandas as pd
import numpy as np
import scipy
import os 
import time
from scipy.optimize import minimize
import sys
# base_path = "/Users/matsunagatakahiro/Desktop/jrres/okinawa_routechoice"
base_path = "/Users/matsunagatakahiro/Desktop/jrres/okinawa_mizutani"

# data loading 
# df_link = pd.read_csv(os.path.join(base_path, "data/input/ped_link.csv"))

df_link = pd.read_csv(os.path.join(base_path, "link_with_latlon.csv")) ## NEW!! 作成したリンクデータを読み込む
df_node = pd.read_csv(os.path.join(base_path, "node.csv"))
df_nw = pd.read_csv(os.path.join(base_path, 'nodebased_matrix.csv'))
df_choice = pd.read_csv(os.path.join(base_path, "synthetic_data.csv"))
df_choice = df_choice.merge(
    df_link[["link_id", "O", "D"]],
    on="link_id",
    how="left"
)

# data loading 
        # df_link = pd.read_csv(os.path.join(base_path, "data/input/ped_link.csv"))
        # df_node = pd.read_csv(os.path.join(base_path, "data/input/ped_node.csv"))
        # df_nw = pd.read_csv(os.path.join(base_path, 'data/input/ped_nodebased_matrix.csv'))
# df_choice = pd.read_csv(os.path.join(base_path, "data/input/1122_16-18_choice_replaced_sorted_filtered_300users__.csv"))


# preparations  
L = len(df_link)
N = len(df_node)
d_list = [19]# df_choice["d"].unique()
D = len(d_list)
# D = 1
df_list = [df_choice] #[df_choice[df_choice['d'] == d_list[d]].reset_index(drop=True) for d in range(D)]
linkid_list = sorted(df_link['link_id'].unique()) # 接続行列に使用
nodeid_list = sorted(df_node['node_id'].unique())
abs_id = 19
nodeid_list.append(abs_id) # 吸収ノードを追加

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
                # + parkave_mat * x[2]
                # + firstst_mat * x[3]
                # + gatest_mat * x[4]
                # + nationalrd_mat * x[5]
                + parking_mat * x[2]
                # + plazastay_mat * x[3]
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


def loglikelihood(x):
    LL = 0
    V = Vset(x)

    for d in range(D): # ODごとに実施
        dfd = df_list[d]
        d_id = d_list[d]
        d_ix = nodeid_list.index(d_id)
        Id = I #Ilist[d]

        z = np.exp(V[:, d]).reshape(N+1, 1)
        ZD = np.tile(z, (1, N+1))
        ZD = ZD.T

        M = Id * np.exp(length_mat * x[0]
                + capacity_mat/1000 * x[1]
                # + parkave_mat * x[2]
                # + firstst_mat * x[3]
                # + gatest_mat * x[4]
                # + nationalrd_mat * x[5]
                + parking_mat * x[2]
                # + plazastay_mat * x[3]
                # + width_mat/10 * x[1]
                # + sidew_mat * x[2]
                )
        M[d_ix, :] = 0

        Mz = (M @ z != 0) * (M @ z) + (M @ z == 0) * 1
        MZ = np.tile(Mz, (1, N+1))
        p = (M * ZD) / MZ
        # print(f'目的地node {d_id} に対する確率行列p:\n{p}') # 確認用でプリント
        # sys.exit() # 確認用でプリント
        # knode = dfd[dfd['link_id'] == df_link['link_id']]['O'].tolist()
        # anode = dfd[dfd['link_id'] == df_link['link_id']]['D'].tolist()
        knode = dfd['O']
        anode = dfd['D']
        # print(f'knode: {knode.tolist()}') # 確認用でプリント
        # print(f'anode: {anode.tolist()}') # 確認用で
        # sys.exit() # 確認用でプリント
        for l in range(len(dfd)):
            pka = 0
            if anode[l] == d_id:
                pka = 1
            else:
                k = nodeid_list.index(knode[l])
                a = nodeid_list.index(anode[l])
                pka = p[k, a]
            pka = (pka == 0) * 1 + (pka != 0) * pka # log0回避
            # print(f'pka for link {dfd.loc[l, "link_id"]} from {knode[l]} to {anode[l]}: {pka}') # 確認用でプリント
            LL += np.log(pka)
    print(f'x={x}でLL={LL}') # 確認用でプリント
    return -LL



x_dim = 3 # 8
x_init = np.zeros(x_dim)
x0 = x_init # パラメタ初期値の設定
# length_init = -1
# width_init = 1
# x0[0] = length_init
# x0[1] = width_init
xbounds = [(-10, 10)] * x_dim #, (-10, 5), (-10, 5)] # 探索範囲を限定
n = 0
dL = 100

def fr(x):
    return -loglikelihood(x)

def hessian(x: np.array) -> np.array:
    h = 10 ** -4
    n = len(x)
    res = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            e_i, e_j = np.zeros(n), np.zeros(n)
            e_i[i] = 1
            e_j[j] = 1

            res[i][j] = (fr(x + h * e_i + h * e_j)
            - fr(x + h * e_i - h * e_j)
            - fr(x - h * e_i + h * e_j)
            + fr(x - h * e_i - h * e_j)) / (4 * h * h)
    return res

def tval(x: np.array) -> np.array:
    print(f'hesse行列の逆行列{np.linalg.inv(hessian(x))}')
    print(f'各パラメタの分散{-np.diag(np.linalg.inv(hessian(x)))}')
    return x / np.sqrt(-np.diag(np.linalg.inv(hessian(x))))

# 実行部分
start_time = time.time()
print('start!')
x = x0
res = minimize(loglikelihood, x, method='BFGS')#, bounds = xbounds) #, options={"maxiter":10,"return_all":True})
x0 = res.x
print('x0=', x0)
tval = tval(x0)
x_first = np.zeros(x_dim)
L0 = -1 * loglikelihood(x_first)
LL = -1 * loglikelihood(x0)
print('LL=', LL)
end_time = time.time()
proc_time = end_time - start_time

###### 最終結果の出力 ######
print("計算時間")
print(proc_time)
print("Inputdata")
print("初期尤度 = ", L0)
print("最終尤度 = ", LL)
print("ρ値 = ", (L0 - LL) / L0)
print("修正済ρ値 = ", (L0 - (LL - len(x0))) / L0)
print("パラメータ初期値 = ", x_init)
print("パラメータ推定値 = ", x0)
print("t値 = ", tval)