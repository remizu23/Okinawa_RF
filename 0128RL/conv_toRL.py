import pandas as pd
import numpy as np
import torch
import os

# --- 設定エリア ---
NPZ_PATH = '/home/mizutani/projects/RF/data/input_real_m5.npz' # パスは適宜調整してください
LOC_CSV = 'ble_location2025.csv'
PAD_TOKEN = 38
END_TOKEN = 39  # <e>ただtokenizationで<e>は入れるので，input_real_m5に入ってないので関係ない


# 隣接マップの定義
ADJACENCY_MAP = {
    0: [1, 2, 4, 11], 1: [0, 2, 4, 5, 9], 2: [0, 1, 5, 6],
    4: [0, 1, 5, 8, 9, 10, 11], 5: [1, 2, 4, 6, 10], 6: [2, 5, 10, 13, 14],
    8: [4, 9, 11], 9: [1, 4, 8, 10, 12], 10: [4, 5, 6, 9, 12, 13],
    11: [0, 4, 8], 12: [9, 10, 13], 13: [6, 10, 12, 14, 15],
    14: [6, 13, 15, 16], 15: [13, 14], 16: [14, 17, 18],
    17: [16, 18], 18: [16, 17]
}

# --- 1. node.csv の作成 ---
df_loc = pd.read_csv(LOC_CSV)
node_coords = df_loc.set_index('token_id')[['lon', 'lat']].to_dict('index')
all_nodes = sorted(list(ADJACENCY_MAP.keys()))

node_rows = []
for nid in all_nodes:
    c = node_coords.get(nid, {'lon': 0, 'lat': 0})
    node_rows.append({
        'node_id': nid, 
        'fid': nid,       # dataset.pyが参照するID
        'x': c['lon'], 
        'y': c['lat']
    })
pd.DataFrame(node_rows).to_csv('node.csv', index=False)
print("Generated node.csv")

# --- 2. link.csv の作成 ---
links = []
link_id_counter = 0

for src in all_nodes:
    # 移動リンク
    for dst in ADJACENCY_MAP[src]:
        links.append({
            'link_id': link_id_counter, 
            'fid': link_id_counter, 
            'O': src,        # dataset.py用: 起点
            'D': dst,        # dataset.py用: 終点
            'type': 'move'
        })
        link_id_counter += 1
    # 滞在リンク (自己ループ)
    links.append({
        'link_id': link_id_counter, 
        'fid': link_id_counter, 
        'O': src,
        'D': src,
        'type': 'stay'
    })
    link_id_counter += 1

df_link = pd.DataFrame(links)

# ★★★ ここで length と capacity を必ず追加 ★★★
df_link['length'] = 1.0
df_link['capacity'] = 1000.0  # 適当な正の値 (正規化されるので定数なら何でもOK)

# ダミー変数の付与
# --- 修正版: ダミー変数の付与部分 ---

# 通りダミーは「移動(move)」のときだけONにする
def get_street_dummy(row, targets):
    # 目的地が対象エリア かつ "移動リンク" の場合のみ 1
    if (row['D'] in targets) and (row['type'] == 'move'):
        return 1
    return 0

# 既存のコードを以下のように書き換え
df_link['park_ave']    = df_link.apply(lambda r: get_street_dummy(r, [14, 16, 18]), axis=1)
df_link['first_st']    = df_link.apply(lambda r: get_street_dummy(r, [4, 5, 6, 8, 9, 10, 11]), axis=1)
df_link['gate_st']     = df_link.apply(lambda r: get_street_dummy(r, [11]), axis=1)
df_link['national_rd'] = df_link.apply(lambda r: get_street_dummy(r, [0, 1, 2]), axis=1)
df_link['parking']     = df_link.apply(lambda r: get_street_dummy(r, [12, 15, 17]), axis=1)

# 広場滞在はそのまま (stay かつ 特定ノード)
df_link['plaza_stay']  = df_link.apply(lambda r: 1 if (r['type']=='stay' and r['D'] in [2, 11, 14]) else 0, axis=1)

df_link.to_csv('link.csv', index=False)
print("Generated link.csv (with O, D, fid, capacity)")
# --- 3. synthetic_data.csv の作成 ---
if os.path.exists(NPZ_PATH):
    data = np.load(NPZ_PATH)
    routes = data['route_arr'] 
    
    # リンク検索用辞書: (Origin, Destination) -> link_id
    link_map = {(r['O'], r['D']): r['link_id'] for _, r in df_link.iterrows()}
    
    obs_rows = []
    for trip_id, seq in enumerate(routes):
        valid_seq = seq[seq != PAD_TOKEN]
        if len(valid_seq) < 2: continue
        
        # トークン変換 (19以降は滞在なので -19 して元のノードに戻す)
        nodes = []
        for t in valid_seq:
            nodes.append(t if t < 19 else t - 19)
            
        for i in range(len(nodes) - 1):
            u, v = nodes[i], nodes[i+1]
            if (u, v) in link_map:
                obs_rows.append({
                    'trip_id': trip_id,
                    'link_id': link_map[(u, v)],
                    'link_length': 1.0  # ここは必須ではないが念のため
                })
    
    pd.DataFrame(obs_rows).to_csv('synthetic_data.csv', index=False)
    print("Generated synthetic_data.csv")
else:
    print(f"Warning: {NPZ_PATH} not found. Skipping synthetic_data.csv generation.")