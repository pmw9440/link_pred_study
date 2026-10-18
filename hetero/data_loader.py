import os
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn as nn


def ready_movielens_merge_data():
    data_path = os.path.join(Path(__file__).parent, "sample_data", "sample_movielens_merged.csv")
    df = pd.read_csv(data_path, sep=",", header=0)

    user_df = df[['user', 'sex', 'age', 'occupation']].drop_duplicates(subset=['user']).copy()
    user_to_idx = {uid: idx for idx, uid in enumerate(user_df['user'])}

    # User 범주형/수치형 피처 전처리
    # sex (F/M -> 0/1)
    user_df['sex_encoded'] = (user_df['sex'] == 'M').astype(int)

    # age 정규화 (StandardScaler)
    scaler = StandardScaler()
    user_df['age_scaled'] = scaler.fit_transform(user_df[['age']])

    # occupation One-hot Encoding
    occ_dummies = pd.get_dummies(user_df['occupation'], prefix='occ')

    # User 노드 최종 feature 텐서 (sex, age, occupation 원핫)
    user_features = pd.concat([user_df[['sex_encoded', 'age_scaled']], occ_dummies], axis=1)
    user_x = torch.tensor(user_features.values.astype(float), dtype=torch.float)

    # (2) Item 정보 추출 및 장르 피처 생성
    item_df = df[['item', 'genre1', 'genre2', 'genre3']].drop_duplicates(subset=['item']).copy()
    item_df = item_df.reset_index(drop=True)

    # Item ID -> 0부터 시작하는 인덱스 매핑 딕셔너리
    item_to_idx = {iid: idx for idx, iid in enumerate(item_df['item'])}

    # 장르 멀티-핫(Multi-hot) 인코딩 처리
    # genre1, genre2, genre3 컬럼의 모든 장르 모으기 ('missing' 제외)
    all_genres = sorted(list(set(
        item_df[['genre1', 'genre2', 'genre3']].values.flatten()
    ) - {'missing'}))
    genre_to_idx = {genre: idx for idx, genre in enumerate(all_genres)}


    # 각 아이템별 멀티핫 벡터 생성
    item_x = torch.zeros((len(item_df), len(all_genres)), dtype=torch.float)
    for i, row in item_df.iterrows():
        for g_col in ['genre1', 'genre2', 'genre3']:
            g = row[g_col]
            if g in genre_to_idx:
                item_x[i, genre_to_idx[g]] = 1.0

    src_user = df['user'].map(user_to_idx).values
    dst_item = df['item'].map(item_to_idx).values

    edge_index = torch.tensor([src_user, dst_item], dtype=torch.long)

    # 라벨(평점) 및 시간(time 정규화 권장)을 edge_attr로 사용
    time_scaled = scaler.fit_transform(df[['time']])
    edge_attr = torch.tensor(
        pd.DataFrame({'label': df['label'], 'time': time_scaled.flatten()}).values,
        dtype=torch.float
    )


    # ---------------------------------------------------------
    # 4. HeteroData 조립
    # ---------------------------------------------------------
    data = HeteroData()

    # (1) User 노드 등록
    data['user'].num_nodes = len(user_df)
    data['user'].x = user_x

    # (2) Item 노드 등록
    data['item'].num_nodes = len(item_df)
    data['item'].x = item_x

    # (3) User -> Item 간선 등록
    data['user', 'rates', 'item'].edge_index = edge_index
    data['user', 'rates', 'item'].edge_attr = edge_attr

    # (4) Item -> User 역방향 간선 등록 (GNN 메시지 패싱용)
    data['item', 'rev_rates', 'user'].edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)

    return data