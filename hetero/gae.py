import os
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

import torch
import torch.nn.functional as F
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn as nn
from torch_geometric.utils import negative_sampling


class HeteroEncoder(nn.Module):

    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class HeteroEdgeDecoder(nn.Module):
    """
    간선의 source와 target 노드 임베딩 간의 내적(dot product)으로
    연결 확률(logits)을 계산하는 디코더
    """
    def forward(self, z_dict, edge_label_index, edge_type):
        src_type, _, dst_type = edge_type
        # z_src = z_dict[src_type]
        # z_dst = z_dict[dst_type]
        z_src = F.normalize(z_dict[src_type], p=2, dim=-1)
        z_dst = F.normalize(z_dict[dst_type], p=2, dim=-1)

        # edge_label_index: [2, num_edges]
        src_idx = edge_label_index[0]
        dst_idx = edge_label_index[1]

        # 노드 임베딩 간 내적 계산
        scores = (z_src[src_idx] * z_dst[dst_idx]).sum(dim=-1)
        return scores

class HeteroGAE(nn.Module):
    def __init__(self, metadata, hidden_channels=64, out_channels=32):
        super().__init__()

        base_encoder = HeteroEncoder(hidden_channels, out_channels)
        self.encoder = to_hetero(base_encoder, metadata, aggr='sum')

        self.decoder = HeteroEdgeDecoder()

    def encode(self, x_dict, edge_index_dict):
        return self.encoder(x_dict, edge_index_dict)

    def decode(self, z_dict, edge_label_index, edge_type):
            return self.decoder(z_dict, edge_label_index, edge_type)


def train(data):
    '''
    Parameter
    ---------
    data: HeteroData
        train, eval, test 합친 데이터
    '''
    def train_each():

        # 1. 노드 임베딩 추출
        z_dict = model.encode(train_data.x_dict, train_data.edge_index_dict)

        # 2. Positive 간선
        pos_edge_index = train_data[target_edge_type].edge_label_index
        num_pos = pos_edge_index.size(1)

        # 3. Negative 간선 샘플링 (Bipartite 엣지 지원)
        src_type, _, dst_type = target_edge_type
        num_src_nodes = train_data[src_type].num_nodes
        num_dst_nodes = train_data[dst_type].num_nodes

        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=(num_src_nodes, num_dst_nodes),
            num_neg_samples=num_pos
        )

        # 4. Pos/Neg 간선 통합
        edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        labels = torch.cat([
            torch.ones(num_pos, device=device),
            torch.zeros(neg_edge_index.size(1), device=device)
        ])

        # 5. 디코딩 및 Loss 계산
        pred = model.decode(z_dict, edge_label_index, target_edge_type)
        loss = criterion(pred, labels)

        loss.backward()

        # 그래디언트 클리핑 추가 (기울기 폭주 방지)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        return loss.item()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device info:',  device)

    # DATA SPLIT!!
    target_edge_type = data.edge_types[0]
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        edge_types=[target_edge_type],
        rev_edge_types=[data.edge_types[1]] if len(data.edge_types) > 1 else None,
        neg_sampling_ratio=1.0,
        add_negative_train_samples=False  # 학습 시 매 에포크마다 직접 Negative Sampling 수행
        )

    train_data, val_data, test_data = transform(data)
    train_data = train_data.to(device)

    model = HeteroGAE(metadata=train_data.metadata(), hidden_channels=64, out_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # 실행
    for epoch in range(1, 501):
        loss = train_each()
        if epoch % 20 == 0:
            val_auc, val_ap = evaluate(model, val_data)
            print(f"Epoch: {epoch:03d} | Train Loss: {loss:.4f} | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")

    
@torch.no_grad()
def evaluate(model, eval_data):
    target_edge = ('user', 'rates', 'item')
    rev_edge = ('item', 'rev_rates', 'user')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_data.to(device)

    model.eval()

    # 1. 인코딩 (평가 데이터의 edge_index_dict 사용 - Message Passing용 간선)
    z_dict = model.encode(eval_data.x_dict, eval_data.edge_index_dict)

    # 2. RandomLinkSplit에 의해 생성된 평가용 엣지 및 라벨 가져오기
    # edge_label_index: [2, num_pos + num_neg]
    # edge_label: [num_pos + num_neg] (1 또는 0)
    edge_label_index = eval_data[target_edge].edge_label_index
    labels = eval_data[target_edge].edge_label.cpu().numpy()

    # 3. 디코더를 통한 점수 예측
    logits = model.decode(z_dict, edge_label_index, target_edge)
    probs = torch.sigmoid(logits).cpu().numpy()

    # 4. AUC-ROC 및 Average Precision(AP) 계산
    auc = roc_auc_score(labels, probs)
    ap = average_precision_score(labels, probs)

    return auc, ap


# Epoch: 020 | Train Loss: 0.6912 | Val AUC: 0.6669 | Val AP: 0.6479
# Epoch: 040 | Train Loss: 0.6995 | Val AUC: 0.6447 | Val AP: 0.6337
# Epoch: 060 | Train Loss: 0.6803 | Val AUC: 0.6887 | Val AP: 0.6652
# Epoch: 080 | Train Loss: 0.6453 | Val AUC: 0.6840 | Val AP: 0.6622
# Epoch: 100 | Train Loss: 0.6668 | Val AUC: 0.6885 | Val AP: 0.6704
# Epoch: 120 | Train Loss: 0.6527 | Val AUC: 0.7088 | Val AP: 0.6842
# Epoch: 140 | Train Loss: 0.6292 | Val AUC: 0.7285 | Val AP: 0.7008
# Epoch: 160 | Train Loss: 0.6402 | Val AUC: 0.7208 | Val AP: 0.7019
# Epoch: 180 | Train Loss: 0.6098 | Val AUC: 0.7529 | Val AP: 0.7218
# Epoch: 200 | Train Loss: 0.6142 | Val AUC: 0.7596 | Val AP: 0.7322
# Epoch: 220 | Train Loss: 0.6083 | Val AUC: 0.7604 | Val AP: 0.7374
# Epoch: 240 | Train Loss: 0.5904 | Val AUC: 0.7793 | Val AP: 0.7488
# Epoch: 260 | Train Loss: 0.5915 | Val AUC: 0.7870 | Val AP: 0.7542
# Epoch: 280 | Train Loss: 0.5895 | Val AUC: 0.7811 | Val AP: 0.7586
# Epoch: 300 | Train Loss: 0.5824 | Val AUC: 0.7929 | Val AP: 0.7648
# Epoch: 320 | Train Loss: 0.5814 | Val AUC: 0.8002 | Val AP: 0.7705
# Epoch: 340 | Train Loss: 0.5761 | Val AUC: 0.7894 | Val AP: 0.7598
# Epoch: 360 | Train Loss: 0.5830 | Val AUC: 0.7907 | Val AP: 0.7680
# Epoch: 380 | Train Loss: 0.5720 | Val AUC: 0.8025 | Val AP: 0.7738
# Epoch: 400 | Train Loss: 0.5723 | Val AUC: 0.8006 | Val AP: 0.7698
# Epoch: 420 | Train Loss: 0.5696 | Val AUC: 0.8017 | Val AP: 0.7711
# Epoch: 440 | Train Loss: 0.5706 | Val AUC: 0.8015 | Val AP: 0.7715
# Epoch: 460 | Train Loss: 0.5705 | Val AUC: 0.8077 | Val AP: 0.7849
# Epoch: 480 | Train Loss: 0.5621 | Val AUC: 0.8094 | Val AP: 0.7812
# Epoch: 500 | Train Loss: 0.5710 | Val AUC: 0.7995 | Val AP: 0.7776