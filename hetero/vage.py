import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.transforms import RandomLinkSplit

# ---------------------------------------------------------
# 1. VGAE 인코더 및 디코더 정의
# ---------------------------------------------------------
class HeteroVGAEEncoder(nn.Module):
    """
    이종 그래프 입력을 받아 각 노드 유형별로 평균(mu)과 로그 분산(log_std)을 출력하는 인코더.
    SAGEConv를 to_hetero로 래핑하여 사용합니다.
    """
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # 공통 첫 번째 계층
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        
        # 평균(Mu)과 로그 분산(Log Variance / Log Std)을 위한 개별 계층들
        self.conv_mu = SAGEConv((-1, -1), out_channels)
        self.conv_logstd = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        # 1. Feature Extraction (공통)
        h = self.conv1(x, edge_index).relu()
        h = F.dropout(h, p=0.2, training=self.training)
        
        # 2. Mu 및 LogStd 생성
        mu = self.conv_mu(h, edge_index)
        logstd = self.conv_logstd(h, edge_index)
        
        return mu, logstd


class StableEdgeDecoder(nn.Module):
    """
    샘플링된 임베딩 쌍(Z_src, Z_dst)을 기반으로 간선 존재 확률(logits)을 추정하는 디코더.
    다차원 분포에서 안정적인 패턴을 추출하기 위해 소형 MLP를 적용합니다.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_channels, 1)
        )

    def forward(self, z_dict, edge_label_index, edge_type):
        src_type, _, dst_type = edge_type
        
        # 수치적 안정을 위한 정규화
        z_src = F.normalize(z_dict[src_type], p=2, dim=-1)
        z_dst = F.normalize(z_dict[dst_type], p=2, dim=-1)

        src_idx = edge_label_index[0]
        dst_idx = edge_label_index[1]

        # 엣지의 소스 노드와 타겟 노드 특징 합치기
        edge_feat = torch.cat([z_src[src_idx], z_dst[dst_idx]], dim=-1)
        scores = self.mlp(edge_feat).squeeze(-1)
        return scores


class HeteroVGAE(nn.Module):
    def __init__(self, metadata, hidden_channels=32, out_channels=16):
        super().__init__()
        # 1. Base Encoder를 'to_hetero'로 이종화
        base_encoder = HeteroVGAEEncoder(hidden_channels, out_channels)
        self.encoder = to_hetero(base_encoder, metadata, aggr='sum')
        
        # 2. Decoder 정의
        self.decoder = StableEdgeDecoder(out_channels)

    def encode(self, x_dict, edge_index_dict):
        # 각 노드 타입별로 mu_dict, logstd_dict를 반환
        return self.encoder(x_dict, edge_index_dict)

    def reparameterize(self, mu_dict, logstd_dict):
        """
        Reparameterization Trick: z = mu + std * epsilon
        """
        z_dict = {}
        for node_type in mu_dict.keys():
            mu = mu_dict[node_type]
            logstd = logstd_dict[node_type]
            
            if self.training:
                # 학습 중에는 샘플링 수행
                std = torch.exp(logstd)
                eps = torch.randn_like(std)
                z_dict[node_type] = mu + eps * std
            else:
                # 평가 시에는 모델 안정성을 위해 평균값(mu)을 그대로 임베딩으로 사용
                z_dict[node_type] = mu
        return z_dict

    def decode(self, z_dict, edge_label_index, edge_type):
        return self.decoder(z_dict, edge_label_index, edge_type)


# ---------------------------------------------------------
# 2. 데이터 분할 및 환경 정의
# ---------------------------------------------------------
target_edge = ('user', 'rates', 'item')
rev_edge = ('item', 'rev_rates', 'user')

transform = RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    edge_types=[target_edge],
    rev_edge_types=[rev_edge],
    neg_sampling_ratio=1.0,
    add_negative_train_samples=True
)

train_data, val_data, test_data = transform(data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HeteroVGAE(metadata=train_data.metadata(), hidden_channels=32, out_channels=16).to(device)

# L2 정규화(weight_decay)를 적용한 Adam 옵티마이저
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
bce_criterion = nn.BCEWithLogitsLoss()

train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)


# ---------------------------------------------------------
# 3. KL Divergence 계산 함수
# ---------------------------------------------------------
def kl_loss(mu_dict, logstd_dict):
    """
    각 노드 타입별 잠재 변수의 분포와 표준 정규 분포 N(0, I) 사이의 KL 다이버전스를 계산합니다.
    """
    total_kl = 0.0
    num_nodes_total = 0
    
    for node_type in mu_dict.keys():
        mu = mu_dict[node_type]
        logstd = logstd_dict[node_type]
        
        # 가우시안 KL divergence 공식을 각 노드 타입에 적용
        kl = -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu.pow(2) - torch.exp(2 * logstd), dim=-1)
        )
        total_kl += kl * mu.size(0)
        num_nodes_total += mu.size(0)
        
    return total_kl / num_nodes_total


# ---------------------------------------------------------
# 4. 학습 & 평가 루프
# ---------------------------------------------------------
def train():
    model.train()
    optimizer.zero_grad()

    # (1) Mu, LogStd 추론
    mu_dict, logstd_dict = model.encode(train_data.x_dict, train_data.edge_index_dict)
    
    # (2) Reparameterization Trick으로 잠재 공간(Z) 샘플링
    z_dict = model.reparameterize(mu_dict, logstd_dict)

    # (3) Positive 및 Negative 엣지에 대한 학습용 라벨
    edge_label_index = train_data[target_edge].edge_label_index
    labels = train_data[target_edge].edge_label

    # (4) Recon Loss (Reconstruction Loss)
    logits = model.decode(z_dict, edge_label_index, target_edge)
    recon_loss = bce_criterion(logits, labels)

    # (5) Regularization Loss (KL Divergence Loss)
    # 가용 데이터 셋의 스케일에 따라 가중치 조정 가능 (이 경우 1/N배 가중)
    kld_loss = kl_loss(mu_dict, logstd_dict)
    
    # 전체 손실 = 복원 오류 + 가중화된 KL Divergence
    loss = recon_loss + (0.01 * kld_loss)

    loss.backward()
    
    # 그라디언트 폭주 방지
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item(), recon_loss.item(), kld_loss.item()


@torch.no_grad()
def evaluate(eval_data):
    model.eval()
    
    # (1) Encode & Get Z
    mu_dict, logstd_dict = model.encode(eval_data.x_dict, eval_data.edge_index_dict)
    z_dict = model.reparameterize(mu_dict, logstd_dict) # 평가시에는 z_dict = mu_dict

    edge_label_index = eval_data[target_edge].edge_label_index
    labels = eval_data[target_edge].edge_label.cpu().numpy()

    # (2) Score Prediction
    logits = model.decode(z_dict, edge_label_index, target_edge)
    probs = torch.sigmoid(logits).cpu().numpy()

    # (3) 지표 산출
    auc = roc_auc_score(labels, probs)
    ap = average_precision_score(labels, probs)
    return auc, ap


# 실행
for epoch in range(1, 101):
    loss, r_loss, k_loss = train()
    if epoch % 20 == 0:
        val_auc, val_ap = evaluate(val_data)
        print(f"Epoch: {epoch:03d} | Loss: {loss:.4f} (Recon: {r_loss:.4f}, KL: {k_loss:.4f}) | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")

# 최종 테스트 데이터 평가
test_auc, test_ap = evaluate(test_data)
print("-" * 65)
print(f"[Test Result] AUC-ROC: {test_auc:.4f} | AP (PR-AUC): {test_ap:.4f}")