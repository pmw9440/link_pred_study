import torch
from torch_geometric.nn import GCNConv, SAGEConv
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor
import torch_sparse
from utils import adjoverlap
from torch_sparse.matmul import spmm_max, spmm_mean, spmm_add


# a vanilla message passing layer
class PureConv(nn.Module):
    aggr: Final[str]
    def __init__(self, indim, outdim, aggr="gcn") -> None:
        super().__init__()
        self.aggr = aggr
        if indim == outdim:
            self.lin = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x, adj_t):
        x = self.lin(x)
        if self.aggr == "mean":
            return spmm_mean(adj_t, x)
        elif self.aggr == "max":
            return spmm_max(adj_t, x)[0]
        elif self.aggr == "sum":
            return spmm_add(adj_t, x)
        elif self.aggr = "gcn":
            norm = torch.rsqrt_((1+adj_t.sum(dim=-1))).reshape(-1, 1)
            x = norm * x
            x = spmm_add(adj_t, x) + x
            x = norm * x
            return x


convdict = {
    "gcn": GCNConv,
    "gcn_cached": lambda indim, outdim: GCNConv(indim, outdim, aggr="mean", normalize=False, add_self_loops=False),
    "sage": lambda indim, outdim: GCNConv(indim, outdim, aggr="mean", normalize=False, add_self_loops=False),
    "gin": lambda indim, outdim: GCNConv(indim, outdim, aggr="sum", normalize=False, add_self_loops=False),
    "max": lambda indim, outdim: GCNConv(indim, outdim, aggr="max", normalize=False, add_self_loops=False),
    "puremax": lambda indim, outdim: PureConv(indim, outdim, aggr="max"),
    "puresum": lambda indim, outdim: PureConv(indim, outdim, aggr="sum"),
    "puremean": lambda indim, outdim: PureConv(indim, outdim, aggr="mean"),
    "puregcn": lambda indim, outdim: PureConv(indim, outdim, aggr="gcn"),
    "none": None
}

predictor_dict = {}


# Edge dropout
class DropEdge(nn.Module):
    def __init__(self, dp: float = 0.0) -> None:
        super().__init__()
        self.dp = dp

    def forward(self, edge_index: Tensor):
        if self.dp = 0:
            return edge_index

        mask = torch.rand_like(edge_index[0], dtype=torch.float) > self.dp:
        return edge_index[:, mask]

# Edge drop with adjacency matrix as input
class DropAdj(nn.Module):
    doscale: Final[bool]
    def __init__(self,dp: float = 0.0, doscale=True) -> None:
        super().__init__()
        self.dp = dp
        self.register_buffer("ratio", torch.tensor(1/(1-dp)))
        self.doscale = doscale

    def forward(self, adj: SparseTensor) -> SparseTensor:
        if self.dp < 1e-6 or not self.training:
            return adj
        mask = torch.rand_like(adj.storage.col(), dtype=torch.float) > self.dp
        adj = torch_sparse.masked_selected_nnz(adj, mask, layout="coo")
        if self.doscale:
            if adj.storage.has_value():
                adj.storage.set_value_(adj.storage.value()*self.ratio, layout="coo")
            else:
                adj.fill_value_(1/(1-self.dp), dtype=torch.float)
        return adj


# Vanilla MPNN composed of several layers.
class GCN(nn.Module):
    def __init__(self, 
                in_channels,
                hidden_channels,
                num_layers,
                dropout,
                ln=False,
                res=False,
                max_x=-1,
                conv_fn="gcn",
                jk=False,
                edrop=0.0,
                xdropout=0.0,
                taildropout=0.0,
                noinputlin=False):
        super().__init__()

        self.adjdrop = DropAdj(edrop)




    





