import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)


class SAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=2):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)


class MLP(torch.nn.Module):
    """Same sizes as the GNNs but ignores edge_index (no graph). Control model."""
    def __init__(self, in_dim, hidden=64, out_dim=2):
        super().__init__()
        self.l1 = torch.nn.Linear(in_dim, hidden)
        self.l2 = torch.nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.l1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.l2(x)