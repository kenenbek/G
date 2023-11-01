import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, VGAE, GATv2Conv
from torch.nn import Linear, BatchNorm1d


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.norm0 = BatchNorm1d(5)
        self.conv1 = GATv2Conv(in_channels=5,
                               out_channels=128,
                               heads=2,
                               edge_dim=1,
                               aggr="add",
                               concat=True,
                               share_weights=False)
        self.norm1 = BatchNorm1d(256)

        # self.conv2 = GATv2Conv(in_channels=128,
        #                        out_channels=128,
        #                        heads=2,
        #                        edge_dim=1,
        #                        aggr="mean",
        #                        concat=False,
        #                        share_weights=False)
        # self.norm2 = BatchNorm1d(128)

        self.mu = GATv2Conv(in_channels=256,
                            out_channels=128,
                            heads=2,
                            edge_dim=1,
                            aggr="add",
                            concat=False,
                            share_weights=False)
        self.log_std = GATv2Conv(in_channels=256,
                                 out_channels=128,
                                 heads=2,
                                 edge_dim=1,
                                 aggr="add",
                                 concat=False,
                                 share_weights=False)
        self.dp = 0.2

    def forward(self, h, edge_index, edge_weight):
        h = self.norm0(h)
        h = self.norm1(self.conv1(h, edge_index, edge_weight)).relu()
        h = F.dropout(h, p=self.dp, training=self.training)

        # h = self.norm2(self.conv2(h, edge_index, edge_weight)).relu()
        # h = F.dropout(h, p=self.dp, training=self.training)

        mu = self.mu(h)
        log_std = self.log_std(h)

        return mu, log_std


class WeightedInnerProductDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(128, 128)
        self.fc2 = Linear(128, 128)
        self.fc3 = Linear(128, 128)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent variables 'z' into edge weights for the given node-pairs 'edge_index'.
        """
        z = self.fc1(z).relu()
        z = self.fc2(z).relu()
        z = self.fc3(z)
        return torch.matmul(z, z.t())  # (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

    def forward_all(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent variables 'z' into a dense adjacency matrix representing edge weights.
        """
        return torch.matmul(z, z.t())


class SimplePredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dp = 0.2
        self.norm0 = BatchNorm1d(128)
        self.fc1 = Linear(128, 128)
        self.norm1 = BatchNorm1d(128)
        self.fc2 = Linear(128, 128)
        self.norm2 = BatchNorm1d(128)
        self.fc3 = Linear(128, 128)

    def forward(self, h):
        h = self.norm0(h)
        h = self.norm1(self.fc1(h)).relu()
        h = F.dropout(h, p=self.dp, training=self.training)
        h = self.norm2(self.fc2(h)).relu()
        h = F.dropout(h, p=self.dp, training=self.training)
        h = self.fc3(h)
        return h
