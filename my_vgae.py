import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, VGAE, GATv2Conv
from torch.nn import Linear, BatchNorm1d, Identity

from my_gatconv import MYGATv2Conv


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.norm0 = BatchNorm1d(5)
        self.conv1 = GATv2Conv(in_channels=5,
                               out_channels=2048,
                               heads=2,
                               edge_dim=1,
                               aggr="add",
                               concat=False,
                               share_weights=False)
        self.norm1 = BatchNorm1d(2048)

        self.conv2 = GATv2Conv(in_channels=2048,
                               out_channels=2048,
                               heads=2,
                               edge_dim=1,
                               aggr="add",
                               concat=False,
                               share_weights=False)
        self.norm2 = BatchNorm1d(2048)

        self.mu = GATv2Conv(in_channels=2048,
                            out_channels=2048,
                            heads=2,
                            edge_dim=1,
                            aggr="add",
                            concat=False,
                            share_weights=False)
        self.log_std = GATv2Conv(in_channels=2048,
                                 out_channels=2048,
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

        h = self.norm2(self.conv2(h, edge_index, edge_weight)).relu()
        h = F.dropout(h, p=self.dp, training=self.training)

        mu = self.mu(h, edge_index, edge_weight)
        log_std = self.log_std(h, edge_index, edge_weight)

        return mu, log_std


class WeightedInnerProductDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent variables 'z' into edge weights for the given node-pairs 'edge_index'.
        """
        return torch.matmul(z, z.t())  # (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)


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


class EncoderGAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(
            in_channels=6,
            out_channels=256,
        )
        self.norm1 = BatchNorm1d(256)

        self.conv_layers = torch.nn.ModuleList([])
        self.batch_norms = torch.nn.ModuleList([])

        for i in range(6):
            self.conv_layers.append(
                GCNConv(
                    in_channels=256,
                    out_channels=256,
                )
            )
            self.batch_norms.append(
                BatchNorm1d(256)
            )
        self.conv_last = GCNConv(
            in_channels=256,
            out_channels=256,
        )
        self.dp = 0.0

    def forward(self, h, edge_index, edge_weight):
        h = self.conv1(h, edge_index, edge_weight)
        h = self.norm1(h)
        h = F.leaky_relu(h)
        h = F.dropout(h, p=self.dp, training=self.training)

        for conv_layer, batch_norm in zip(self.conv_layers, self.batch_norms):
            h = conv_layer(h, edge_index, edge_weight)
            h = batch_norm(h)
            h = F.leaky_relu(h)
            h = F.dropout(h, p=self.dp, training=self.training)

        h = self.conv_last(h, edge_index, edge_weight)
        return h

# class EncoderGAE(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = MYGATv2Conv(in_channels=6,
#                                  out_channels=2048,
#                                  heads=2,
#                                  edge_dim=6,
#                                  aggr="add",
#                                  concat=False,
#                                  share_weights=False,
#                                  add_self_loops=True)
#         self.norm1 = BatchNorm1d(2048)
#
#         self.conv2 = MYGATv2Conv(in_channels=2048,
#                                  out_channels=2048,
#                                  heads=2,
#                                  edge_dim=6,
#                                  aggr="add",
#                                  concat=False,
#                                  share_weights=False,
#                                  add_self_loops=True)
#         self.norm2 = BatchNorm1d(2048)
#
#         self.fc1 = Linear(2048, 2048)
#         self.fc_norm1 = BatchNorm1d(2048)
#         self.fc2 = Linear(2048, 2048)
#         self.fc_norm2 = BatchNorm1d(2048)
#         self.fc3 = Linear(2048, 2048)
#
#         self.dp = 0.2
#
#     def forward(self, h, edge_index, edge_weight):
#         h = self.norm1(self.conv1(h, edge_index, edge_weight)).relu()
#         h = F.dropout(h, p=self.dp, training=self.training)
#
#         h = self.norm2(self.conv2(h, edge_index, edge_weight)).relu()
#         h = F.dropout(h, p=self.dp, training=self.training)
#
#         h = self.fc_norm1(self.fc1(h)).relu()
#         h = F.dropout(h, p=self.dp, training=self.training)
#
#         h = self.fc_norm2(self.fc2(h)).relu()
#         h = F.dropout(h, p=self.dp, training=self.training)
#
#         h = self.fc3(h)
#
#         return h
