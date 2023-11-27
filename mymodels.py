import torch
from torch.nn import Linear, BatchNorm1d, LayerNorm
from torch_geometric.nn import GCNConv, TAGConv, GATv2Conv, TransformerConv, GMMConv, GINConv
from torch_geometric.nn.conv import SAGEConv
import torch.nn.functional as F

from my_gatconv import MYGATv2Conv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AttnGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels=5,
                               out_channels=1024,
                               heads=2,
                               edge_dim=1,
                               aggr="add",
                               concat=False,
                               share_weights=False,
                               add_self_loops=False)
        self.norm1 = BatchNorm1d(1024)

        self.conv_layers = torch.nn.ModuleList([])
        self.batch_norms = torch.nn.ModuleList([])

        for i in range(1):
            self.conv_layers.append(
                GATv2Conv(in_channels=1024,
                          out_channels=1024,
                          heads=2,
                          edge_dim=1,
                          aggr="add",
                          concat=False,
                          share_weights=False,
                          add_self_loops=True)
            )

            self.batch_norms.append(
                BatchNorm1d(1024)
            )
        self.fc1 = Linear(1039, 256)
        self.fc2 = Linear(256, 5)
        self.dp = 0.0

    def forward(self, h, bf, edge_index, edge_weight):
        h = self.conv1(h, edge_index, edge_weight)
        h = self.norm1(h)
        h = F.leaky_relu(h)
        h = F.dropout(h, p=self.dp, training=self.training)

        for conv_layer, batch_norm in zip(self.conv_layers, self.batch_norms):
            h = conv_layer(h, edge_index, edge_weight)
            h = batch_norm(h)
            h = F.leaky_relu(h)
            h = F.dropout(h, p=self.dp, training=self.training)

        h = torch.cat((h, bf), dim=-1)
        h = self.fc1(h).relu()
        h = self.fc2(h)
        return h


class SimpleNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dp = 0.2
        self.norm0 = BatchNorm1d(15)
        self.fc1 = Linear(15, 64)
        self.fc2 = Linear(64, 64)
        self.fc3 = Linear(64, 5)

    def forward(self, x_input, h, edge_index, edge_weight):
        h = self.norm0(h)
        h = self.fc1(h).relu()
        h = self.fc2(h).relu()
        h = self.fc3(h)
        return h


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1_sum_ibd = GCNConv(
            in_channels=20,
            out_channels=64,
            add_self_loops=False,
            normalize=False,
            aggr="add"
        )
        self.conv1_mean_ibd = GCNConv(
            in_channels=20,
            out_channels=64,
            add_self_loops=False,
            normalize=False,
            aggr="mean"
        )

        self.conv1_num_edges = GCNConv(
            in_channels=20,
            out_channels=64,
            add_self_loops=False,
            normalize=False,
            aggr="add"
        )

        self.norm1 = LayerNorm(192)

        self.attn_conv = GATv2Conv(in_channels=192,
                                   out_channels=192,
                                   heads=1,
                                   edge_dim=1,
                                   aggr="mean",
                                   concat=False,
                                   share_weights=False,
                                   add_self_loops=True
                                   )
        self.attn_norm = LayerNorm(192)

        self.mean_norm = LayerNorm(5)
        self.std_norm = LayerNorm(5)
        self.edge_norm = LayerNorm(5)
        self.fc1 = Linear(192, 192)
        self.fc2 = Linear(192, 5)

    def forward(self, h, big_features, edge_index, edge_weight):
        big_features = torch.cat((self.mean_norm(big_features[:, :5]),
                                  self.std_norm(big_features[:, 5:10]),
                                  self.edge_norm(big_features[:, 10:15])), dim=-1)

        h = torch.cat((h, big_features), dim=-1)

        h1 = self.conv1_sum_ibd(h, edge_index, edge_weight).relu()
        h2 = self.conv1_mean_ibd(h, edge_index, edge_weight).relu()
        h3 = self.conv1_num_edges(h, edge_index).relu()

        h = torch.cat((h1, h2, h3), dim=-1)
        h = self.norm1(h)

        h = self.attn_conv(h, edge_index, edge_weight).relu()
        h = self.attn_norm(h)

        h = self.fc1(h).relu()
        h = self.fc2(h)
        return h


class GCN_simple(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.norm1 = BatchNorm1d(5)

        self.attn_conv = GATv2Conv(in_channels=5,
                                   out_channels=128,
                                   heads=2,
                                   edge_dim=1,
                                   aggr="mean",
                                   concat=False,
                                   share_weights=False,
                                   add_self_loops=False
                                   )
        self.attn_norm = BatchNorm1d(128)

        self.fc1 = Linear(128, 128)
        self.norm_fc1 = BatchNorm1d(128)
        self.fc2 = Linear(128, 5)
        self.dp = 0.2

    def forward(self, h, edge_index, edge_weight):
        h = self.norm1(h)

        h = self.attn_conv(h, edge_index, edge_weight)

        h = self.fc1(h)
        h = self.norm_fc1(h)
        h = F.leaky_relu(h)
        h = F.dropout(h, p=self.dp, training=self.training)

        h = self.fc2(h)

        return h


class GINNet(torch.nn.Module):
    def __init__(self, num_features, dim):
        super(GINNet, self).__init__()

        nn1 = torch.nn.Sequential(torch.nn.Linear(num_features, dim), torch.nn.ReLU(), torch.nn.Linear(dim, dim))
        self.conv1 = GINConv(nn1)

        nn2 = torch.nn.Sequential(torch.nn.Linear(dim, dim), torch.nn.ReLU(), torch.nn.Linear(dim, dim))
        self.conv2 = GINConv(nn2)

        self.fc1 = torch.nn.Linear(dim, dim)
        self.fc2 = torch.nn.Linear(dim, 2)  # Assuming binary classification for simplicity

    def forward(self, h, bf, edge_index, edge_weight):
        h = F.relu(self.conv1(h, edge_index))
        h = F.relu(self.conv2(h, edge_index))

        h = F.relu(self.fc1(h))
        h = self.fc2(h)

        return h