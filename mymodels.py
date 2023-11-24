import torch
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import GCNConv, TAGConv, GATv2Conv, TransformerConv, GMMConv
from torch_geometric.nn.conv import SAGEConv
import torch.nn.functional as F

from my_gatconv import MYGATv2Conv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AttnGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = GATv2Conv(in_channels=6,
                               out_channels=128,
                               heads=2,
                               edge_dim=1,
                               aggr="mean",
                               concat=False,
                               share_weights=False,
                               add_self_loops=True)
        self.norm1 = BatchNorm1d(128)

        self.conv_layers = torch.nn.ModuleList([])
        self.batch_norms = torch.nn.ModuleList([])

        for i in range(2):
            self.conv_layers.append(
                GATv2Conv(in_channels=128,
                          out_channels=128,
                          heads=2,
                          edge_dim=1,
                          aggr="mean",
                          concat=False,
                          share_weights=False,
                          add_self_loops=True)
            )

            self.batch_norms.append(
                BatchNorm1d(128)
            )
        self.fc1 = Linear(128, 5)
        self.dp = 0.1

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

        # h = self.fc_norm1(self.fc1(h))
        # h = F.leaky_relu(h)
        # h = F.dropout(h, p=self.dp, training=self.training)
        #
        # h = self.fc_norm2(self.fc2(h))
        # h = F.leaky_relu(h)
        # h = F.dropout(h, p=self.dp, training=self.training)

        h = self.fc1(h)

        return h


class SimpleNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dp = 0.2
        self.norm0 = BatchNorm1d(5)
        self.fc1 = Linear(5, 128)
        self.norm1 = BatchNorm1d(128)
        self.fc2 = Linear(128, 128)
        self.norm2 = BatchNorm1d(128)
        self.fc3 = Linear(128, 5)

    def forward(self, h, edge_index, edge_weight):
        h = self.norm1(self.fc1(h)).relu()
        h = F.dropout(h, p=self.dp, training=self.training)
        h = self.norm2(self.fc2(h)).relu()
        h = F.dropout(h, p=self.dp, training=self.training)
        h = self.fc3(h)
        return h


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1_sum_ibd = GCNConv(
            in_channels=5,
            out_channels=64,
            add_self_loops=False,
            normalize=False,
            aggr="add"
        )
        self.conv1_mean_ibd = GCNConv(
            in_channels=5,
            out_channels=64,
            add_self_loops=False,
            normalize=False,
            aggr="mean"
        )

        self.conv1_max_ibd = GCNConv(
            in_channels=5,
            out_channels=64,
            add_self_loops=False,
            normalize=False,
            aggr="max"
        )

        self.conv1_num_edges = GCNConv(
            in_channels=5,
            out_channels=64,
            add_self_loops=False,
            normalize=False,
            aggr="add"
        )

        self.conv1_num_edges_max = GCNConv(
            in_channels=5,
            out_channels=64,
            add_self_loops=False,
            normalize=False,
            aggr="max"
        )
        self.norm1 = BatchNorm1d(325)

        self.attn_conv = GATv2Conv(in_channels=325,
                                   out_channels=325,
                                   heads=2,
                                   edge_dim=1,
                                   aggr="mean",
                                   concat=False,
                                   share_weights=False,
                                   add_self_loops=True
                                   )
        self.attn_norm = BatchNorm1d(192)

        self.fc1 = Linear(325, 5)
        self.norm_fc1 = BatchNorm1d(325)
        self.fc2 = Linear(325, 5)
        self.norm_fc2 = BatchNorm1d(325)
        self.fc3 = Linear(325, 325)
        self.norm_fc3 = BatchNorm1d(325)
        self.fc4 = Linear(325, 325)
        self.norm_fc4 = BatchNorm1d(325)
        self.fc5 = Linear(325, 5)
        self.dp = 0.2

    def forward(self, h, edge_num, edge_index, edge_weight):
        h1 = self.conv1_sum_ibd(h, edge_index, edge_weight)
        h2 = self.conv1_mean_ibd(h, edge_index, edge_weight)
        h3 = self.conv1_max_ibd(h, edge_index, edge_weight)
        h4 = self.conv1_num_edges(h, edge_index)
        h5 = self.conv1_num_edges_max(h, edge_index)
        #
        h = torch.cat((h1, h2, h3, h4, h5, edge_num), dim=-1)
        # h = self.norm1(h)
        # h = F.leaky_relu(h)
        # h = F.dropout(h, p=self.dp, training=self.training)

        h = self.attn_conv(h, edge_index, edge_weight)
        # h = self.attn_norm(h)
        # h = F.leaky_relu(h)
        # h = F.dropout(h, p=self.dp, training=self.training)

        # h = self.fc1(h)
        # h = self.norm_fc1(h)
        # h = F.leaky_relu(h)
        # h = F.dropout(h, p=self.dp, training=self.training)
        #
        # h = self.fc2(h)
        # h = self.norm_fc2(h)
        # h = F.leaky_relu(h)
        # h = F.dropout(h, p=self.dp, training=self.training)
        #
        # h = self.fc3(h)
        # h = self.norm_fc3(h)
        # h = F.leaky_relu(h)
        # h = F.dropout(h, p=self.dp, training=self.training)
        #
        # h = self.fc4(h)
        # h = self.norm_fc4(h)
        # h = F.leaky_relu(h)
        # h = F.dropout(h, p=self.dp, training=self.training)

        # h = self.fc1(h)
        # h = self.norm_fc1(h)
        # h = F.leaky_relu(h)
        # h = F.dropout(h, p=self.dp, training=self.training)
        #
        h = self.fc1(h)

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
