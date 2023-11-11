import torch
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import GCNConv, TAGConv, GATv2Conv, TransformerConv
from torch_geometric.nn.conv import SAGEConv
import torch.nn.functional as F

from my_gatconv import MYGATv2Conv


class AttnGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels=6,
                               out_channels=128,
                               heads=2,
                               edge_dim=1,
                               aggr="add",
                               concat=False,
                               share_weights=False,
                               add_self_loops=True)
        self.norm1 = BatchNorm1d(128)

        self.conv_layers = torch.nn.ModuleList([])
        self.batch_norms = torch.nn.ModuleList([])

        for i in range(1):
            self.conv_layers.append(
                GATv2Conv(in_channels=128,
                          out_channels=128,
                          heads=2,
                          edge_dim=1,
                          aggr="add",
                          concat=False,
                          share_weights=False,
                          add_self_loops=True)
            )

            self.batch_norms.append(
                BatchNorm1d(128)
            )
        self.fc1 = Linear(128, 128)
        self.fc_norm1 = BatchNorm1d(128)
        self.fc2 = Linear(128, 128)
        self.fc_norm2 = BatchNorm1d(128)
        self.fc3 = Linear(128, 5)

        self.dp = 0.2

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

        h = self.fc_norm1(self.fc1(h))
        h = F.leaky_relu(h)
        h = F.dropout(h, p=self.dp, training=self.training)

        h = self.fc_norm2(self.fc2(h))
        h = F.leaky_relu(h)
        h = F.dropout(h, p=self.dp, training=self.training)

        h = self.fc3(h)

        return h


class AttnGCN_OLD(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels=6,
                               out_channels=128,
                               heads=2,
                               edge_dim=1,
                               aggr="add",
                               concat=False,
                               share_weights=False,
                               add_self_loops=True)
        self.norm1 = BatchNorm1d(128)

        self.conv_layers = torch.nn.ModuleList([])
        self.batch_norms = torch.nn.ModuleList([])

        for i in range(1):
            self.conv_layers.append(
                GATv2Conv(in_channels=128,
                          out_channels=128,
                          heads=2,
                          edge_dim=1,
                          aggr="add",
                          concat=False,
                          share_weights=False,
                          add_self_loops=True)
            )

            self.batch_norms.append(
                BatchNorm1d(128)
            )

        self.fc1 = Linear(128, 128)
        self.fc_norm1 = BatchNorm1d(128)
        self.fc2 = Linear(128, 128)
        self.fc_norm2 = BatchNorm1d(128)
        self.fc3 = Linear(128, 5)

        self.dp = 0.2

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
        #
        # h = self.fc_norm1(self.fc1(h))
        # h = F.leaky_relu(h)
        # h = F.dropout(h, p=self.dp, training=self.training)
        #
        # h = self.fc_norm2(self.fc2(h))
        # h = F.leaky_relu(h)
        # h = F.dropout(h, p=self.dp, training=self.training)
        #
        # h = self.fc3(h)
        h = self.tag_conv(h)
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
        self.fc3 = Linear(128, 128)
        self.norm3 = BatchNorm1d(128)
        self.fc4 = Linear(128, 128)
        self.norm4 = BatchNorm1d(128)
        self.fc5 = Linear(128, 5)

    def forward(self, h, edge_index, edge_weight):
        h = self.norm0(h)
        h = self.norm1(self.fc1(h)).relu()
        h = F.dropout(h, p=self.dp, training=self.training)
        h = self.norm2(self.fc2(h)).relu()
        h = F.dropout(h, p=self.dp, training=self.training)
        h = self.norm3(self.fc3(h)).relu()
        h = F.dropout(h, p=self.dp, training=self.training)
        h = self.norm4(self.fc4(h)).relu()
        h = F.dropout(h, p=self.dp, training=self.training)
        h = self.fc5(h)
        return h


class TAGConv_3l_128h_w_k3(torch.nn.Module):
    def __init__(self):
        super(TAGConv_3l_128h_w_k3, self).__init__()
        self.conv1 = TAGConv(6, 128)
        self.conv2 = TAGConv(128, 128)
        self.conv3 = TAGConv(128, 128)

    def forward(self, x, edge_index, edge_attr):
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)
        return x


class GCNConv_3l_128h_w_l128(torch.nn.Module):
    def __init__(self):
        super(GCNConv_3l_128h_w_l128, self).__init__()
        self.conv1 = GCNConv(data.num_features, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 128)

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.weight.float()
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        # self.norm = BatchNorm1d(15)

        self.convs = torch.nn.ModuleList()

        for _ in range(20):
            self.convs.append(GCNConv(5, 5))

        self.fc1 = Linear(5, 5)
        self.fc2 = Linear(5, 5)
        self.fc3 = Linear(5, 5)

    def forward(self, h, edge_index, edge_weight):

        for i in range(len(self.convs)):
            h = self.convs[i](h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=0.5, training=self.training)

        h = self.fc1(h)
        h = h.relu()
        h = self.fc2(h)
        h = h.relu()
        h = self.fc3(h)
        return h


class GCN_Embedding(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.convs = torch.nn.ModuleList()

        for _ in range(3):
            self.convs.append(GCNConv(in_channels=5,
                                      out_channels=5,
                                      add_self_loops=False,
                                      normalize=False,
                                      aggr="mean"
                                      )
                              )
        self.fc1 = Linear(5, 5)
        self.fc2 = Linear(5, 5)
        self.fc3 = Linear(5, 5)

    def forward(self, h, edge_index, edge_weight):

        for i in range(len(self.convs)):
            h = self.convs[i](h, edge_index, edge_weight).relu()
        h = self.fc1(h).relu()
        h = self.fc2(h).relu()
        h = self.fc3(h)
        return h


class TAGConv_3l_512h_w_k3(torch.nn.Module):
    def __init__(self):
        super(TAGConv_3l_512h_w_k3, self).__init__()
        self.conv1 = TAGConv(5, 512)
        self.conv2 = TAGConv(512, 512)
        self.conv3 = TAGConv(512, 5)

    def forward(self, x, edge_index, edge_weight):
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        x = F.elu(self.conv2(x, edge_index, edge_weight))
        x = self.conv3(x, edge_index, edge_weight)
        return x


class SAGE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.norm0 = BatchNorm1d(5)
        self.conv1 = SAGEConv(in_channels=5,
                              out_channels=64,
                              aggr="mean",
                              normalize=True,
                              root_weight=True,
                              project=True
                              )

        self.conv2 = SAGEConv(in_channels=5,
                              out_channels=64,
                              aggr="mean",
                              normalize=True,
                              root_weight=True,
                              project=True
                              )
        self.conv2_norm = BatchNorm1d(64)

        self.conv3 = SAGEConv(in_channels=64,
                              out_channels=5,
                              aggr="mean",
                              normalize=True,
                              root_weight=True,
                              project=True
                              )

        self.dp = 0.1

    def forward(self, h, edge_index, edge_weight):
        h = self.norm0(h)
        h = self.conv1(h, edge_index, edge_weight).relu()
        h = F.dropout(h, p=self.dp, training=self.training)

        h = self.conv2(h, edge_index, edge_weight).relu()
        h = F.dropout(h, p=self.dp, training=self.training)

        h = self.conv3(h, edge_index, edge_weight)
        h = F.dropout(h, p=self.dp, training=self.training)

        return h


class TransformNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.norm0 = BatchNorm1d(5)
        self.att_conv1 = TransformerConv(in_channels=5,
                                         out_channels=128,
                                         heads=2,
                                         concat=False,
                                         beta=False,
                                         dropout=0.2,
                                         edge_dim=1,
                                         root_weight=True
                                         )
        self.att_conv1_norm = BatchNorm1d(128)

        self.att_conv2 = TransformerConv(in_channels=128,
                                         out_channels=128,
                                         heads=2,
                                         concat=False,
                                         beta=False,
                                         dropout=0.2,
                                         edge_dim=1,
                                         root_weight=True
                                         )
        self.att_conv2_norm = BatchNorm1d(128)
        self.att_conv3 = TransformerConv(in_channels=128,
                                         out_channels=128,
                                         heads=2,
                                         concat=False,
                                         beta=False,
                                         dropout=0.2,
                                         edge_dim=1,
                                         root_weight=True
                                         )
        self.att_conv3_norm = BatchNorm1d(128)
        self.fc1 = Linear(128, 128)
        self.fc1_norm = BatchNorm1d(128)
        self.fc2 = Linear(128, 5)

        self.dp = 0.2

    def forward(self, h, edge_index, edge_weight):
        h = self.norm0(h)
        h = self.att_conv1(h, edge_index, edge_weight).relu()
        h = F.dropout(h, p=self.dp, training=self.training)
        h = self.att_conv1_norm(h)

        h_initial = h.clone()
        h = self.att_conv2(h, edge_index, edge_weight).relu()
        h = F.dropout(h, p=self.dp, training=self.training)
        h = self.att_conv2_norm(h)
        h += h_initial

        h_initial = h.clone()
        h = self.att_conv3(h, edge_index, edge_weight).relu()
        h = F.dropout(h, p=self.dp, training=self.training)
        h = self.att_conv3_norm(h)
        h += h_initial

        h_initial = h.clone()
        h = self.fc1(h).relu()
        h = F.dropout(h, p=self.dp, training=self.training)
        h = self.fc1_norm(h)
        h += h_initial

        h = self.fc2(h)

        return h
