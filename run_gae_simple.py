import wandb

import torch
import torch.nn.functional as F
from torch_geometric.utils import subgraph, to_dense_adj
from torch_geometric.nn import GAE
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from torch.optim.lr_scheduler import StepLR

from mydata import ClassBalancedNodeSplit, MyDataset, create_hidden_train_mask
from my_vgae import EncoderGAE, WeightedInnerProductDecoder, SimplePredictor
from mymodels import AttnGCN
from utils import evaluate_one_by_one, evaluate_batch, evaluate_one_by_one_load_from_file, calc_accuracy, \
    set_global_seed, vgae_evaluate_one_by_one
from utils import inductive_train, to_one_hot


def reconstruct_edges(z, edge_index):
    # Assuming data.x is a tensor of node features [num_nodes, num_features]
    # and edge_index is a tensor of shape [2, num_edges] where the first row contains
    # the indices of the source nodes and the second row contains the indices of the target nodes.

    # Gather the source node features using edge_index
    x_i = z[edge_index[0]]

    # Gather the target node features using edge_index
    x_j = z[edge_index[1]]

    # x_i and x_j are gathered node features from the previous example
    # Perform scalar multiplication (dot product) along the feature dimension
    scalar_products = (x_i * x_j).sum(dim=1).unsqueeze(1)

    return scalar_products


if __name__ == "__main__":
    set_global_seed(42)

    wandb.init(project="Genomics", entity="kenenbek")

    # Store configurations/hyperparameters
    wandb.config.lr = 0.001
    wandb.config.weight_decay = 5e-4
    wandb.config.epochs = 1500

    full_dataset = MyDataset(root="full_data/")
    full_data = full_dataset[0]
    num_nodes = full_data.y.shape[0]
    train_indices_full = torch.load("full_data/0/train_indices.pt")
    test_indices = torch.load("full_data/0/test_indices.pt")

    train_mask_f = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask_f[train_indices_full] = True
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[test_indices] = True

    assert torch.equal(train_mask_f, ~test_mask), "Error"

    train_mask_sub, train_mask_h = create_hidden_train_mask(train_indices_full, num_nodes, hide_frac=0.0)
    full_data.recalculate_input_features(train_mask_h)

    gae_model = GAE(encoder=EncoderGAE(),
                    decoder=WeightedInnerProductDecoder()
                    )
    gae_optimizer = torch.optim.Adam(gae_model.parameters(), lr=wandb.config.lr,
                                     weight_decay=wandb.config.weight_decay)
    gae_scheduler = StepLR(gae_optimizer, step_size=500, gamma=0.1)

    t = trange(wandb.config.epochs, leave=True)
    losses = []

    # Extract the subgraph associated with the training nodes
    train_nodes = torch.nonzero(train_mask_f).squeeze()

    train_edge_index, train_edge_attr_multi = subgraph(
        train_nodes, full_data.edge_index, edge_attr=full_data.edge_attr_multi, relabel_nodes=True
    )

    _, train_edge_weight = subgraph(
        train_nodes, full_data.edge_index, edge_attr=full_data.edge_attr, relabel_nodes=True
    )

    weighted_matrix = to_dense_adj(train_edge_index, edge_attr=train_edge_weight).squeeze(0).squeeze(-1)

    zeros_tensor = torch.zeros_like(full_data.train_x[train_mask_f])

    for epoch in t:
        gae_model.train()
        gae_optimizer.zero_grad()

        z = gae_model.encode(full_data.x_one_hot[train_mask_f], train_edge_index, train_edge_attr_multi)
        adj_reconstructed = gae_model.decode(z)

        # pred_edge_weights = reconstruct_edges(z, train_edge_index)

        recon_loss = F.mse_loss(adj_reconstructed, weighted_matrix)
        recon_loss.backward()
        gae_optimizer.step()
        gae_scheduler.step()

        wandb.log({"recon_loss": recon_loss.item()})

    torch.save(gae_model.state_dict(), "gae.pt")
    wandb.finish()
