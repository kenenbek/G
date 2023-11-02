import wandb

import torch
import torch.nn.functional as F
from torch_geometric.utils import subgraph, to_dense_adj
from torch_geometric.nn import VGAE
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from torch.optim.lr_scheduler import StepLR

from mydata import ClassBalancedNodeSplit, MyDataset, create_hidden_train_mask
from my_vgae import Encoder, WeightedInnerProductDecoder, SimplePredictor
from mymodels import AttnGCN
from utils import evaluate_one_by_one, evaluate_batch, evaluate_one_by_one_load_from_file, calc_accuracy, \
    set_global_seed, vgae_evaluate_one_by_one
from utils import inductive_train, to_one_hot

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

    vgae_model = VGAE(encoder=Encoder(),
                      decoder=WeightedInnerProductDecoder()
                      )
    vgae_optimizer = torch.optim.Adam(vgae_model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)
    vgae_scheduler = StepLR(vgae_optimizer, step_size=500, gamma=0.1)

    t = trange(wandb.config.epochs, leave=True)
    losses = []

    # Extract the subgraph associated with the training nodes
    train_nodes = torch.nonzero(train_mask_f).squeeze()
    train_edge_index, train_edge_weight = subgraph(
        train_nodes, full_data.edge_index, edge_attr=full_data.edge_attr, relabel_nodes=True
    )

    weighted_matrix = to_dense_adj(train_edge_index, edge_attr=train_edge_weight).squeeze(0).squeeze(-1)

    for epoch in t:
        vgae_model.train()
        vgae_optimizer.zero_grad()

        z = vgae_model.encode(full_data.train_x[train_mask_f], train_edge_index, train_edge_weight)
        adj_reconstructed = vgae_model.decode(z)

        recon_loss = F.mse_loss(adj_reconstructed, weighted_matrix)
        kl_loss = vgae_model.kl_loss()
        vgae_loss = recon_loss + kl_loss
        vgae_loss.backward()
        vgae_optimizer.step()
        vgae_scheduler.step()

        wandb.log({"kl_loss": vgae_loss.item()})

    torch.save(vgae_model.state_dict(), "vgae.pt")
    wandb.finish()
