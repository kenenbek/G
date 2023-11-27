import wandb

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from torch_geometric.transforms import RandomNodeSplit
import pandas as pd
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import random
from collections import defaultdict
from sklearn.metrics import ConfusionMatrixDisplay
from torch.optim.lr_scheduler import StepLR

from mydata import ClassBalancedNodeSplit, MyDataset, create_hidden_train_mask, recalculate_input_features
from mymodels import AttnGCN, SimpleNN, GCN, GCN_simple
from utils import evaluate_one_by_one, evaluate_batch, calc_accuracy, \
    set_global_seed
from torch_geometric.transforms import GDC
from torch_geometric.utils import to_undirected

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def change_input(x_input, train_edge_index, train_edge_attr_multi):
    x_input = x_input.clone()
    # train_edge_attr_multi = train_edge_attr_multi.clone()

    num_nodes = x_input.size(0)  # Assume data.y contains your node labels
    unknown_label = torch.tensor([0, 0, 0, 0, 0]).type(torch.float).to(device)

    # Randomly select 10% of your node indices
    indices = torch.randperm(num_nodes)[: int(num_nodes) // 10].to(device)

    # Update the labels of these selected nodes to the unknown label
    x_input[indices] = unknown_label

    # Assuming edge_index is of shape [2, E] and edge_attr is of shape [E, num_labels]
    # where E is the number of edges, and that we have already computed 'indices' of nodes to change

    # Create a mask for edges where the source node label has been changed to unknown
    src_nodes = train_edge_index[0]  # Get the source nodes of all edges
    mask = src_nodes.unsqueeze(1) == indices.unsqueeze(0)  # Compare against changed indices
    mask = mask.any(dim=1)  # Reduce to get a mask for edges

    # Update the edge attributes
    # new_edge_attr = torch.zeros_like(train_edge_attr_multi).to(device)
    # new_edge_attr[mask, -1] = torch.max(train_edge_attr_multi[mask], dim=1)[0]  # Move src label info to the last position
    # train_edge_attr_multi[mask] = new_edge_attr[mask]

    node_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
    node_mask[indices] = True
    return x_input, train_edge_attr_multi, node_mask


if __name__ == "__main__":
    set_global_seed(42)
    wandb.init(project="Genomics", entity="kenenbek")

    # Store configurations/hyperparameters
    wandb.config.lr = 0.001
    wandb.config.weight_decay = 5e-3
    wandb.config.epochs = 1200

    full_dataset = MyDataset(root="fake_data/")
    full_data = full_dataset[0]
    num_nodes = full_data.y.shape[0]
    train_indices = torch.load("fake_data/train_indices.pt")
    test_indices = torch.load("fake_data/test_indices.pt")

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_indices] = True
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[test_indices] = True

    assert torch.equal(train_mask, ~test_mask), "Error"

    _, edge_num, big_features = recalculate_input_features(full_data, train_mask)
    full_data.edge_num = edge_num
    full_data.big_features = big_features

    model = AttnGCN()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)
    scheduler = StepLR(optimizer, step_size=500,
                       gamma=0.1)  # Decay the learning rate by a factor of 0.1 every 10 epochs

    # wandb.watch(model, log="all", log_freq=10)

    t = trange(wandb.config.epochs, leave=True)
    losses = []

    # Extract the subgraph associated with the training nodes
    # train_edge_index, train_edge_attr_multi = subgraph(
    #     train_nodes, full_data.edge_index, edge_attr=full_data.edge_attr_multi, relabel_nodes=True
    # )

    train_edge_index, train_edge_weight = subgraph(
        train_indices, full_data.edge_index, edge_attr=full_data.edge_attr, relabel_nodes=True
    )
    train_subgraph = full_data.subgraph(train_indices)

    model = model.to(device)
    full_data = full_data.to(device)
    train_mask = train_mask.to(device)
    train_edge_index = train_edge_index.to(device)
    train_edge_weight = train_edge_weight.to(device)
    # train_edge_attr_multi = train_edge_attr_multi.to(device)

    for epoch in t:
        model.train()
        optimizer.zero_grad()

        x_input, _, node_mask = change_input(full_data.x_one_hot[train_mask], train_edge_index, None)

        out = model(x_input,
                    train_edge_index, train_edge_weight)
        loss = criterion(out[node_mask], full_data.y[train_mask][node_mask])

        loss.backward()
        optimizer.step()
        scheduler.step()

        wandb.log({"loss": loss.item()})
        losses.append(loss)
        t.set_description(str(round(loss.item(), 6)))

    # TEST one by one
    y_true, y_pred = evaluate_one_by_one(model, full_data, train_mask, test_mask)
    metrics = calc_accuracy(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 10))
    sub_etnos = ["1", "2", "3", "4", "5"]

    cm_display = ConfusionMatrixDisplay.from_predictions(y_true,
                                                         y_pred,
                                                         display_labels=sub_etnos,
                                                         ax=ax)
    fig.savefig("confusion_matrix.png")  # Save the figure to a file
    wandb.log({"confusion_matrix": wandb.Image("confusion_matrix.png")})  # Log the saved figure to wandb
    wandb.log(metrics)

    torch.save(model.state_dict(), "attn_full.pt")
    torch.save(y_true, "y_true.pt")
    torch.save(y_pred, "y_pred.pt")

    # Create a new artifact
    # artifact = wandb.Artifact('model-artifact', type='model')
    # # Add a file to the artifact (the model file)
    # artifact.add_file('attn_full.pt')
    # # Log the artifact
    # wandb.log_artifact(artifact)

    wandb.finish()

#
# r_edge_index, r_edge_weight, r_mask = create_connected_subgraph_with_mask_random(train_subgraph)
#
#         x, attr, noisy_mask = change_input(full_data.x_one_hot[train_mask][r_mask],
#                                            r_edge_index, None)