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

from mydata import ClassBalancedNodeSplit, MyDataset, create_hidden_train_mask
from mymodels import AttnGCN, TransformNet, TAGConv_3l_512h_w_k3, SAGE
from utils import evaluate_one_by_one, evaluate_batch, evaluate_one_by_one_load_from_file, calc_accuracy, set_global_seed
from utils import inductive_train, to_one_hot


def change_input(x_input, train_edge_index, train_edge_attr_multi):
    x_input = x_input.clone()
    train_edge_index = train_edge_index.clone()
    train_edge_attr_multi = train_edge_attr_multi.clone()

    num_nodes = x_input.size(0)  # Assume data.y contains your node labels
    unknown_label = torch.tensor([0, 0, 0, 0, 0, 1]).type(torch.float)

    # Randomly select 10% of your node indices
    indices = torch.randperm(num_nodes)[:num_nodes // 2]

    # Update the labels of these selected nodes to the unknown label
    x_input[indices] = unknown_label

    # Assuming edge_index is of shape [2, E] and edge_attr is of shape [E, num_labels]
    # where E is the number of edges, and that we have already computed 'indices' of nodes to change

    # Create a mask for edges where the source node label has been changed to unknown
    src_nodes = train_edge_index[0]  # Get the source nodes of all edges
    mask = src_nodes.unsqueeze(1) == indices.unsqueeze(0)  # Compare against changed indices
    mask = mask.any(dim=1)  # Reduce to get a mask for edges

    # Update the edge attributes
    new_edge_attr = torch.zeros_like(train_edge_attr_multi)
    new_edge_attr[mask, -1] = torch.max(train_edge_attr_multi[mask], dim=1)[0]  # Move src label info to the last position
    train_edge_attr_multi[mask] = new_edge_attr[mask]
    return x_input, train_edge_index, train_edge_attr_multi


if __name__ == "__main__":
    set_global_seed(42)

    wandb.init(project="Genomics", entity="kenenbek")

    # Store configurations/hyperparameters
    wandb.config.lr = 0.001
    wandb.config.weight_decay = 5e-4
    wandb.config.epochs = 1000

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

    model = AttnGCN()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)
    scheduler = StepLR(optimizer, step_size=500, gamma=0.1)   # Decay the learning rate by a factor of 0.1 every 10 epochs

    wandb.watch(model, log="all", log_freq=10)

    t = trange(wandb.config.epochs, leave=True)
    losses = []

    # Extract the subgraph associated with the training nodes
    train_nodes = torch.nonzero(train_mask_f).squeeze()
    train_edge_index, train_edge_attr_multi = subgraph(
        train_nodes, full_data.edge_index, edge_attr=full_data.edge_attr_multi, relabel_nodes=True
    )

    for epoch in t:
        model.train()
        optimizer.zero_grad()

        x, edges, attr = change_input(full_data.x_one_hot[train_mask_f], train_edge_index, train_edge_attr_multi)

        out = model(x, edges, attr)
        loss = criterion(out[train_mask_sub], full_data.y[train_mask_h])
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss)
        t.set_description(str(round(loss.item(), 6)))
        wandb.log({"loss": loss.item()})

    # TEST one by one
    y_true, y_pred = evaluate_one_by_one(model, full_data, train_mask_f, test_mask)
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

    # Create a new artifact
    artifact = wandb.Artifact('model-artifact', type='model')
    # Add a file to the artifact (the model file)
    artifact.add_file('attn_full.pt')
    # Log the artifact
    wandb.log_artifact(artifact)

    wandb.finish()
