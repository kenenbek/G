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
from mymodels import AttnGCN, SimpleNN, GCN, GCN_simple, BigAttn
from utils import calc_accuracy, set_global_seed
from torch_geometric.utils import to_undirected
from torch_geometric.utils import to_dense_adj

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_one_by_one(model, data, train_mask, test_mask):
    model.eval()

    # Get the indices of test nodes
    test_indices = torch.where(test_mask)[0].tolist()

    y_true_list = []
    pred_list = []

    model = model.to(device)
    data = data.to(device)
    train_mask = train_mask.to(device)

    with torch.no_grad():
        for test_index in trange(len(test_indices)):
            # Get the actual node index
            idx = test_indices[test_index]

            # Combine training indices with the current test index
            sub_indices = torch.cat([torch.where(train_mask)[0], torch.tensor([idx]).to(device)])
            sub_indices, _ = torch.sort(sub_indices)

            # Extract sub-graph
            sub_data = data.subgraph(sub_indices)

            # Find the position of the test node in the subgraph
            test_node_position = torch.where(sub_indices == idx)[0].item()

            weighted_adj_matrix = to_dense_adj(sub_data.edge_index, edge_attr=sub_data.edge_attr)[0].squeeze(2)

            single = weighted_adj_matrix[test_node_position]
            single = torch.cat((single[:test_node_position], single[test_node_position + 1:]))

            out = model(single)

            pred = out.argmax(dim=0).item()
            true_label = sub_data.y[test_node_position].item()

            y_true_list.append(true_label)
            pred_list.append(pred)

    return y_true_list, pred_list


if __name__ == "__main__":
    set_global_seed(42)
    wandb.init(project="Genomics", entity="kenenbek")

    # Store configurations/hyperparameters
    wandb.config.lr = 0.001
    wandb.config.weight_decay = 5e-3
    wandb.config.epochs = 10

    full_dataset = MyDataset(root="full_data/")
    full_data = full_dataset[0]
    num_nodes = full_data.y.shape[0]
    train_indices = torch.load("full_data/0/train_indices.pt")
    test_indices = torch.load("full_data/0/test_indices.pt")

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_indices] = True
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[test_indices] = True

    assert torch.equal(train_mask, ~test_mask), "Error"

    _, edge_num, big_features, neighbors = recalculate_input_features(full_data, train_mask)
    full_data.edge_num = edge_num
    full_data.big_features = big_features

    model = SimpleNN()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)
    scheduler = StepLR(optimizer, step_size=500,
                       gamma=0.1)  # Decay the learning rate by a factor of 0.1 every 10 epochs

    t = trange(wandb.config.epochs, leave=True)
    losses = []

    train_edge_index, train_edge_weight = subgraph(
        train_indices, full_data.edge_index, edge_attr=full_data.edge_attr, relabel_nodes=True
    )

    model = model.to(device)
    full_data = full_data.to(device)
    train_mask = train_mask.to(device)
    train_edge_index = train_edge_index.to(device)
    train_edge_weight = train_edge_weight.to(device)

    weighted_adj_matrix = to_dense_adj(train_edge_index, edge_attr=train_edge_weight)[0].squeeze(2)

    for epoch in t:
        model.train()
        optimizer.zero_grad()

        out = model(weighted_adj_matrix)
        loss = criterion(out, full_data.y[train_mask])

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

    wandb.finish()


