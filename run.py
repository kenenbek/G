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

from mydata import ClassBalancedNodeSplit, MyDataset, create_hidden_train_mask
from mymodels import AttnGCN
from utils import evaluate_one_by_one, evaluate_one_by_one_load_from_file, calc_accuracy, set_global_seed

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

    train_mask_sub, train_mask_h = create_hidden_train_mask(train_indices_full, num_nodes, hide_frac=0.0)
    full_data.recalculate_input_features(train_mask_h)

    model = AttnGCN()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)

    wandb.watch(model, log="all", log_freq=10)

    t = trange(wandb.config.epochs, leave=True)
    losses = []

    # Extract the subgraph associated with the training nodes
    train_nodes = torch.nonzero(train_mask_f).squeeze()
    train_edge_index, train_edge_weight = subgraph(
        train_nodes, full_data.edge_index, edge_attr=full_data.edge_attr, relabel_nodes=True
    )

    for epoch in t:
        model.train()
        optimizer.zero_grad()

        out = model(full_data.train_x[train_mask_f], train_edge_index, train_edge_weight)
        loss = criterion(out[train_mask_sub], full_data.y[train_mask_h])
        loss.backward()
        optimizer.step()

        losses.append(loss)
        t.set_description(str(round(loss.item(), 6)))
        wandb.log({"loss": loss.item()})

    # TEST
    y_true, y_pred = evaluate_one_by_one_load_from_file(model)
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
