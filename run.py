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



from mydata import ClassBalancedNodeSplit, MyDataset
from mymodels import AttnGCN
from utils import evaluate_one_by_one, calc_accuracy, set_global_seed

if __name__ == "__main__":
    set_global_seed(42)

    wandb.init(project="Genomics", entity="kenenbek")

    # Store configurations/hyperparameters
    wandb.config.lr = 0.001
    wandb.config.weight_decay = 5e-4
    wandb.config.epochs = 1000

    class_balanced_split = ClassBalancedNodeSplit(train=0.7, val=0.0, test=0.3)
    full_dataset = MyDataset(root="full_data/", transform=class_balanced_split)
    
    full_data = full_dataset[0]

    model = AttnGCN()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)
    
    wandb.watch(model, log="all", log_freq=10)

    
    t = trange(wandb.config.epochs, leave=True)
    losses = []

    hidden_train_mask_sub, hidden_train_mask_full = full_data.create_hidden_train_mask(hide_frac=0.0)

    train_nodes_full = torch.nonzero(full_data.train_mask).squeeze()
    train_nodes_sub = torch.nonzero(full_data.hidden_train_mask_full).squeeze()

    full_data.recalculate_one_hot()
    
    
    # Extract the subgraph associated with the training nodes
    train_edge_index, train_edge_weight = subgraph(
        train_nodes_full, full_data.edge_index, edge_attr=full_data.edge_attr, relabel_nodes=True
    )
    
    for epoch in t:
        model.train()
        optimizer.zero_grad()
            
        out = model(full_data.x_one_hot_hidden[full_data.train_mask], train_edge_index, train_edge_weight)
        
        loss = criterion(out[hidden_train_mask_sub], full_data.y[hidden_train_mask_full])
        loss.backward()  
        optimizer.step()
        
        losses.append(loss)
        t.set_description(str(round(loss.item(), 6)))
    
        wandb.log({"loss": loss.item()})



    # TEST
    y_true, y_pred = evaluate_one_by_one(model, full_data)
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

    torch.save(model, "attn_full.pt")

    # Create a new artifact
    artifact = wandb.Artifact('model-artifact', type='model')
    # Add a file to the artifact (the model file)
    artifact.add_file('attn_full.pt')
    # Log the artifact
    wandb.log_artifact(artifact)

    
    wandb.finish()
