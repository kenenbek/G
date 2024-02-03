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
import argparse
from mydata import ClassBalancedNodeSplit, MyDataset, create_hidden_train_mask, recalculate_input_features
from mymodels import AttnGCN, GINNet, SimpleNN, GCN, TAGConv_3l_512h_w_k3, GCN_simple, BigAttn, Transformer
from utils import evaluate_one_by_one, evaluate_batch, calc_accuracy, \
    set_global_seed, change_input, create_5_graphs, create_25_graphs
from torch_geometric.transforms import GDC
from torch_geometric.utils import to_undirected
from builtins import NotImplementedError

parser = argparse.ArgumentParser(description='Your script description here.')
parser.add_argument('--model', type=str, help='Specify the model (e.g., attention)')
parser.add_argument('--data', type=str, help='Specify the model (e.g., attention)')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

set_global_seed(42)

for k in range(10):

    if args.data == "nc":
        path = "nc"
    elif args.data == "cr":
        path = "cr"
    else:
        raise NotImplementedError()

    full_dataset = MyDataset(root=f"full_data/{path}/", dataset=path)
    full_data = full_dataset[0]
    num_nodes = full_data.y.shape[0]
    train_indices = torch.load(f"full_data/{path}/{k}/train_indices.pt")
    val_indices = torch.load(f"full_data/{path}/{k}/val_indices.pt")
    test_indices = torch.load(f"full_data/{path}/{k}/test_indices.pt")

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_indices] = True

    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[val_indices] = True

    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[test_indices] = True

    _, edge_num, big_features, neighbors = recalculate_input_features(full_data, train_mask)
    full_data.edge_num = edge_num
    full_data.big_features = big_features

    if args.model == "attn":
        model = AttnGCN()
        output = "attn"
    elif args.model == "tagconv":
        model = TAGConv_3l_512h_w_k3()
        output = "tagconv"
    elif args.model == "gcn":
        model = GCN()
        output = "gcn"
    elif args.model == "gin":
        model = GINNet()
        output = "gin"
    else:
        raise NotImplementedError

    best_model = None

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-3)
    scheduler = StepLR(optimizer, step_size=500,
                       gamma=0.1)  # Decay the learning rate by a factor of 0.1 every 10 epochs


    t = trange(5000, leave=True)
    losses = []

    train_indices = train_indices.to(device)
    full_data = full_data.to(device)

    train_edge_index, train_edge_weight = subgraph(
        train_indices, full_data.edge_index, edge_attr=full_data.edge_attr, relabel_nodes=True
    )

    model = model.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    train_edge_index = train_edge_index.to(device)
    train_edge_weight = train_edge_weight.to(device)

    best_metric = -1000

    for epoch in t:
        model.train()
        optimizer.zero_grad()

        input_data, rand_mask = change_input(full_data.x_one_hot[train_mask], q=100)

        out = model(input_data,
                    train_edge_index, train_edge_weight)
        loss = criterion(out[rand_mask], full_data.y[train_mask][rand_mask])

        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss)
        t.set_description(str(round(loss.item(), 6)))

        if epoch % 200 == 0:
            y_true, y_pred = evaluate_one_by_one(model, full_data,
                                                 train_mask, val_mask, disable=True)
            metrics = calc_accuracy(y_true, y_pred)

            if metrics["6"] > best_metric:
                best_metric = metrics["6"]
                best_model = model
                print("Epoch: ", epoch)
                print(metrics)

    # TEST one by one

    y_true, y_pred = evaluate_one_by_one(best_model, full_data,
                                         train_mask, test_mask)
    metrics = calc_accuracy(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 10))
    sub_etnos = ["1", "2", "3", "4", "5", "6", "7", "8"]

    cm_display = ConfusionMatrixDisplay.from_predictions(y_true,
                                                         y_pred,
                                                         display_labels=sub_etnos,
                                                         ax=ax)
    fig.savefig(f"models/{path}/{args.model}/confusion_matrix_{k}.png")  # Save the figure to a file

    torch.save(best_model.state_dict(), f"models/{path}/{args.model}/model_{k}.pt")
    torch.save(y_true, f"models/{path}/{args.model}/y_true_{k}.pt")
    torch.save(y_pred, f"models/{path}/{args.model}/y_pred_{k}.pt")

    results = str(metrics["0"]) + ", " + str(metrics["1"]) + ", " + str(metrics["2"]) + ", " + str(metrics["3"]) + ", " + str(metrics["4"]) + ", " + str(metrics["5"]) + ", " + str(metrics["6"]) + ", " + str(metrics["7"]) + ", " + str(metrics["8"]) + ", " + str(metrics["9"])

    with open(f"models/{path}/{args.model}/results.csv", "a") as file:
        file.write(results + "\n")
