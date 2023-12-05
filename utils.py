from matplotlib import pyplot as plt
import torch
import numpy as np
import random
from sklearn import metrics
from tqdm import tqdm, trange
from torch_geometric.nn.models import LabelPropagation
from node2vec import Node2Vec
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def fine_tune(sub_data, input_x, test_node_position, model, steps=50):
    with torch.enable_grad():
        model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-3)

        train_mask = torch.ones(sub_data.size(0), dtype=torch.bool)
        train_mask[test_node_position] = False

        for i in range(steps):
            optimizer.zero_grad()

            out = model(input_x,
                        sub_data.big_features,
                        sub_data.edge_index,
                        sub_data.edge_attr
                        )
            loss = criterion(out[train_mask], sub_data.y[train_mask])

            loss.backward()
            optimizer.step()

    return model


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

            preds = []
            for i in range(5):
                # Clean subgraph
                unknown_label = torch.tensor([0, 0, 0, 0, 0]).type(torch.float).to(device)
                unknown_label[i] = 1
                x_input = sub_data.x_one_hot.clone()
                x_input[test_node_position] = unknown_label

                y = sub_data.y.clone()
                y[test_node_position] = i

                _, sub_data_25_filtered = create_25_graphs(y, sub_data.edge_index, sub_data.edge_attr, test=True)

                out = model(x_input, sub_data.big_features, sub_data_25_filtered, sub_data.edge_index, sub_data.edge_attr)  # NB
                #pred = out[test_node_position].argmax(dim=0).item()
                test_node_probs = F.softmax(out[test_node_position], dim=-1)
                pred = test_node_probs[i].item()
                preds.append(pred)

            pred = np.argmax(preds)
            true_label = sub_data.y[test_node_position].item()

            y_true_list.append(true_label)
            pred_list.append(pred)

    return y_true_list, pred_list


def get_neighbors(node_id, edge_index):
    # Returns the indices where the source node (node_id) appears in the edge_index[0]
    return edge_index[1][edge_index[0] == node_id]


def evaluate_batch(model, full_data, test_mask):
    model.eval()

    out = model(None)
    pred = out.argmax(dim=1)

    y_true = full_data.y[test_mask]
    y_pred = pred[test_mask]

    return y_true, y_pred


def calc_accuracy(y_true, y_pred):
    accuracy = metrics.accuracy_score(y_true, y_pred)

    precision_micro = metrics.precision_score(y_true, y_pred, average='micro', zero_division=1)
    recall_micro = metrics.recall_score(y_true, y_pred, average='micro', zero_division=1)
    f1_micro = metrics.f1_score(y_true, y_pred, average='micro', zero_division=1)

    precision_macro = metrics.precision_score(y_true, y_pred, average='macro', zero_division=1)
    recall_macro = metrics.recall_score(y_true, y_pred, average='macro', zero_division=1)
    f1_macro = metrics.f1_score(y_true, y_pred, average='macro', zero_division=1)

    precision_w = metrics.precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall_w = metrics.recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1_w = metrics.f1_score(y_true, y_pred, average='weighted', zero_division=1)

    # Print results
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Micro-Averaged Precision: {precision_micro:.4f}, Recall: {recall_micro:.4f}, F1 Score: {f1_micro:.4f}')
    print(f'Macro-Averaged Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1 Score: {f1_macro:.4f}')
    print(f'Weighted-Averaged Precision: {precision_w:.4f}, Recall: {recall_w:.4f}, F1 Score: {f1_w:.4f}')

    stats = {
        "Accuracy": accuracy,
        "Micro-Averaged Precision": precision_micro,
        "Micro-Averaged Recall": recall_micro,
        "Micro-Averaged F1 Score": f1_micro,
        "Macro-Averaged Precision": precision_macro,
        "Macro-Averaged Recall": recall_macro,
        "Macro-Averaged F1 Score": f1_macro,
        "Weighted-Averaged Precision": precision_w,
        "Weighted-Averaged Recall": recall_w,
        "Weighted-Averaged F1 Score": f1_w,
        "x": f1_w,
    }

    return stats


def set_global_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def filter_graph_by_class(full_data, train_mask, target_class):
    # Step 1: Identify nodes belonging to the target class
    target_nodes = (full_data.y == target_class).nonzero(as_tuple=True)[0]

    # Step 2: Filter edges
    # Extract source and target nodes from edge_index
    src_nodes, dest_nodes = full_data.edge_index
    # Check if the source nodes are in target_nodes
    mask = torch.isin(src_nodes, target_nodes)
    # Select only the edges that satisfy the condition
    filtered_edge_index = full_data.edge_index[:, mask]

    # Step 3: Create a new subgraph
    subgraph = Data(x=full_data.x[train_mask], edge_index=filtered_edge_index, y=full_data.y)
    return subgraph


def create_5_graphs(y, train_edge_index, train_edge_weight):
    sub_data_s = []

    for target_class in range(5):
        target_nodes = (y == target_class).nonzero(as_tuple=True)[0]

        # Extract source and target nodes from edge_index
        src_nodes, dest_nodes = train_edge_index
        # Check if the source nodes are in target_nodes
        mask = torch.isin(src_nodes, target_nodes)

        # Select only the edges that satisfy the condition
        filtered_edge_index = train_edge_index[:, mask]
        filtered_edge_weights = train_edge_weight[mask]

        sub_data_s.append((filtered_edge_index, filtered_edge_weights))

    return sub_data_s


def change_input(x_input, q=10):
    x_input = x_input.clone()
    # train_edge_attr_multi = train_edge_attr_multi.clone()

    num_nodes = x_input.size(0)  # Assume data.y contains your node labels
    unknown_label = torch.tensor([0, 0, 0, 0, 0, 1]).type(torch.float).to(device)

    # Randomly select 10% of your node indices
    indices = torch.randperm(num_nodes)[: int(num_nodes) // q].to(device)

    # Update the labels of these selected nodes to the unknown label
    x_input[indices] = unknown_label

    node_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
    node_mask[indices] = True
    return x_input, node_mask


def create_25_graphs(y, train_edge_index, train_edge_weight, q=0.1, test=False):
    if not test:
        y = y.clone()
        num_labels_to_change = int(len(y) * q)

        # Randomly choose indices to change
        indices_to_change = torch.randperm(len(y))[:num_labels_to_change].to(device)

        # Assign random labels to these indices
        y[indices_to_change] = torch.randint(0, 5, (num_labels_to_change,)).to(device)

    sub_data_s = []
    for src in range(5):
        for dst in range(5):
            nodes_class_src = (y == src).nonzero(as_tuple=True)[0]
            nodes_class_dst = (y == dst).nonzero(as_tuple=True)[0]

            src_nodes, dest_nodes = train_edge_index
            mask = torch.isin(src_nodes, nodes_class_src) & torch.isin(dest_nodes, nodes_class_dst)
            filtered_edge_index = train_edge_index[:, mask]
            filtered_edge_weights = train_edge_weight[mask]

            sub_data_s.append((filtered_edge_index, filtered_edge_weights))
    modified_x_one_hot = F.one_hot(y, num_classes=5).type(torch.float)
    return modified_x_one_hot, sub_data_s

# edge_attr_multi = sub_data.edge_attr_multi.clone()
# mask = sub_data.edge_index[0] == test_node_position
# new_edge_attr = torch.zeros_like(edge_attr_multi).to(device)
# new_edge_attr[mask, -1] = torch.max(edge_attr_multi[mask], dim=1)[0]
# edge_attr_multi[mask] = new_edge_attr[mask]

# model = fine_tune(sub_data, input_x, test_node_position, model, steps=10)
# model.eval()

