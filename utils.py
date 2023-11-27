from matplotlib import pyplot as plt
import torch
import numpy as np
import random
from sklearn import metrics
from tqdm import tqdm, trange
from torch_geometric.nn.models import LabelPropagation
from node2vec import Node2Vec


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

            # Clean subgraph
            unknown_label = torch.tensor([0, 0, 0, 0, 0]).type(torch.float).to(device)
            input_x = sub_data.x_one_hot.clone()
            input_x[test_node_position] = unknown_label
            # edge_attr_multi = sub_data.edge_attr_multi.clone()
            # mask = sub_data.edge_index[0] == test_node_position
            # new_edge_attr = torch.zeros_like(edge_attr_multi).to(device)
            # new_edge_attr[mask, -1] = torch.max(edge_attr_multi[mask], dim=1)[0]
            # edge_attr_multi[mask] = new_edge_attr[mask]

            #model = fine_tune(sub_data, input_x, test_node_position, model, steps=10)
            #model.eval()

            out = model(sub_data.big_features,
                        sub_data.edge_index,
                        sub_data.edge_attr)  # NB

            # Use the test_node_position to get the prediction and true label
            pred = out[test_node_position].argmax(dim=0).item()
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


def node2vec():
    node2vec = Node2Vec(nx_graph, dimensions=64, walk_length=30, num_walks=200)

















