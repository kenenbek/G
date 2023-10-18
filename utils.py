from matplotlib import pyplot as plt
import torch
import numpy as np
import random
from sklearn import metrics
from tqdm import tqdm, trange


def evaluate_one_by_one(model, data):
    model.eval()

    train_mask = data.train_mask
    test_mask = data.test_mask

    # Get the indices of test nodes
    test_indices = torch.where(test_mask)[0].tolist()

    y_true_list = []
    pred_list = []

    with torch.no_grad():
        for test_index in trange(len(test_indices)):
            # Get the actual node index
            idx = test_indices[test_index]

            # Combine training indices with the current test index
            sub_indices = torch.cat([torch.where(train_mask)[0], torch.tensor([idx])])
            sub_indices, _ = torch.sort(sub_indices)

            # Extract sub-graph
            sub_data = data.subgraph(sub_indices)

            # Find the position of the test node in the subgraph
            test_node_position = torch.where(sub_indices == idx)[0].item()

            # Predict on the sub-graph
            out = model(sub_data.x_one_hot_hidden, sub_data.edge_index, sub_data.edge_attr)

            # Use the test_node_position to get the prediction and true label
            pred = out[test_node_position].argmax(dim=0).item()
            true_label = sub_data.y[test_node_position].item()

            y_true_list.append(true_label)
            pred_list.append(pred)

    return y_true_list, pred_list


def evaluate_batch(model, full_data, test_mask):
    model.eval()

    out = model(full_data.train_x, full_data.edge_index, full_data.edge_attr)
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

    # Print results
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Micro-Averaged Precision: {precision_micro:.4f}, Recall: {recall_micro:.4f}, F1 Score: {f1_micro:.4f}')
    print(f'Macro-Averaged Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1 Score: {f1_macro:.4f}')

    stats = {
        "Accuracy": accuracy,
        "Micro-Averaged Precision": precision_micro,
        "Micro-Averaged Recall": recall_micro,
        "Micro-Averaged F1 Score": f1_micro,
        "Macro-Averaged Precision": precision_macro,
        "Macro-Averaged Recall": recall_macro,
        "Macro-Averaged F1 Score": f1_macro
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


def evaluate_one_by_one_load_from_file(model, run: int = 0):
    model.eval()
    # Get the indices of test nodes
    test_indices = torch.load(f"full_data/{run}/test_indices.pt")

    y_true_list = []
    pred_list = []

    with torch.no_grad():
        for test_index in trange(len(test_indices)):
            # load sub_data
            sub_data = torch.load(f"full_data/0/test_sub_graph_{test_index}.pt")

            # Predict on the sub-graph
            out = model(sub_data.train_x, sub_data.edge_index, sub_data.edge_attr)

            # Use the test_node_position to get the prediction and true label
            pred = out[sub_data.test_node_position].argmax(dim=0).item()
            true_label = sub_data.y[sub_data.test_node_position].item()

            y_true_list.append(true_label)
            pred_list.append(pred)

    return y_true_list, pred_list
