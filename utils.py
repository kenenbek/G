from matplotlib import pyplot as plt
import torch
import numpy as np
import random
from sklearn import metrics
from tqdm import tqdm, trange


def evaluate_one_by_one(model, data, train_mask, test_mask):
    model.eval()

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
            out = model(sub_data.train_x, sub_data.edge_index, sub_data.edge_attr)

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


def to_one_hot(labels):
    num_classes = labels.max().item() + 1  # assuming labels are 0-indexed
    one_hot = torch.zeros(labels.size(0), num_classes)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return one_hot


def inductive_train(model, optimizer, scheduler, criterion, data, train_mask):
    y = full_data.y[train_mask_h]
    x_one_hot = to_one_hot(full_data.y[train_mask_h])
    for epoch in t:
        model.train()

        total_loss = 0
        for node in range(len(train_nodes)):  # Loop over nodes in the training mask
            optimizer.zero_grad()
            # Zero out the feature of the current node
            saved_features = x_one_hot[node]
            x_one_hot[node] = torch.zeros(5)

            # Get output
            out = model(x_one_hot, train_edge_index, train_edge_weight)

            # Compute the loss for the current node
            loss = criterion(out[node].unsqueeze(0), y[node].unsqueeze(0))
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item()

            # Restore the original features for the next iteration
            x_one_hot[node] = saved_features

        scheduler.step()
        losses.append(total_loss)
        t.set_description(str(round(loss.item(), 6)))
        wandb.log({"loss": loss.item()})


def ordinary_training(model, optimizer, scheduler, data, criterion):
    model.train()
    optimizer.zero_grad()

    out = model(data.train_x[train_mask_f], train_edge_index, train_edge_weight)
    loss = criterion(out[train_mask_sub], data.y[train_mask_h])
    loss.backward()
    optimizer.step()
    scheduler.step()

    losses.append(loss)
    t.set_description(str(round(loss.item(), 6)))
    wandb.log({"loss": loss.item()})
