from matplotlib import pyplot as plt
import torch
import numpy as np
import random
from sklearn import metrics
from tqdm import tqdm, trange
from torch_geometric.nn.models import LabelPropagation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def label_propagation_one_by_one(data, train_mask, test_mask):
    model = LabelPropagation(num_layers=3, alpha=0.8)

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
            training_nodes = torch.arange(sub_data.y.size()[0])
            training_nodes = torch.cat((training_nodes[:test_node_position], training_nodes[test_node_position + 1:]))

            # Predict on the sub-graph
            out = model(y=sub_data.y,
                        edge_index=sub_data.edge_index,
                        mask=training_nodes,
                        edge_weight=sub_data.edge_attr,
                        post_step=None)

            # Use the test_node_position to get the prediction and true label
            pred = out[test_node_position].argmax(dim=0).item()
            true_label = sub_data.y[test_node_position].item()

            y_true_list.append(true_label)
            pred_list.append(pred)

    return y_true_list, pred_list


from scipy.sparse.linalg import eigs
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix


def reconstruct_full(dim, deg, pr, n, m, fr, to):
    selected = np.random.choice(np.arange(n), m, replace=False)
    unselected = np.array(list(set(np.arange(n)) - set(selected)))
    s = (deg / (pr * n * n)) ** 0.25
    W = csr_matrix(([s[x] for x in fr], (fr, to)))
    spd = shortest_path(W, indices=selected)
    pos_inf = (spd == np.inf)
    spd[pos_inf] = 0
    spd[pos_inf] = spd.max()
    selected_spd = spd[:, selected]
    sspd = (selected_spd + selected_spd.T) / 2
    sspd = sspd ** 2
    H = np.eye(m) - np.ones(m) / n
    Ker = - H @ sspd @ H / 2
    w, v = np.linalg.eigh(Ker)
    rec_unnormalized = v[:, -dim:] @ np.diag(w[-dim:])
    rec_orig = np.zeros((n, dim))
    rec_orig[selected] = rec_unnormalized
    # rec_orig[unselected] = rec_unnormalized[spd[:, unselected].argmin(0)]
    return rec_orig


def stationary(A):
    eig = eigs(A.T)
    ind = eig[0].real.argsort()[-1]
    est = eig[1][:, ind].real
    pr = est / est.sum() * A.shape[0]
    return pr


def prep_for_reconstruct(edge_index, n, m, dim=128):
    deg = torch.bincount(edge_index.reshape(-1), minlength=n)
    fr = np.concatenate([edge_index[0].numpy(), edge_index[1].numpy()])
    to = np.concatenate([edge_index[1].numpy(), edge_index[0].numpy()])
    A = csr_matrix(([1 / deg[x] for x in fr], (fr, to)))
    X = torch.tensor([[deg[i], n] for i in range(n)], dtype=torch.float)
    ind = torch.eye(n)[:, torch.randperm(n)[:m]]
    X_extended = torch.hstack([X, ind])

    pr = stationary(A)
    pr = np.maximum(pr, 1e-9)
    rec_orig = reconstruct_full(dim, deg.numpy(), pr, n, m, fr, to)
    rec_orig = torch.FloatTensor(rec_orig)

    return rec_orig


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

            out = model(input_x,
                        sub_data.edge_num, sub_data.edge_index, sub_data.edge_attr)  # NB

            # Use the test_node_position to get the prediction and true label
            pred = out[test_node_position].argmax(dim=0).item()
            true_label = sub_data.y[test_node_position].item()

            y_true_list.append(true_label)
            pred_list.append(pred)

    return y_true_list, pred_list


def gae_evaluate_one_by_one(gae, predictor, data, train_mask, test_mask):
    gae.eval()
    predictor.eval()

    ones_tensor = torch.ones((train_mask.size()[0] + 1, 6))
    # Get the indices of test nodes
    test_indices = torch.where(test_mask)[0].tolist()

    y_true_list = []
    pred_list = []

    gae = gae.to(device)
    predictor = predictor.to(device)
    data = data.to(device)
    train_mask = train_mask.to(device)
    ones_tensor = ones_tensor.to(device)

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

            z = gae.encode(ones_tensor, sub_data.edge_index, sub_data.edge_attr)  # NB
            out = predictor(z, sub_data.edge_index, sub_data.edge_attr)

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
        "Weighted-Averaged F1 Score": f1_w
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


def stationary_torch(A):
    # Convert A to a PyTorch tensor and transpose it
    A_torch = torch.tensor(A).type(torch.float).to(device)

    # Perform eigenvalue decomposition
    eigvals, eigvecs = torch.linalg.eig(A_torch.t())

    # Find the index of the largest real part of the eigenvalues
    _, ind = torch.max(eigvals.real, 0)

    # Extract the corresponding eigenvector and normalize it
    est = eigvecs[:, ind].real
    pr = est / est.sum() * A_torch.shape[0]

    return pr  # Convert back to numpy array if needed


def prep_for_reconstruct_torch(edge_index, n, m, dim=128):
    deg = torch.bincount(edge_index.reshape(-1), minlength=n).float().cuda()
    fr = torch.cat([edge_index[0], edge_index[1]]).cuda()
    to = torch.cat([edge_index[1], edge_index[0]]).cuda()

    # Sparse matrix operations need to be adapted for PyTorch and CUDA
    # A = csr_matrix([...])

    X = torch.stack([deg[i] for i in range(n)], 1).cuda()
    ind = torch.eye(n).cuda()[:, torch.randperm(n)[:m]]
    X_extended = torch.hstack([X, ind]).cuda()

    pr = stationary(A)
    pr = torch.clamp(pr, min=1e-9)
    # rec_orig = reconstruct_full(dim, deg.cpu().numpy(), pr.cpu().numpy(), n, m, fr.cpu().numpy(), to.cpu().numpy())
    # rec_orig = torch.FloatTensor(rec_orig).cuda()

    # return rec_orig


def evaluate_one_by_one_rec(model, data, train_mask, test_mask):
    model.eval()

    # Get the indices of test nodes
    test_indices = torch.where(test_mask)[0].tolist()

    y_true_list = []
    pred_list = []

    model = model.to(device)
    data = data.to(device)
    train_mask = train_mask.to(device)

    with torch.no_grad():
        for i in trange(1132):
            rec_orig, test_position = torch.load(f"recs/rec_{i}.pt")
            sub_data = torch.load(f"recs/sub_data_{i}.pt")

            out = model(rec_orig, sub_data.edge_index, sub_data.edge_attr)  # NB

            # Use the test_node_position to get the prediction and true label
            pred = out[test_position].argmax(dim=0).item()
            true_label = sub_data.y[test_position].item()

            y_true_list.append(true_label)
            pred_list.append(pred)

    return y_true_list, pred_list


import torch
import torch_geometric
from torch_geometric.utils import subgraph, to_networkx
import networkx as nx
import random

import torch
import torch_geometric
from torch_geometric.utils import subgraph, to_networkx
import networkx as nx
import random


def create_connected_subgraph_with_mask_random(data, lower_bound=0.95, upper_bound=1.0):
    """
    Create a connected subgraph from a PyG graph data and return node mask.
    Runs on GPU if available, otherwise on CPU.

    :param data: PyG Data object representing the graph.
    :param lower_bound: Lower bound for the percentage of nodes to include.
    :param upper_bound: Upper bound for the percentage of nodes to include.
    :return: Connected subgraph as a PyG Data object and node mask.
    """

    # Ensure the data is on the correct device
    data = data.to(device)

    # Convert to NetworkX graph
    G = to_networkx(data, to_undirected=True)

    num_nodes = len(G)
    subgraph_size = int(
        num_nodes * torch.rand(1, device=device).item() * (upper_bound - lower_bound) + lower_bound * num_nodes)

    start_node = random.choice(list(G.nodes))
    visited = {start_node}
    queue = [start_node]

    while len(visited) < subgraph_size:
        current = queue.pop(0)
        for neighbor in G.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                if len(visited) == subgraph_size:
                    break

    subgraph_nodes = torch.tensor(list(visited), device=device)
    edge_index, edge_weight = subgraph(subgraph_nodes, data.edge_index,
                                       edge_attr=data.edge_attr, num_nodes=num_nodes, relabel_nodes=True)

    node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    node_mask[subgraph_nodes] = True

    return edge_index, edge_weight, node_mask
