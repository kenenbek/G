import numpy as np
import networkx as nx
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import bernoulli
import networkx as nx
from torch_geometric.data import Data
import torch.nn.functional as F
import torch
from torch_geometric.utils import coalesce, to_undirected

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

EPS = 1e-10


def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())


def generate_matrices(population_sizes, p, teta=5, offset=3.0):
    pop_index = []
    n_pops = len(population_sizes)
    for i in range(n_pops):
        pop_index += [i] * population_sizes[i]

    pop_index = np.array(pop_index)
    print(f"{n_pops=}")
    blocks_sums = [[np.zeros(shape=(population_sizes[i], population_sizes[j])) for i in range(n_pops)] for j in
                   range(n_pops)]
    blocks_counts = [[np.zeros(shape=(population_sizes[i], population_sizes[j])) for i in range(n_pops)] for j in
                     range(n_pops)]

    for pop_i in range(n_pops):
        for pop_j in range(pop_i + 1):
            print(f"{pop_i=} {pop_j=}")
            pop_cross = population_sizes[pop_i] * population_sizes[pop_j]
            bern_samples = bernoulli.rvs(p[pop_i, pop_j], size=pop_cross)
            total_segments = np.sum(bern_samples)
            print(f"{total_segments=}")
            exponential_samples = np.random.exponential(teta, size=total_segments) + offset
            position = 0
            exponential_totals_samples = np.zeros(pop_cross, dtype=np.float64)
            mean_totals_samples = np.zeros(pop_cross, dtype=np.float64)
            exponential_totals_samples[bern_samples == 1] = exponential_samples

            bern_samples = np.reshape(bern_samples, newshape=(population_sizes[pop_i], population_sizes[pop_j]))
            exponential_totals_samples = np.reshape(exponential_totals_samples,
                                                    newshape=(population_sizes[pop_i], population_sizes[pop_j]))
            if (pop_i == pop_j):
                bern_samples = np.tril(bern_samples, -1)
                exponential_totals_samples = np.tril(exponential_totals_samples, -1)
            blocks_counts[pop_i][pop_j] = bern_samples
            blocks_sums[pop_i][pop_j] = exponential_totals_samples
    return np.nan_to_num(symmetrize(np.block(blocks_counts))), np.nan_to_num(
        symmetrize(np.block(blocks_sums))), pop_index


def generate_graph(means, counts, pop_index):
    indiv = list(range(counts.shape[0]))
    G = nx.Graph()
    G.add_nodes_from([(indiv[i], {"means": np.concatenate((means[i], counts[i])), "y": pop_index[i]}) for i in
                      range(len(pop_index))])
    for i in range(counts.shape[0]):
        for j in range(i):
            if (means[i][j]):
                G.add_edge(i, j, len=1 / (counts[i][j] + EPS), weigth=counts[i][j])

    # remove isolated nodes
    # G.remove_nodes_from(list(nx.isolates(G)))
    return G


if __name__ == '__main__':
    real_probs = np.array([[0.11097308, 0.0012959, 0.00127431, 0.00258547, 0.00579579],
                           [0.0012959, 0.01644648, 0.00550097, 0.00708765, 0.00633572],
                           [0.00127431, 0.00550097, 0.01359845, 0.00528573, 0.00422982],
                           [0.00258547, 0.00708765, 0.00528573, 0.01403417, 0.00744347],
                           [0.00579579, 0.00633572, 0.00422982, 0.00744347, 0.01977209]])

    ### code for modelling
    N = 1000
    population_sizes = [N, N, N, N, N]  # [70, 463, 426, 2177, 631]

    means, counts, pop_index = generate_matrices(population_sizes, p=real_probs)
    G = generate_graph(means, counts, pop_index)

    edge_list = []
    edge_attr = []

    for src, trg, weight in G.edges(data='weigth'):
        edge_list.append([src, trg])
        edge_attr.append([weight])

    labels = {node: data['y'] for node, data in G.nodes(data=True)}
    labels = dict(sorted(labels.items()))
    labels = torch.Tensor(list(labels.values())).type(torch.long)

    one_hot_labels = F.one_hot(labels, num_classes=5).type(torch.float)

    edge_index = torch.tensor(edge_list).type(torch.long).T
    edge_attr = torch.tensor(edge_attr).type(torch.float)

    edge_index, edge_attr = coalesce(edge_index, edge_attr, reduce='mean')

    two_edge_index, two_edge_attr = to_undirected(edge_index, edge_attr)
    print(edge_index.shape, "edge_index")
    print(two_edge_index.shape, "two_edge_index")

    data = Data(x=one_hot_labels,
                x_one_hot=one_hot_labels,
                edge_index=two_edge_index,
                edge_attr=two_edge_attr,
                edge_attr_multi=two_edge_attr.clone().detach().repeat(1, 5),
                y=labels)

    # Total number of nodes
    num_nodes = data.x.shape[0]

    # Creating a tensor of all node indices
    node_indices = torch.arange(num_nodes)

    # Shuffle the node indices
    node_indices = node_indices[torch.randperm(num_nodes)]

    # Split indices for train and test sets (70% train, 30% test)
    num_train = int(0.7 * num_nodes)
    train_indices = node_indices[:num_train]
    test_indices = node_indices[num_train:]

    torch.save(data, "fake_data/processed/data_0.pt")
    torch.save(train_indices, "fake_data/train_indices.pt")
    torch.save(test_indices, "fake_data/test_indices.pt")
