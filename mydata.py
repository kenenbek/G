import os.path as osp
from collections import defaultdict
import pandas as pd
from tqdm import tqdm, trange
import torch
from torch_geometric.data import Data, Dataset, download_url
from torch_geometric.transforms import BaseTransform
import torch.nn.functional as F
import networkx as nx
import numpy as np
import random
from builtins import NotImplementedError

num_classes = {
    "we": 4,
    "sim_we": 4,
    "scand": 3,
    "volga": 4,
    "nc": 8,
    "sim_nc": 8,
}

class MyDataset(Dataset):
    def __init__(self, root, dataset, transform=None, pre_transform=None, pre_filter=None):
        datasets = {"cr", "nc", "we", "scand", "volga", "sim_nc", "sim_we"}
        assert dataset in datasets, "Incorrect name for dataset"
        self.dataset = dataset
        self.class_num = None
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        if self.dataset == "cr":
            return ["CR_graph_rel.csv"]
        elif self.dataset == "nc":
            return ["NC_graph_rel.csv"]
        elif self.dataset == "sim_nc":
            return ["simulated_nc.csv"]
        elif self.dataset == "we":
            return ["Western-Europe_weights_partial_labels.csv"]
        elif self.dataset == "sim_we":
            return ["simulated_we.csv"]
        elif self.dataset == "scand":
            return ["Scandinavia_weights_partial_labels.csv"]
        elif self.dataset == "volga":
            return ["Volga_weights_partial_labels.csv"]

    @property
    def processed_file_names(self):
        return ['data_0.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        ind = {}
        if self.dataset == "cr":
            ind = {
                'мордвины': 0,
                'белорусы': 1,
                'украинцы': 2,
                'южные-русские': 3,
                'северные-русские': 4
            }
        elif self.dataset == "nc" or self.dataset == "sim_nc":
            ind = {'карачаевцы,балкарцы': 0,
                 'осетины': 1,
                 'кабардинцы,черкесы,адыгейцы': 2,
                 'ингуши': 3,
                 'кумыки': 4,
                 'ногайцы': 5,
                 'чеченцы': 6,
                 'дагестанские народы': 7,
            }
        elif self.dataset == "we" or self.dataset == "sim_we":
            ind = {
                "English": 0,
                "Germans": 1,
                "French": 2,
                # "Russians": 3,
                # "Ashkenazim": 4,
                "Belgium": 3,
                # "Finns": 6,
                # "Norwegians": 7,
                # "Tatars,Volga-Tatars,Mishar-Tatars,Kryashens": 8,
                # "Swedes": 9,
                # "Lithuanians": 10,
                # "Danes": 11,
                # "Ukrainians": 12,
                # "Belarusians": 13,
                # "Puerto-Ricans": 14,
                # "Chuvash": 15,
                # "Bashkirs": 16,
                # "Poles": 17,
                # "Irish": 18,
                # "Tuscans": 19,
                # "Balkan": 20,
                # "Spaniards": 21,
            }
        elif self.dataset == "scand":
            ind = {
                # "Finns": 0,
                # "English": 1,
                "Norwegians": 0,
                "Swedes": 1,
                # "Tatars,Volga-Tatars,Mishar-Tatars,Kryashens": 4,
                # "Russians": 5,
                "Danes": 2,
                # "Ashkenazim": 7,
                # "Germans": 8,
                # "Chuvash": 9,
                # "Karelians,Veps": 4,
                # "Belgium": 11,
                # "Bashkirs": 12,
                # "Lithuanians": 13,
                # "Ukrainians": 14,
                # "Belarusians": 15,
                # "Estonians": 5,
            }
        elif self.dataset == "volga":
            ind = {
                "Tatars,Volga-Tatars,Mishar-Tatars,Kryashens": 0,
                # "Russians": 1,
                # "Finns": 2,
                # "Bashkirs": 2,
                "Chuvash": 1,
                # "Lithuanians": 5,
                # "Germans": 6,
                # "Ukrainians": 7,
                # "Belarusians": 8,
                # "Swedes": 9,
                # "Ashkenazim": 4,
                # "Kazakhs": 5,
                # "Dolgans,Yakuts": 12,
                "Udmurts,Besermyan": 2,
                # "Mordvins": 5,
                # "Norwegians": 15,
                # "Karelians,Veps": 16,
                # "English": 17,
                # "Poles": 18,
                "Mari": 3,
                # "Buryats,Hamnigan,Mongols": 20,
                # "Khanty,Mansi": 21,
                # "Komi": 22,
                # "Estonians": 23,
                # "Kyrgyz": 24,
                # "Kabardians,Cherkess,Adygeans": 25,
                # "Balkan": 26,
            }
        else:
            NotImplementedError()
        self.class_num = len(ind)
        idx = 0
        for raw_path in self.raw_paths:
            edge_index = []
            edge_attr = []
            edge_attr_multi = []

            y_labels = {}

            x_data = defaultdict(lambda: (self.class_num * [0]))
            edge_num = defaultdict(lambda: (self.class_num * [0]))

            dataset_csv = pd.read_csv(raw_path)
            for index, row in tqdm(dataset_csv.iterrows()):
                node1 = row["node_id1"]
                node2 = row["node_id2"]
                label1 = row["label_id1"]
                label2 = row["label_id2"]
                ibd_sum = row["ibd_sum"]

                if not (label1 in ind and label2 in ind):
                    continue

                id1 = int(node1[5:])
                id2 = int(node2[5:])

                x_data[id1][ind[label2]] += ibd_sum
                x_data[id2][ind[label1]] += ibd_sum
                edge_num[id1][ind[label2]] += 1
                edge_num[id2][ind[label1]] += 1

                edge_index.append([id1, id2])
                edge_index.append([id2, id1])
                edge_attr.append([ibd_sum])
                edge_attr.append([ibd_sum])

                y_labels[id1] = ind[label1]
                y_labels[id2] = ind[label2]

                # eam1 = [0, 0, 0, 0, 0, 0]
                # eam2 = [0, 0, 0, 0, 0, 0]
                # eam1[ind[label1]] = ibd_sum
                # eam2[ind[label2]] = ibd_sum
                # edge_attr_multi.append(eam1)
                # edge_attr_multi.append(eam2)

            y_labels = dict(sorted(y_labels.items()))
            y = torch.Tensor(list(y_labels.values())).type(torch.long)

            x_data = dict(sorted(x_data.items()))
            x = torch.Tensor(list(x_data.values())).type(torch.float)
            edge_num = dict(sorted(edge_num.items()))
            edge_num = torch.Tensor(list(edge_num.values())).type(torch.float)

            edge_attr = torch.Tensor(edge_attr).type(torch.float).contiguous()
            #edge_attr_multi = torch.Tensor(edge_attr_multi).type(torch.float).contiguous()
            edge_index = torch.Tensor(edge_index).type(torch.long).t().contiguous()

            x_one_hot = F.one_hot(y, num_classes=int(y.max()) + 1).type(torch.float)

            data = MyData(x=x,
                          edge_num=edge_num,
                          edge_index=edge_index,
                          edge_attr=edge_attr,
                          #edge_attr_multi=edge_attr_multi,
                          x_one_hot=x_one_hot,
                          y=y,
                          dataset=self.dataset,
                          class_num=len(ind))

            if not data.validate(raise_on_error=False):
                mapping, edge_index = fix_edge_index(edge_index)
                data.edge_index = edge_index

                torch.save(mapping, osp.join(self.processed_dir, f'mapping_indices.pt'))
                assert data.validate()

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data


def fix_edge_index(edge_index):
    unique_nodes = torch.unique(edge_index)
    mapping = {old_node.item(): new_node for new_node, old_node in enumerate(unique_nodes)}

    # Apply the mapping to the edge_index
    edge_index_fixed = torch.tensor([[mapping[edge.item()] for edge in row] for row in edge_index])

    return mapping, edge_index_fixed

def recalculate_input_features(full_data, train_mask):
    available_node_indices = torch.nonzero(train_mask).squeeze()
    known_training_set = set(available_node_indices.tolist())

    hidden_x_data = {}
    edge_num = {}
    neighbors = {}

    means = {}
    stds = {}
    edges = {}

    for i in range(full_data.x.shape[0]):
        hidden_x_data[i] = [0 for _ in range(full_data.class_num)]
        edge_num[i] = [0 for _ in range(full_data.class_num)]
        neighbors[i] = [[] for _ in range(full_data.class_num)]

        means[i] = []
        stds[i] = []
        edges[i] = []

    for i, edge in tqdm(enumerate(full_data.edge_index.t())):
        start_node = edge[0].item()
        dest_node = edge[1].item()

        start_ethnicity = full_data.y[start_node].item()

        if start_node in known_training_set:
            hidden_x_data[dest_node][start_ethnicity] += full_data.edge_attr[i].item()
            edge_num[dest_node][start_ethnicity] += 1
            neighbors[dest_node][start_ethnicity].append(full_data.edge_attr[i].item())

    for i in range(full_data.x.shape[0]):
        for j in range(len(neighbors[i])):
            if len(neighbors[i][j]) == 0:
                means[i].append(0)
                stds[i].append(0)
                edges[i].append(0)
            else:
                means[i].append(np.mean(neighbors[i][j]))
                stds[i].append(np.std(neighbors[i][j]))
                edges[i].append(len(neighbors[i][j]))

    hidden_x_data = convert_dict_to_tensor(hidden_x_data)
    edge_num = convert_dict_to_tensor(edge_num)
    means = convert_dict_to_tensor(means)
    stds = convert_dict_to_tensor(stds)
    edges = convert_dict_to_tensor(edges)

    return hidden_x_data, edge_num, torch.cat((means, stds, edges), dim=-1), neighbors


def convert_dict_to_tensor(d):
    d = dict(sorted(d.items()))
    d = torch.Tensor(list(d.values())).contiguous()
    return d

from torch_geometric.transforms import BaseTransform


class ClassBalancedNodeSplit(BaseTransform):
    def __init__(self, train, val, test):
        assert train + val + test == 1.0, "The sum of train, val, and test ratios should be 1.0"
        self.train = train
        self.val = val
        self.test = test

    def __call__(self, data):
        y = data.y
        num_nodes = y.size(0)

        # Initialize masks
        train_mask = torch.zeros(num_nodes, dtype=bool)
        val_mask = torch.zeros(num_nodes, dtype=bool)
        test_mask = torch.zeros(num_nodes, dtype=bool)

        num_classes = int(y.max()) + 1
        for c in range(num_classes):
            class_indices = (y == c).nonzero().squeeze().tolist()

            num_train_samples = int(len(class_indices) * self.train)
            num_val_samples = int(len(class_indices) * self.val)
            num_test_samples = len(class_indices) - num_train_samples - num_val_samples

            # Randomly shuffle class indices
            random_indices = torch.randperm(len(class_indices)).tolist()
            train_class_indices = [class_indices[i] for i in random_indices[:num_train_samples]]
            val_class_indices = [class_indices[i] for i in
                                 random_indices[num_train_samples:num_train_samples + num_val_samples]]
            test_class_indices = [class_indices[i] for i in random_indices[num_train_samples + num_val_samples:]]

            # Assign to masks
            train_mask[train_class_indices] = True
            val_mask[val_class_indices] = True
            test_mask[test_class_indices] = True

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        return data

    def __repr__(self):
        return '{}(train={}, val={}, test={})'.format(self.__class__.__name__, self.train, self.val, self.test)


class MyData(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_x = None
        self.hidden_train_mask = None
        self.x_one_hot_hidden = None

    def create_hidden_train_mask(self, hide_frac=0.0):
        """
        Generate two masks:
        - One that hides a fraction of the True values within the subsetted training data.
        - Another that hides the same fraction of the True values but within the full data.
        """
        # Get the indices of the True values in the training mask
        train_indices_full = torch.nonzero(self.train_mask).squeeze()

        # Number of training samples
        num_train_samples = train_indices_full.size(0)

        # Determine the number of nodes to hide
        num_to_hide = int(hide_frac * num_train_samples)

        # Randomly select relative nodes to hide from the subset of training data
        hide_relative_indices = torch.randperm(num_train_samples)[:num_to_hide]

        # Create a new mask for the subsetted training data
        hidden_train_mask_subset = torch.ones(num_train_samples, dtype=torch.bool)

        # Set the mask value of the selected nodes to False
        hidden_train_mask_subset[hide_relative_indices] = False

        # Create a new mask for the full data
        hidden_train_mask_full = self.train_mask.clone()

        # Convert relative hide indices to full data indices
        hide_full_indices = train_indices_full[hide_relative_indices]

        # Set the mask value of the selected nodes to False in the full mask
        hidden_train_mask_full[hide_full_indices] = False

        self.hidden_train_mask_subset = hidden_train_mask_subset
        self.hidden_train_mask_full = hidden_train_mask_full
        return hidden_train_mask_subset, hidden_train_mask_full

    def recalculate_one_hot(self):
        assert self.hidden_train_mask_subset is not None, "Error"
        assert self.hidden_train_mask_full is not None, "Error"

        hidden_x_data = {}

        for i in range(self.x_one_hot.shape[0]):
            if self.hidden_train_mask_full[i] or self.test_mask[i]:
                hidden_x_data[i] = list(self.x_one_hot[i])
            else:
                hidden_x_data[i] = [0, 0, 0, 0, 0]

        hidden_x_data = dict(sorted(hidden_x_data.items()))
        hidden_x_data = torch.Tensor(list(hidden_x_data.values())).contiguous()

        self.x_one_hot_hidden = hidden_x_data


def generate_train_test_indices(y, run=10, train=.6, val=.2, test=.2, path=None):
    num_nodes = y.size(0)
    num_classes = int(y.max()) + 1

    for i in range(run):
        # Initialize masks
        train_mask = torch.zeros(num_nodes, dtype=bool)
        val_mask = torch.zeros(num_nodes, dtype=bool)
        test_mask = torch.zeros(num_nodes, dtype=bool)

        for c in range(num_classes):
            class_indices = (y == c).nonzero().squeeze().tolist()

            num_train_samples = int(len(class_indices) * train)
            num_val_samples = int(len(class_indices) * val)
            num_test_samples = len(class_indices) - num_train_samples - num_val_samples

            # Randomly shuffle class indices
            random_indices = torch.randperm(len(class_indices)).tolist()
            train_class_indices = [class_indices[i] for i in random_indices[:num_train_samples]]
            val_class_indices = [class_indices[i] for i in
                                 random_indices[num_train_samples:num_train_samples + num_val_samples]]
            test_class_indices = [class_indices[i] for i in random_indices[num_train_samples + num_val_samples:]]

            # Assign to masks
            train_mask[train_class_indices] = True
            val_mask[val_class_indices] = True
            test_mask[test_class_indices] = True

        train_indices = torch.nonzero(train_mask).squeeze()
        val_indices = torch.nonzero(val_mask).squeeze()
        test_indices = torch.nonzero(test_mask).squeeze()

        torch.save(train_indices, f"full_data/{path}/{i}/train_indices.pt")
        torch.save(val_indices, f"full_data/{path}/{i}/val_indices.pt")
        torch.save(test_indices, f"full_data/{path}/{i}/test_indices.pt")

    return


def generate_subgraphs():
    full_dataset = MyDataset(root="full_data/")
    full_data = full_dataset[0]

    for i in range(10):
        train_indices = torch.load(f"full_data/{i}/train_indices.pt")
        test_indices = torch.load(f"full_data/{i}/test_indices.pt")

        train_mask_f = torch.zeros(full_data.y.shape[0], dtype=torch.bool)
        train_mask_f[train_indices] = True

        full_data.recalculate_input_features(train_mask_f)

        for test_index in trange(len(test_indices)):
            # Get the actual node index
            idx = test_indices[test_index]

            # Combine training indices with the current test index
            sub_indices = torch.cat([train_indices, torch.tensor([idx])])
            sub_indices, _ = torch.sort(sub_indices)

            # Extract sub-graph
            sub_data = full_data.subgraph(sub_indices)

            # Find the position of the test node in the subgraph
            test_node_position = torch.where(sub_indices == idx)[0].item()
            sub_data.test_node_position = test_node_position
            torch.save(sub_data, f"full_data/{i}/test_sub_graph_{test_index}.pt")


def create_hidden_train_mask(train_indices_full, num_nodes_full, hide_frac=0.0):
    """
    Generate two masks:
    - One that hides a fraction of the True values within the subsetted training data.
    - Another that hides the same fraction of the True values but within the full data.
    """
    train_mask = torch.zeros(num_nodes_full, dtype=torch.bool)
    train_mask[train_indices_full] = True

    # Number of training samples
    num_train_samples = train_indices_full.size(0)

    # Determine the number of nodes to hide
    num_to_hide = int(hide_frac * num_train_samples)

    # Randomly select relative nodes to hide from the subset of training data
    hide_relative_indices = torch.randperm(num_train_samples)[:num_to_hide]

    # Create a new mask for the subsetted training data
    hidden_train_mask_subset = torch.ones(num_train_samples, dtype=torch.bool)

    # Set the mask value of the selected nodes to False
    hidden_train_mask_subset[hide_relative_indices] = False

    # Convert relative hide indices to full data indices
    hide_full_indices = train_indices_full[hide_relative_indices]

    # Set the mask value of the selected nodes to False in the full mask
    train_mask[hide_full_indices] = False

    return hidden_train_mask_subset, train_mask
