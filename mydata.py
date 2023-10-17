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


class MyDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ["CR_graph_rel.csv"]

    @property
    def processed_file_names(self):
        return ['data_0.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):

        ind = {
            'мордвины': 0,
            'белорусы': 1,
            'украинцы': 2,
            'южные-русские': 3,
            'северные-русские': 4
        }

        idx = 0
        for raw_path in self.raw_paths:
            edge_index = []
            edge_attr = []

            y_labels = {}
            x_data = defaultdict(lambda: (5 * [0]))

            dataset_csv = pd.read_csv(raw_path)
            for index, row in tqdm(dataset_csv.iterrows()):
                node1 = row["node_id1"]
                node2 = row["node_id2"]
                label1 = row["label_id1"]
                label2 = row["label_id2"]
                ibd_sum = row["ibd_sum"]

                id1 = int(node1[5:])
                id2 = int(node2[5:])

                x_data[id1][ind[label2]] += ibd_sum
                x_data[id2][ind[label1]] += ibd_sum

                edge_index.append([id1, id2])
                edge_index.append([id2, id1])
                edge_attr.append([ibd_sum])
                edge_attr.append([ibd_sum])

                y_labels[id1] = ind[label1]
                y_labels[id2] = ind[label2]

            y_labels = dict(sorted(y_labels.items()))
            y = torch.Tensor(list(y_labels.values())).type(torch.long)

            x_data = dict(sorted(x_data.items()))
            x = torch.Tensor(list(x_data.values())).type(torch.float)
            edge_attr = torch.Tensor(edge_attr).type(torch.float).contiguous()
            edge_index = torch.Tensor(edge_index).type(torch.long).t().contiguous()

            x_one_hot = F.one_hot(y, num_classes=int(y.max()) + 1).type(torch.float)

            data = MyData(x=x,
                          edge_index=edge_index,
                          edge_attr=edge_attr,
                          x_one_hot=x_one_hot,
                          y=y)
            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data


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
        self.hidden_x = None
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

    def recalculate_input_features(self):
        assert self.train_mask is not None, "Error"
        assert self.hidden_train_mask_full is not None, "Error"

        available_node_indices = torch.nonzero(self.hidden_train_mask_full).squeeze()
        known_trainig_set = set(available_node_indices.tolist())

        hidden_x_data = {}
        for i in range(self.x.shape[0]):
            hidden_x_data[i] = [0, 0, 0, 0, 0]

        for i, edge in tqdm(enumerate(self.edge_index.t())):
            start_node = edge[0].item()
            dest_node = edge[1].item()

            start_ethnicity = self.y[start_node].item()

            if start_node in known_trainig_set:
                hidden_x_data[dest_node][start_ethnicity] += self.edge_attr[i]

        hidden_x_data = dict(sorted(hidden_x_data.items()))
        hidden_x = torch.Tensor(list(hidden_x_data.values())).contiguous()

        self.hidden_x = hidden_x


def generate_train_test_indices(y, run=10, train=.7, val=.0, test=.3):
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

        torch.save(train_indices, f"full_data/{i}/train_indices.pt")
        torch.save(val_indices, f"full_data/{i}/val_indices.pt")
        torch.save(test_indices, f"full_data/{i}/test_indices.pt")

    return


def fix_last_test_index(data, train_mask, test_index):
    # Create a copy of the original data
    data_copy = data.clone()

    # Relabel the test_index node to a value larger than any other node index
    max_index = data.num_nodes
    data_copy.edge_index[data_copy.edge_index == test_index] = max_index

    # Concatenate sub_indices with the new test index value
    sub_indices = torch.cat([torch.where(train_mask)[0], torch.tensor([max_index])])

    # Get the subgraph
    sub_data = data_copy.subgraph(sub_indices)


def generate_subgraphs():
    full_dataset = MyDataset(root="full_data/")
    full_data = full_dataset[0]

    train_indices = torch.load("full_data/0/train_indices.pt")
    test_indices = torch.load("full_data/0/test_indices.pt")

    for test_index in trange(len(test_indices)):
        full_data_copy = full_data.clone()

        max_index = full_data_copy.num_nodes + 1000000
        full_data_copy.edge_index[full_data_copy.edge_index == test_index] = max_index

        sub_indices = torch.cat([train_indices, torch.tensor([max_index])])
        sub_data = full_data.subgraph(sub_indices)

        torch.save(sub_data, "full_data/0/test_sub_graph_0.pt")