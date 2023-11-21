from matplotlib import pyplot as plt
import torch
import numpy as np
import random
from sklearn import metrics
from tqdm import tqdm, trange
from torch_geometric.nn.models import LabelPropagation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from mydata import MyDataset
from utils import prep_for_reconstruct


if __name__ == '__main__':
    full_dataset = MyDataset(root="full_data/")
    full_data = full_dataset[0]
    num_nodes = full_data.y.shape[0]
    train_indices_full = torch.load("full_data/0/train_indices.pt")
    test_indices = torch.load("full_data/0/test_indices.pt").tolist()

    for test_index in trange(len(test_indices)):
        # Get the actual node index
        idx = test_indices[test_index]

        # Combine training indices with the current test index
        sub_indices = torch.cat([train_indices_full, torch.tensor([idx])])
        sub_indices, _ = torch.sort(sub_indices)

        # Extract sub-graph
        sub_data = full_data.subgraph(sub_indices)

        # Find the position of the test node in the subgraph
        test_node_position = torch.where(sub_indices == idx)[0].item()
        n = sub_data.y.shape[0]
        rec = prep_for_reconstruct(sub_data.edge_index, n=n, m=n, dim=128)

        torch.save((rec, test_node_position), f"recs/rec_{test_index}.pt")
