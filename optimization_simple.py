import torch
from torch_geometric.nn import Node2Vec
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import subgraph
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import numpy as np
import wandb

from mydata import MyDataset, recalculate_input_features
from utils import evaluate_batch, calc_accuracy

wandb.init(project="Genomics", entity="kenenbek")

# Store configurations/hyperparameters
wandb.config.lr = 0.001
wandb.config.weight_decay = 5e-3
wandb.config.epochs = 1200

full_dataset = MyDataset(root="full_data/")
full_data = full_dataset[0]
num_nodes = full_data.y.shape[0]
train_indices = torch.load("full_data/0/train_indices.pt")
test_indices = torch.load("full_data/0/test_indices.pt")

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_indices] = True
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask[test_indices] = True

assert torch.equal(train_mask, ~test_mask), "Error"

_, edge_num, big_features = recalculate_input_features(full_data, train_mask)
full_data.edge_num = edge_num
full_data.big_features = big_features

# Placeholder for Node2Vec parameters
embedding_size = 64
walk_length = 20
context_size = 10
walks_per_node = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_edge_index, train_edge_weight = subgraph(
        train_indices, full_data.edge_index, edge_attr=full_data.edge_attr, relabel_nodes=True
    )

model = Node2Vec(full_data.edge_index,
                 embedding_dim=embedding_size,
                 walk_length=walk_length,
                 context_size=context_size,
                 walks_per_node=walks_per_node,
                 num_negative_samples=1).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

nodes = torch.arange(full_data.y.size(0)).to(device)
pos_sample = model.pos_sample(nodes)
neg_sample = model.neg_sample(nodes)
# Training loop (simplified)

model.train()
for epoch in range(10):
    optimizer.zero_grad()
    output = model(None)
    print(output.shape)
    print(full_data.y.shape)
    loss = criterion(output[train_mask], full_data.y[train_mask])
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch}, Loss: {loss}')

# Evaluate the model and return a metric (e.g., accuracy)
# Note: Implement the evaluation logic based on your dataset
y_true, y_pred = evaluate_batch(model, full_data, test_mask)
calc_accuracy(y_true, y_pred)


