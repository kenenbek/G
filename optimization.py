import torch
from torch_geometric.nn import Node2Vec
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import numpy as np
import wandb
import pickle

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

# Placeholder for Node2Vec parameters
embedding_size = 64
walk_length = 20
context_size = 10
walks_per_node = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

space = [
    Integer(8, 128, name='embedding_size'),
    Integer(64, 256, name='walk_length'),
    Integer(8, 64, name='context_size'),
    Integer(8, 128, name='walks_per_node')
]

full_data = full_data.to(device)
train_mask = train_mask.to(device)
test_mask = test_mask.to(device)


@use_named_args(space)
def objective(**params):
    model = Node2Vec(full_data.edge_index,
                     embedding_dim=params['embedding_size'],
                     walk_length=params['walk_length'],
                     context_size=params['context_size'],
                     walks_per_node=params['walks_per_node'],
                     num_negative_samples=1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        output = model(None)
        loss = criterion(output[train_mask], full_data.y[train_mask])
        loss.backward()
        optimizer.step()
        # print(f'Epoch {epoch}, Loss: {loss}')

    y_true, y_pred = evaluate_batch(model, full_data, test_mask)
    y_true = y_true.to('cpu').detach().numpy()
    y_pred = y_pred.to('cpu').detach().numpy()
    stats = calc_accuracy(y_true, y_pred)
    print(stats["x"])
    return stats["x"]  # Negative accuracy because gp_minimize seeks to minimize the objective


result = gp_minimize(objective, space, n_calls=10, random_state=0)

print("Best parameters: {}".format(result.x))
with open('optimization.pickle', 'wb') as handle:
    pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
