import wandb

import torch
import torch.nn.functional as F
from torch_geometric.utils import subgraph, to_dense_adj
from torch_geometric.nn import GAE
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, r2_score
from torch.optim.lr_scheduler import StepLR

from mydata import ClassBalancedNodeSplit, MyDataset, create_hidden_train_mask
from my_vgae import EncoderGAE, WeightedInnerProductDecoder, SimplePredictor
from mymodels import AttnGCN
from utils import evaluate_one_by_one, evaluate_batch, evaluate_one_by_one_load_from_file, calc_accuracy, \
    set_global_seed, gae_evaluate_one_by_one
from utils import inductive_train, to_one_hot


def change_input(x_input, train_edge_index, train_edge_attr_multi):
    x_input = x_input.clone()
    train_edge_attr_multi = train_edge_attr_multi.clone()

    num_nodes = x_input.size(0)  # Assume data.y contains your node labels
    unknown_label = torch.tensor([0, 0, 0, 0, 0, 1]).type(torch.float).to(device)

    # Randomly select 10% of your node indices
    indices = torch.randperm(num_nodes)[: int(num_nodes) // 10].to(device)

    # Update the labels of these selected nodes to the unknown label
    x_input[indices] = unknown_label

    # Assuming edge_index is of shape [2, E] and edge_attr is of shape [E, num_labels]
    # where E is the number of edges, and that we have already computed 'indices' of nodes to change

    # Create a mask for edges where the source node label has been changed to unknown
    src_nodes = train_edge_index[0]  # Get the source nodes of all edges
    mask = src_nodes.unsqueeze(1) == indices.unsqueeze(0)  # Compare against changed indices
    mask = mask.any(dim=1)  # Reduce to get a mask for edges

    # Update the edge attributes
    new_edge_attr = torch.zeros_like(train_edge_attr_multi).to(device)
    new_edge_attr[mask, -1] = torch.max(train_edge_attr_multi[mask], dim=1)[0]  # Move src label info to the last position
    train_edge_attr_multi[mask] = new_edge_attr[mask]

    node_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
    node_mask[indices] = True
    return x_input, train_edge_attr_multi, node_mask


def reconstruct_edges(z, edge_index):
    # Assuming data.x is a tensor of node features [num_nodes, num_features]
    # and edge_index is a tensor of shape [2, num_edges] where the first row contains
    # the indices of the source nodes and the second row contains the indices of the target nodes.

    # Gather the source node features using edge_index
    x_i = z[edge_index[0]]

    # Gather the target node features using edge_index
    x_j = z[edge_index[1]]

    # x_i and x_j are gathered node features from the previous example
    # Perform scalar multiplication (dot product) along the feature dimension
    scalar_products = (x_i * x_j).sum(dim=1).unsqueeze(1)

    return scalar_products


if __name__ == "__main__":
    set_global_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wandb.init(project="Genomics", entity="kenenbek")

    # Store configurations/hyperparameters
    wandb.config.lr = 0.001
    wandb.config.weight_decay = 5e-4
    wandb.config.epochs = 5000

    full_dataset = MyDataset(root="full_data/")
    full_data = full_dataset[0]
    num_nodes = full_data.y.shape[0]
    train_indices_full = torch.load("full_data/0/train_indices.pt")
    test_indices = torch.load("full_data/0/test_indices.pt")

    train_mask_f = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask_f[train_indices_full] = True
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[test_indices] = True

    assert torch.equal(train_mask_f, ~test_mask), "Error"

    train_mask_sub, train_mask_h = create_hidden_train_mask(train_indices_full, num_nodes, hide_frac=0.0)
    full_data.recalculate_input_features(train_mask_h)

    gae_model = GAE(encoder=EncoderGAE(),
                    decoder=WeightedInnerProductDecoder()
                    ).to(device)
    predictor = AttnGCN().to(device)

    criterion = torch.nn.CrossEntropyLoss()

    combined_parameters = list(gae_model.parameters()) + list(predictor.parameters())

    gae_optimizer = torch.optim.Adam(combined_parameters, lr=wandb.config.lr,
                                     weight_decay=wandb.config.weight_decay)
    gae_scheduler = StepLR(gae_optimizer, step_size=500, gamma=0.1)

    t = trange(wandb.config.epochs, leave=True)
    losses = []

    # Extract the subgraph associated with the training nodes
    train_nodes = torch.nonzero(train_mask_f).squeeze()

    train_edge_index, train_edge_attr_multi = subgraph(
        train_nodes, full_data.edge_index, edge_attr=full_data.edge_attr_multi, relabel_nodes=True
    )

    _, train_edge_weight = subgraph(
        train_nodes, full_data.edge_index, edge_attr=full_data.edge_attr, relabel_nodes=True
    )

    weighted_matrix = to_dense_adj(train_edge_index, edge_attr=train_edge_weight).squeeze(0).squeeze(-1)

    gae_model = gae_model.to(device)
    full_data = full_data.to(device)
    train_mask_f = train_mask_f.to(device)
    train_edge_index = train_edge_index.to(device)
    train_edge_attr_multi = train_edge_attr_multi.to(device)
    train_edge_weight = train_edge_weight.to(device)
    ones_tensor = torch.ones_like(full_data.x_one_hot[train_mask_f]).to(device)

    pred_edge_weights = None
    for epoch in t:
        gae_model.train()
        gae_optimizer.zero_grad()

        x, attr, node_mask = change_input(full_data.x_one_hot[train_mask_f], train_edge_index, train_edge_attr_multi)

        z = gae_model.encode(ones_tensor, train_edge_index, train_edge_weight)
        adj_reconstructed = gae_model.decode(z)

        pred_edge_weights = reconstruct_edges(z, train_edge_index)

        recon_loss = F.mse_loss(pred_edge_weights, train_edge_weight)

        y_pred = predictor(z, train_edge_index, train_edge_weight)
        class_loss = criterion(y_pred[train_mask_sub], full_data.y[train_mask_h])

        loss = class_loss + recon_loss

        loss.backward()
        gae_optimizer.step()
        gae_scheduler.step()

        r2 = r2_score(train_edge_weight.detach().cpu().numpy(), pred_edge_weights.detach().cpu().numpy())

        wandb.log({"recon_loss": recon_loss.item(),
                   "class_loss": class_loss.item(),
                   "r2_squared": r2.item()})

    torch.save(gae_model.state_dict(), "gae.pt")

    torch.save(pred_edge_weights, "gae_edge_attr.pt")
    torch.save(train_edge_weight, "train_edge_attr.pt")

    train_edge_weight = train_edge_weight.squeeze(1).cpu().detach().numpy()[:200]
    pred_edge_weights = pred_edge_weights.squeeze(1).cpu().detach().numpy()[:200]

    logged_table = wandb.Table(columns=["True", "Pred"])

    for expected, output in zip(train_edge_weight, pred_edge_weights):
        logged_table.add_data(expected, output)

    wandb.log({"comparison_edges": logged_table})

    # TEST one by one
    y_true, y_pred = gae_evaluate_one_by_one(gae_model, predictor,
                                             full_data, train_mask_f, test_mask)
    metrics = calc_accuracy(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 10))
    sub_etnos = ["1", "2", "3", "4", "5"]

    cm_display = ConfusionMatrixDisplay.from_predictions(y_true,
                                                         y_pred,
                                                         display_labels=sub_etnos,
                                                         ax=ax)
    fig.savefig("confusion_matrix.png")  # Save the figure to a file
    wandb.log({"confusion_matrix": wandb.Image("confusion_matrix.png")})  # Log the saved figure to wandb
    wandb.log(metrics)

    torch.save(model.state_dict(), "attn_full.pt")
    torch.save(y_true, "y_true.pt")
    torch.save(y_pred, "y_pred.pt")

    # Create a new artifact
    artifact = wandb.Artifact('model-artifact', type='model')
    # Add a file to the artifact (the model file)
    artifact.add_file('attn_full.pt')
    # Log the artifact
    wandb.log_artifact(artifact)

    wandb.finish()
