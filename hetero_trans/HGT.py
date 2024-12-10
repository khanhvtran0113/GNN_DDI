import os
import torch
import subprocess
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import HGTConv
from torch_geometric.loader import DataLoader

if 'IS_GRADESCOPE_ENV' not in os.environ:
    torch_version = str(torch.__version__)
    scatter_src = f"https://pytorch-geometric.com/whl/torch-{torch_version}.html"
    subprocess.check_call(["pip", "install", "torch-scatter", "-f", scatter_src])
    subprocess.check_call(["pip", "install", "torch-sparse", "-f", scatter_src])
    subprocess.check_call(["pip", "install", "torch-geometric"])

# Heterogeneous Graph Transformer (HGT) Model
class HGT(torch.nn.Module):
    def __init__(self, metadata, hidden_dim, output_dim, num_heads, num_layers, dropout):
        super(HGT, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.dropout = dropout
        
        for i in range(num_layers):
            self.layers.append(
                HGTConv(
                    in_channels=-1 if i == 0 else hidden_dim,
                    out_channels=hidden_dim if i < num_layers - 1 else output_dim,
                    metadata=metadata,
                    heads=num_heads,
                    dropout=dropout
                )
            )

    def forward(self, x_dict, edge_index_dict):
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)
            x_dict = {key: F.dropout(F.relu(x), p=self.dropout, training=self.training) for key, x in x_dict.items()}
        return x_dict

    def loss(self, pred, label, mask):
        return F.cross_entropy(pred[mask], label[mask])

# Training Function
def train_hetero(dataset, args):
    print("Training Heterogeneous Graph Transformer...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = dataset.to(device)
    
    # Create the HGT model
    metadata = dataset.metadata()
    model = HGT(metadata, hidden_dim=args.hidden_dim, output_dim=dataset['drug'].y.max().item() + 1,
                num_heads=args.heads, num_layers=args.num_layers, dropout=args.dropout).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Train
    model.train()
    losses = []
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        
        # Forward pass
        out_dict = model(dataset.x_dict, dataset.edge_index_dict)
        pred = out_dict['drug']
        loss = model.loss(pred, dataset['drug'].y, dataset['drug'].train_mask)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 10 == 0:
            acc = test_hetero(dataset, model, device)
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Test Accuracy: {acc:.4f}")
    
    return losses, model

# Testing Function
def test_hetero(dataset, model, device):
    model.eval()
    dataset = dataset.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        out_dict = model(dataset.x_dict, dataset.edge_index_dict)
        pred = out_dict['drug'].argmax(dim=1)
        mask = dataset['drug'].test_mask
        correct = (pred[mask] == dataset['drug'].y[mask]).sum().item()
        total = mask.sum().item()
    return correct / total

# Main Script
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

if 'IS_GRADESCOPE_ENV' not in os.environ:
    for args in [
        {'model_type': 'HGT', 'dataset': 'DDI', 'num_layers': 2, 'heads': 2, 'batch_size': 32, 
         'hidden_dim': 32, 'dropout': 0.5, 'epochs': 100, 'opt': 'adam', 'opt_scheduler': 'none', 
         'opt_restart': 0, 'weight_decay': 5e-3, 'lr': 0.01},
    ]:
        args = objectview(args)
        
        if args.dataset == 'DDI':
            dataset = torch.load("/Users/ishaansingh/Downloads/GNN_DDI/full_data/ddi_graph.pt")
        else:
            raise NotImplementedError("Unknown dataset")
        
        # Train and test
        losses, model = train_hetero(dataset, args)
        acc = test_hetero(dataset, model, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Final Test Accuracy: {acc:.4f}")

        # Plot losses
        import matplotlib.pyplot as plt
        plt.plot(losses, label="Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
