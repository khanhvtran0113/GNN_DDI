# Define functions to load processed data, possibly as PyTorch Geometric Data objects. 
# Also include code to split data into training, validation, and test sets.

import torch
from torch_geometric.data import Data

def load_processed_data(data_path):
    """Load the processed data object."""
    data = torch.load(data_path)
    return data

def split_data(data, train_ratio=0.8):
    """
    Splits the edges in the data into training and test sets.

    Args:
    - data (Data): The processed Data object.
    - train_ratio (float): Proportion of edges to use for training.

    Returns:
    - train_data (Data): Data object for training with a subset of edges.
    - test_data (Data): Data object for testing with remaining edges.
    """
    # Get total number of edges
    num_edges = data.edge_index.size(1)

    # Calculate the number of edges for training
    train_size = int(num_edges * train_ratio)

    # Shuffle edges randomly
    perm = torch.randperm(num_edges)
    train_edge_index = data.edge_index[:, perm[:train_size]]
    test_edge_index = data.edge_index[:, perm[train_size:]]

    # Create train and test Data objects
    train_data = Data(x=data.x, edge_index=train_edge_index, y=data.y if 'y' in data else None)
    test_data = Data(x=data.x, edge_index=test_edge_index, y=data.y if 'y' in data else None)

    return train_data, test_data

if __name__ == "__main__":
    data_path = "processed_data.pt"

    # Load the processed data
    data = load_processed_data(data_path)

    # Split into train and test sets
    train_data, test_data = split_data(data)

    # Save train and test sets
    torch.save(train_data, "train_data.pt")
    torch.save(test_data, "test_data.pt")

    print("Train and test sets saved.")

