import torch
import networkx as nx
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.utils import to_networkx
from torch_geometric.transforms import RandomLinkSplit


DDI_graph = torch.load("/Users/ishaansingh/Downloads/GNN_DDI/full_data/ddi_graph.pt")

# Assign the same label to all nodes
label = 0  # Use 0 for all drugs
DDI_graph['drug'].y = torch.full((DDI_graph['drug'].num_nodes,), label, dtype=torch.long)

# Extract node features and graph structure
m_type1 = ("drug", 0, "drug")
m_type2 = ("drug", 1, "drug")
# Apply RandomLinkSplit to split edges
transform = RandomLinkSplit(
    num_val=0.0,           # 20% of edges for validation
    num_test=0.2,          # 20% of edges for testing
    is_undirected=False,   # Adjust based on your graph
    split_labels=True,     # Generates positive and negative edge labels
    edge_types = (("drug", "affects", "drug"))
)
train_data, val_data, test_data = transform(DDI_graph)

tt1_idx = torch.argwhere(train_data["drug", "affects", "drug"].edge_attr == 0)
# Shuffle indices
num_tt1 = len(tt1_idx)
perm = torch.randperm(num_tt1)

# Define split size
train_size = int(num_tt1 * 0.8)

# Split into 80% train and 20% test
train_tt1_idx = tt1_idx[perm[:train_size]]
test_tt1_idx = tt1_idx[perm[train_size:]]

print(train_tt1_idx)
print(test_tt1_idx)