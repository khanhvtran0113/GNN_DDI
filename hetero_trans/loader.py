import torch
import networkx as nx
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.utils import to_networkx

# Load the graph
DDI_graph = torch.load('/Users/ishaansingh/Downloads/GNN_DDI/full_data/ddi_graph.pt')

# Print details of the graph
print("Graph Details:\n", DDI_graph)

# Drug node features
print("\nDrug Node Features:")
print(DDI_graph['drug'].x)  

# Edge index for (drug, affects, drug)
print("\nEdge Index for (drug, affects, drug):")
edge_index = DDI_graph['drug', 'affects', 'drug'].edge_index
print(edge_index)

# Edge attributes for (drug, affects, drug)
print("\nEdge Attributes for (drug, affects, drug):")
edge_attr = DDI_graph['drug', 'affects', 'drug'].edge_attr
print(edge_attr)

# Number of drug nodes
num_nodes = DDI_graph['drug'].x.size(0)
print("\nNumber of Drug Nodes:", num_nodes)

# Number of edges
num_edges = edge_index.size(1)
print("\nNumber of Edges in (drug, affects, drug):", num_edges)

# Function to check for bidirectional edges
def check_bidirectional_edges(edge_index):
    # Convert edge_index to a set of tuples
    edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    # Find reverse edges
    reverse_edges = set(zip(edge_index[1].tolist(), edge_index[0].tolist()))
    # Identify bidirectional edges
    bidirectional_edges = edges & reverse_edges
    
    print("\nNumber of Bidirectional Edges:", len(bidirectional_edges))
    if len(bidirectional_edges) > 0:
        print("Bidirectional Edges Found:")
        print(bidirectional_edges)
    else:
        print("No Bidirectional Edges Found.")

# Run the check on the edge index
check_bidirectional_edges(edge_index)

# Print metadata
print("\nGraph Metadata:")
print(vars(DDI_graph))

# Visualize the graph
label = 0  # Use 0 for all drugs
DDI_graph['drug'].y = torch.full((DDI_graph['drug'].num_nodes,), label, dtype=torch.long)

# Apply node split transform
transform = RandomNodeSplit(split='random', num_train_per_class=60, num_val=0, num_test=19, num_splits=1)
data = transform(DDI_graph)
print("\nTransformed Data:")
print(data)

# Print number of nodes and labels
print("Number of nodes:", data['drug'].num_nodes)
print("Node labels:", data['drug'].y.unique() if 'y' in data['drug'] else "No labels found")

# Check available keys in the edge store
print("\nEdge Store Keys for (drug, affects, drug):")
print(DDI_graph["drug", "affects", "drug"].keys)
