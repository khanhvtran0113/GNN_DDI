import torch
import networkx as nx
from torch_geometric.transforms import RandomNodeSplit

DDI_graph = torch.load('/Users/ishaansingh/Downloads/GNN_DDI/full_data/ddi_graph.pt')
print(DDI_graph)

print("\nDrug Node Features:")
print(DDI_graph['drug'].x)  

print("\nEdge Index for (drug, affects, drug):")
print(DDI_graph['drug', 'affects', 'drug'].edge_index)

print("\nEdge Attributes for (drug, affects, drug):")
print(DDI_graph['drug', 'affects', 'drug'].edge_attr)

print("\nNumber of Drug Nodes:")
print(DDI_graph['drug'].x.size(0))

print("\nNumber of Edges in (drug, affects, drug):")
print(DDI_graph['drug', 'affects', 'drug'].edge_index.size(1))

print(vars(DDI_graph))

import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
label = 0 # use 0 for all drugs
DDI_graph['drug'].y = torch.full((DDI_graph['drug'].num_nodes,), label, dtype=torch.long)
transform = RandomNodeSplit(split='random', num_train_per_class=60, num_val=0, num_test=19, num_splits=1)
data = transform(DDI_graph)
print(data)
print("Number of nodes:", data['drug'].num_nodes)
print("Node labels:", data['drug'].y.unique() if 'y' in data['drug'] else "No labels found")
