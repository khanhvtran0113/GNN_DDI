import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import MessagePassing, RGCNConv
from torch_geometric.utils import add_self_loops, degree

class ComplExSeHGNN(torch.nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, hidden_dim, dropout):
        super(ComplExSeHGNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # ComplEx embeddings for entities and relations
        self.ent_real = nn.Embedding(num_entities, embedding_dim)
        self.ent_imag = nn.Embedding(num_entities, embedding_dim)
        self.rel_real = nn.Embedding(num_relations, embedding_dim)
        self.rel_imag = nn.Embedding(num_relations, embedding_dim)

        nn.init.xavier_uniform_(self.ent_real.weight)
        nn.init.xavier_uniform_(self.ent_imag.weight)
        nn.init.xavier_uniform_(self.rel_real.weight)
        nn.init.xavier_uniform_(self.rel_imag.weight)

        # Linear projection from embedding space to hidden space
        self.fc = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, head, relation, tail, edge_index, edge_type):
        """
        Forward pass through ComplEx and graph structure.
        """
        # ComplEx scoring
        h_real = self.ent_real(head)
        h_imag = self.ent_imag(head)
        r_real = self.rel_real(relation)
        r_imag = self.rel_imag(relation)
        t_real = self.ent_real(tail)
        t_imag = self.ent_imag(tail)

        # ComplEx score computation
        score_real = (h_real * r_real * t_real) + (h_imag * r_real * t_imag) + \
                     (h_real * r_imag * t_imag) - (h_imag * r_imag * t_real)
        score = score_real.sum(dim=-1)
        # Node features via projection
        node_features = self.fc(self.ent_real.weight + self.ent_imag.weight)
        node_features = self.dropout(torch.relu(node_features))

        return score, node_features

class Matcher(torch.nn.Module):
    def __init__(self, n_hid):
        super(Matcher, self).__init__()
        self.left_linear = torch.nn.Linear(n_hid, n_hid)
        self.right_linear = torch.nn.Linear(n_hid, n_hid)
        self.sqrt_hd = torch.sqrt(torch.tensor(n_hid, dtype=torch.float))
    
    def forward(self, x, y, pair=True):
        left = self.left_linear(x)
        right = self.right_linear(y)
        if pair:
            return (left * right).sum(dim=-1) / self.sqrt_hd
        return torch.matmul(left, right.T) / self.sqrt_hd
    
import torch
from torch_geometric.utils import negative_sampling
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.transforms import RandomLinkSplit

# Load graph
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

# Access train, validation, and test splits
#train_edge_index = train_data.edge_index

train_edge_index = {}
tt1_idx = torch.argwhere(train_data["drug", "affects", "drug"].edge_attr == 0)
tt2_idx = torch.argwhere(train_data["drug", "affects", "drug"].edge_attr == 1)
train_edge_index[m_type1] = train_data["drug", "affects", "drug"].edge_index[:,tt1_idx]
train_edge_index[m_type2] = train_data["drug", "affects", "drug"].edge_index[:,tt2_idx]

test_edge_index = {}
tt1_idx = torch.argwhere(test_data["drug", "affects", "drug"].edge_attr == 0)
tt2_idx = torch.argwhere(test_data["drug", "affects", "drug"].edge_attr == 1)
test_edge_index[m_type1] = test_data["drug", "affects", "drug"].edge_index[:,tt1_idx]
test_edge_index[m_type2] = test_data["drug", "affects", "drug"].edge_index[:,tt2_idx]


node_feature = DDI_graph["drug"].x
num_nodes = node_feature.size(0)

# Node type (all nodes are of the same type)
node_type = torch.zeros(num_nodes, dtype=torch.long)

from torch.nn import Module

num_nodes = DDI_graph['drug'].num_nodes
num_relations = 2  # Two edge types: 0 and 1
embedding_dim = 32
hidden_dim = 32
dropout = 0.1

model = ComplExSeHGNN(num_nodes, num_relations, embedding_dim, hidden_dim, dropout)
matcher = Matcher(hidden_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Combine edge indices and assign edge types
train_edge_index = torch.cat([train_edge_index[m_type1], train_edge_index[m_type2]], dim=1)
train_edge_type = torch.cat([
    torch.zeros(train_data["drug", "affects", "drug"].edge_index[:,tt1_idx].size(1), dtype=torch.float),  # Type 0 edges
    torch.ones(train_data["drug", "affects", "drug"].edge_index[:,tt2_idx].size(1), dtype=torch.float)   # Type 1 edges
])

test_edge_index = torch.cat([test_edge_index[m_type1], test_edge_index[m_type2]], dim=1)
test_edge_type = torch.cat([
    torch.zeros(test_data["drug", "affects", "drug"].edge_index[:,tt1_idx].size(1), dtype=torch.float),  # Type 0 edges
    torch.ones(test_data["drug", "affects", "drug"].edge_index[:,tt2_idx].size(1), dtype=torch.float)   # Type 1 edges
])

# Edge times (optional; set to zero if not available)
train_edge_time = torch.zeros(train_edge_index.size(1), dtype=torch.float)
test_edge_time = torch.zeros(test_edge_index.size(1), dtype=torch.float)
for epoch in range(100):
    model.train()
    optimizer.zero_grad()

    # Prepare edge indices and types
    head, tail = train_edge_index
    edge_type = train_edge_type.long()

    # Forward pass
    pos_scores, node_features = model(head, edge_type, tail, train_edge_index, train_edge_type)

    # Negative sampling
    neg_edge_index = negative_sampling(edge_index=train_edge_index, num_nodes=num_nodes, num_neg_samples=head.size(0))
    neg_head, neg_tail = neg_edge_index
    neg_scores, _ = model(neg_head, edge_type, neg_tail, train_edge_index, train_edge_type)


    labels = torch.cat([torch.ones_like(pos_scores[:, 0]), torch.zeros_like(neg_scores)])
    scores = torch.cat([pos_scores[:, 0], neg_scores])

    # Binary cross-entropy loss
    loss = F.binary_cross_entropy_with_logits(scores, labels)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model.eval()
with torch.no_grad():
    test_head, test_tail = test_edge_index
    test_edge_type = test_edge_type.long()
    # Positive edge scores
    pos_scores, _ = model(test_head, test_edge_type, test_tail, test_edge_index, test_edge_type)

    # Negative edge scores
    neg_edge_index = negative_sampling(edge_index=test_edge_index, num_nodes=num_nodes, num_neg_samples=test_head.size(0))
    neg_head, neg_tail = neg_edge_index
    neg_scores, _ = model(neg_head, test_edge_type, neg_tail, test_edge_index, test_edge_type)

    # Combine scores
    labels = torch.cat([torch.ones_like(pos_scores[:, 0]), torch.zeros_like(neg_scores)])
    scores = torch.cat([pos_scores[:, 0], neg_scores])
    # AUC
    auc = roc_auc_score(labels.cpu(), scores.cpu())
    predictions = (scores > 0.5).cpu().numpy()

    # Ground-truth labels
    labels_np = labels.cpu().numpy()

    # Calculate metrics
    accuracy = accuracy_score(labels_np, predictions)
    precision = precision_score(labels_np, predictions)
    recall = recall_score(labels_np, predictions)
    f1 = f1_score(labels_np, predictions)

    # Print results
    print(f"Test AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Test AUC: {auc:.4f}")

import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import json

with open('/Users/ishaansingh/Downloads/GNN_DDI/full_data/feature_encoders.json', 'r') as f:
    feature_encoders = json.load(f)

# Reverse the feature encoders for quick lookup
name_mapping = {v: k for k, v in feature_encoders['name'].items()}  # Specifically for the "name" mapping

# Convert PyG graph to NetworkX
nx_graph = to_networkx(DDI_graph, edge_attrs=['edge_attr'], to_undirected=False)

# Extract edge attributes (edge types)
edge_types = nx.get_edge_attributes(nx_graph, 'edge_attr')

# Define a color map for each edge type
unique_edge_types = set(edge_types.values())
color_map = {edge_type: plt.cm.tab10(i) for i, edge_type in enumerate(unique_edge_types)}

# Assign colors to edges based on their type
edge_colors = [color_map[edge_types[edge]] for edge in nx_graph.edges]

# Adjust layout and figure size
pos = nx.spring_layout(nx_graph, k=2.3)  # Use a spring layout for better separation
plt.figure(figsize=(15, 10))

# Draw nodes
nx.draw_networkx_nodes(nx_graph, pos, node_size=900, node_color='skyblue')

# Extract node labels (use the mapping from feature_encoders)
node_labels = {node: name_mapping.get(node, f"Node {node}") for node in nx_graph.nodes}

# Draw node labels
nx.draw_networkx_labels(nx_graph, pos, labels=node_labels, font_size=4, font_color='black')

# Draw directed edges with curved arrows for bidirectional edges
arc_rad = 0.8  # Radius of the arc for curved edges
for edge, color in zip(nx_graph.edges, edge_colors):
    if nx_graph.has_edge(edge[1], edge[0]):  # If bidirectional
        nx.draw_networkx_edges(
            nx_graph, pos,
            edgelist=[edge],
            edge_color=[color],
            connectionstyle=f'arc3,rad={arc_rad}',
            arrowstyle='-|>',
            arrowsize=30,
            width=1
        )
    else:  # Single direction
        nx.draw_networkx_edges(
            nx_graph, pos,
            edgelist=[edge],
            edge_color=[color],
            arrowstyle='-|>',
            arrowsize=30,
            width=1
        )

# Create a legend for edge types
edge_type_labels = {0: "Increases", 1: "Decreases"}  # Custom labels for edge types
legend_labels = [edge_type_labels[edge_type] for edge_type in sorted(unique_edge_types)]
handles = [plt.Line2D([0], [0], color=color_map[edge_type], lw=2) for edge_type in sorted(unique_edge_types)]
plt.legend(handles, legend_labels, title="Edge Types", loc="upper right")

plt.title("Drug-Drug Interaction Graph")
plt.show()