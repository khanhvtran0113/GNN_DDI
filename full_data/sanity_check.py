import torch
import json
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx


def load_graph(graph_path):
    """Load the saved PyTorch Geometric graph."""
    graph = torch.load(graph_path)
    print(f"Graph loaded from {graph_path}")
    return graph


def load_feature_encoders(encoder_path):
    """Load feature encoders from a JSON file."""
    with open(encoder_path, 'r') as f:
        encoders = json.load(f)
    print(f"Feature encoders loaded from {encoder_path}")
    return encoders


def decode_node_features(node_features, feature_encoders):
    """Decode node features into human-readable format."""
    decoded = {}
    offset = 0

    # Decode categorical features
    for key in ['superclass', 'class', 'subclass']:
        encoder = feature_encoders[key]
        encoded_value = node_features[offset].item()
        decoded[key] = list(encoder.keys())[encoded_value]
        offset += 1

    # Decode set-based features
    for key in ['pathways', 'targets', 'enzymes', 'carriers', 'transporters']:
        encoder = feature_encoders[key]
        feature_vector = node_features[offset:offset + len(encoder)].tolist()
        decoded[key] = [list(encoder.keys())[i] for i, val in enumerate(feature_vector) if val == 1]
        offset += len(encoder)

    return decoded


def sanity_check_graph(graph, feature_encoders):
    """Perform sanity checks on the graph."""
    print("\nSanity Check: Graph Properties")
    print("-" * 40)
    print(f"Number of nodes: {graph['drug'].x.size(0)}")
    print(f"Node feature shape: {graph['drug'].x.shape}")
    print(f"Edge index shape: {graph['drug', 'affects', 'drug'].edge_index.shape}")
    print(f"Edge attribute shape: {graph['drug', 'affects', 'drug'].edge_attr.shape}")

    # Check feature encoders
    print("\nSanity Check: Feature Encoders")
    print("-" * 40)
    for feature, encoder in feature_encoders.items():
        print(f"Feature: {feature}, Encoded Values: {len(encoder)} items")

    # Verify edge types
    edge_types = graph['drug', 'affects', 'drug'].edge_attr.unique().tolist()
    print("\nSanity Check: Edge Types")
    print("-" * 40)
    print(f"Unique edge types: {edge_types} (0: increases, 1: decreases)")

    # Decode example node features
    print("\nExample Decoded Node Features")
    print("-" * 40)
    for i in range(min(5, graph['drug'].x.size(0))):  # Display up to 5 nodes
        node_features = graph['drug'].x[i]
        decoded_features = decode_node_features(node_features, feature_encoders)
        print(f"Node {i}: {decoded_features}")


def visualize_graph(graph, max_nodes=100):
    """Visualize a heterogeneous graph using NetworkX."""
    print("\nVisualizing Graph...")

    # Initialize an empty NetworkX graph
    nx_graph = nx.DiGraph()  # Use DiGraph for directed graphs

    # Iterate over edge types in the heterogeneous graph
    for edge_type in graph.edge_types:
        edge_index = graph[edge_type].edge_index

        # Add edges to the NetworkX graph
        source_nodes, target_nodes = edge_index
        nx_graph.add_edges_from(zip(source_nodes.tolist(), target_nodes.tolist()), edge_type=edge_type)

    # Convert to undirected if needed
    nx_graph = nx_graph.to_undirected()

    # Limit the number of nodes visualized
    if nx_graph.number_of_nodes() > max_nodes:
        subgraph = nx.subgraph(nx_graph, list(nx_graph.nodes)[:max_nodes])
    else:
        subgraph = nx_graph

    # Visualize the graph
    pos = nx.spring_layout(subgraph, seed=42)  # Use a spring layout for visualization
    plt.figure(figsize=(12, 8))
    nx.draw(
        subgraph, pos, with_labels=False, node_size=50, alpha=0.8,
        edge_color="gray"
    )
    plt.title("Graph Visualization")
    plt.show()



def main():
    # Paths to the saved files
    graph_path = "ddi_graph.pt"
    encoder_path = "feature_encoders.json"

    # Load graph and encoders
    graph = load_graph(graph_path)
    feature_encoders = load_feature_encoders(encoder_path)

    # Perform sanity checks
    sanity_check_graph(graph, feature_encoders)

    # Visualize the graph
    visualize_graph(graph)


if __name__ == "__main__":
    main()
