import torch
import copy
import deepsnap
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import pandas as pd
from sklearn.metrics import f1_score
from deepsnap.hetero_graph import HeteroGraph
from torch_sparse import SparseTensor, matmul
from torch_geometric.transforms import RandomLinkSplit

class HeteroGNNConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels_src, in_channels_dst, out_channels):
        super(HeteroGNNConv, self).__init__(aggr="mean")

        self.in_channels_src = in_channels_src
        self.in_channels_dst = in_channels_dst
        self.out_channels = out_channels

        # To simplify implementation, please initialize both self.lin_dst
        # and self.lin_src out_features to out_channels
        self.lin_dst = None
        self.lin_src = None

        self.lin_update = None

        ## 1. Initialize the 3 linear layers.
        ## 2. Think through the connection between the mathematical
        ##    definition of the update rule and torch linear layers!

        self.lin_dst = nn.Linear(self.in_channels_src, self.out_channels, bias=False)
        self.lin_src = nn.Linear(self.in_channels_dst, self.out_channels, bias=False)
        self.lin_update = nn.Linear(self.out_channels * 2, self.out_channels, bias=False)

    def forward(
            self,
            node_feature_src,
            node_feature_dst,
            edge_index,
            size=None
    ):
        return self.propagate(edge_index, size=size, node_feature_src=node_feature_src, node_feature_dst=node_feature_dst)

    def message_and_aggregate(self, edge_index, node_feature_src):
        out = matmul(edge_index, node_feature_src, reduce=self.aggr)
        return out

    def update(self, aggr_out, node_feature_dst):
       W_d = self.lin_dst(node_feature_dst)
       W_s = self.lin_src(aggr_out)
       combined = torch.cat([W_d, W_s], dim=-1)
       out = self.lin_update(combined)
       return out

class HeteroGNNWrapperConv(deepsnap.hetero_gnn.HeteroConv):
    def __init__(self, convs, args, aggr="mean"):
        super(HeteroGNNWrapperConv, self).__init__(convs, None)
        self.aggr = aggr

        # Map the index and message type
        self.mapping = {}

        # A numpy array that stores the final attention probability
        self.alpha = None

        self.attn_proj = None

        if self.aggr == "attn":
            self.attn_proj = nn.Sequential(nn.Linear(args['hidden_size'], args['attn_size'], bias=True),
                nn.Tanh(),  # Non-linear activation
                nn.Linear(args['attn_size'], 1, bias=False))

            ##########################################

    def reset_parameters(self):
        super(HeteroGNNWrapperConv, self).reset_parameters()
        if self.aggr == "attn":
            for layer in self.attn_proj.children():
                layer.reset_parameters()

    def forward(self, node_features, edge_indices):
        message_type_emb = {}

        print(node_features.keys())
        for message_key, message_type in edge_indices.items():
            src_type, edge_type, dst_type = message_key
            edge_index = edge_indices[message_key]
            node_feature_src = node_features[src_type]
            node_feature_dst = node_features[dst_type]
            message_type_emb[message_key] = (
                self.convs[message_key](
                    node_feature_src,
                    node_feature_dst,
                    edge_index,
                    size=None
                )
            )
        node_emb = {dst: [] for _, _, dst in message_type_emb.keys()}
        mapping = {}
        for (src, edge_type, dst), item in message_type_emb.items():
            mapping[len(node_emb[dst])] = (src, edge_type, dst)
            node_emb[dst].append(item)
        self.mapping = mapping
        for node_type, embs in node_emb.items():
            if len(embs) == 1:
                node_emb[node_type] = embs[0]
            else:
                node_emb[node_type] = self.aggregate(embs)
        return node_emb

    def aggregate(self, xs):
        if self.aggr == "mean":
            x = torch.stack(xs, dim=0)
            return x.mean(dim=0)

        elif self.aggr == "attn":
            N = xs[0].shape[0] # Number of nodes for that node type
            M = len(xs) # Number of message types for that node type

            x = torch.cat(xs, dim=0).view(M, N, -1) # M * N * D
            z = self.attn_proj(x).view(M, N) # M * N * 1
            z = z.mean(1) # M * 1
            alpha = torch.softmax(z, dim=0) # M * 1

            # Store the attention result to self.alpha as np array
            self.alpha = alpha.view(-1).data.cpu().numpy()

            alpha = alpha.view(M, 1, 1)
            x = x * alpha
            return x.sum(dim=0)


def generate_convs(hetero_graph, conv, hidden_size, first_layer=False):
    convs = {}

    m_type1 = ("drug", 0, "drug")
    m_type2 = ("drug", 1, "drug")
    message_types = [m_type1, m_type2]

    for message_type in message_types:
        if first_layer:
          in_channels_src = hetero_graph.num_node_features("drug")
          in_channels_dst = hetero_graph.num_node_features("drug")
        else:
          in_channels_src = hidden_size
          in_channels_dst = hidden_size

        convs[message_type] = conv(in_channels_src, in_channels_dst, hidden_size)

    return convs

class HeteroGNN(torch.nn.Module):
    def __init__(self, hetero_graph, args, aggr="mean"):
        super(HeteroGNN, self).__init__()

        self.aggr = aggr
        self.hidden_size = args['hidden_size']

        self.convs1 = None
        self.convs2 = None

        self.bns1 = nn.ModuleDict()
        self.bns2 = nn.ModuleDict()
        self.relus1 = nn.ModuleDict()
        self.relus2 = nn.ModuleDict()
        self.post_mps = nn.ModuleDict()

        self.convs1 = HeteroGNNWrapperConv(
            generate_convs(hetero_graph, HeteroGNNConv, self.hidden_size, first_layer=True),
            args,
            aggr=aggr
        )

        self.convs2 = HeteroGNNWrapperConv(
            generate_convs(hetero_graph, HeteroGNNConv, self.hidden_size, first_layer=False),
            args,
            aggr=aggr
        )

        for node_type in hetero_graph.node_types:
          self.bns1[node_type] = torch.nn.BatchNorm1d(self.hidden_size, eps=1)
          self.bns2[node_type] = torch.nn.BatchNorm1d(self.hidden_size, eps=1)
          self.relus1[node_type] = nn.LeakyReLU()
          self.relus2[node_type] = nn.LeakyReLU()
          self.post_mps[node_type] = nn.Linear(self.hidden_size, hetero_graph.num_node_labels(node_type), bias=False)


    def forward(self, node_feature, edge_index, edge_attr=None):
        # First convolutional layer
        x = self.convs1(node_feature, edge_index)
        x = deepsnap.hetero_gnn.forward_op(x, self.bns1)
        x = deepsnap.hetero_gnn.forward_op(x, self.relus1)

        # Second convolutional layer
        x = self.convs2(x, edge_index)
        x = deepsnap.hetero_gnn.forward_op(x, self.bns2)
        x = deepsnap.hetero_gnn.forward_op(x, self.relus2)

        # Prepare edge embeddings for link prediction
        edge_embeddings = []
        for (src, _, dst), edge_idx in edge_index.items():
            src_features = x[src][edge_idx[0]]  # Source node features
            dst_features = x[dst][edge_idx[1]]  # Destination node features
            edge_feature = torch.cat([src_features, dst_features], dim=1)

            # If edge_attr is provided, concatenate it as well
            if edge_attr is not None:
                edge_attr_features = edge_attr[edge_idx]
                edge_feature = torch.cat([edge_feature, edge_attr_features], dim=1)

            edge_embeddings.append(edge_feature)

        # Concatenate all edge embeddings
        return torch.cat(edge_embeddings, dim=0)

    def loss(self, preds, y, edge_label):
        loss_func = F.cross_entropy if edge_label.ndim == 1 else F.binary_cross_entropy
        return loss_func(preds, edge_label)

def train(model, optimizer, hetero_graph, edge_split):
    model.train()
    optimizer.zero_grad()

    # Get predictions for edges
    edge_index = edge_split["edge_index"]
    edge_label = edge_split["edge_label"]
    preds = model(hetero_graph.node_feature, edge_index)

    # Compute loss
    loss = model.loss(preds, edge_label)

    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, graph, edge_split, best_model=None, best_val=0, save_preds=False, agg_type=None):
    model.eval()
    accs = []

    for phase in ["train", "test"]:
        edge_index = edge_split[phase]["edge_index"]
        edge_label = edge_split[phase]["edge_label"]

        preds = model(graph.node_feature, edge_index)
        pred_classes = preds.max(1)[1]  # Predicted edge types (or binary predictions for existence)

        # Calculate F1 score
        micro = f1_score(edge_label.cpu(), pred_classes.cpu(), average='micro')
        macro = f1_score(edge_label.cpu(), pred_classes.cpu(), average='macro')
        accs.append((micro, macro))

    # Compare validation micro F1 score with the best so far
    if accs[1][0] > best_val:  # If current validation micro F1 is better
        best_val = accs[1][0]  # Update best validation score
        best_model = copy.deepcopy(model)  # Save the current model

    return accs, best_model, best_val


def main():
    args = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'hidden_size': 64,
        'epochs': 100,
        'weight_decay': 1e-5,
        'lr': 0.003,
        'attn_size': 32,
    }

    file_path = "../full_data/ddi_graph.pt"
    data = torch.load(file_path)

    # Extract node features and graph structure
    m_type1 = ("drug", 0, "drug")
    m_type2 = ("drug", 1, "drug")
    # Apply RandomLinkSplit to split edges
    transform = RandomLinkSplit(
        num_val=0.0,  # 20% of edges for validation
        num_test=0.2,  # 20% of edges for testing
        is_undirected=False,  # Adjust based on your graph
        split_labels=True,  # Generates positive and negative edge labels
        edge_types=(("drug", "affects", "drug"))
    )
    edge_index = {}
    tt1_idx = torch.argwhere(data["drug", "affects", "drug"].edge_attr == 0)
    tt2_idx = torch.argwhere(data["drug", "affects", "drug"].edge_attr == 1)
    edge_index[m_type1] = data["drug", "affects", "drug"].edge_index[:, tt1_idx]
    edge_index[m_type2] = data["drug", "affects", "drug"].edge_index[:, tt2_idx]

    train_data, val_data, test_data = transform(data)

    train_edge_index = {}
    tt1_idx = torch.argwhere(train_data["drug", "affects", "drug"].edge_attr == 0).squeeze()
    tt2_idx = torch.argwhere(train_data["drug", "affects", "drug"].edge_attr == 1).squeeze()
    train_edge_index[m_type1] = train_data["drug", "affects", "drug"].edge_index[:, tt1_idx]
    train_edge_index[m_type2] = train_data["drug", "affects", "drug"].edge_index[:, tt2_idx]

    test_edge_index = {}
    tt1_idx = torch.argwhere(test_data["drug", "affects", "drug"].edge_attr == 0).squeeze()
    tt2_idx = torch.argwhere(test_data["drug", "affects", "drug"].edge_attr == 1).squeeze()
    test_edge_index[m_type1] = test_data["drug", "affects", "drug"].edge_index[:, tt1_idx]
    test_edge_index[m_type2] = test_data["drug", "affects", "drug"].edge_index[:, tt2_idx]

    node_feature = data["drug"].x
    num_nodes = node_feature.size(0)

    edge_attr = data["drug", "affects", "drug"].edge_attr
    # Construct a HeteroGraph
    hetero_graph = HeteroGraph(
        node_feature={"drug": data["drug"].x},
        edge_index={"drug": edge_index},
        edge_label={"drug": edge_attr},
        directed=True,
    )

    # train_edge_index = torch.cat([train_edge_index[m_type1], train_edge_index[m_type2]], dim=1)
    train_edge_label = torch.cat([
        torch.zeros(train_data["drug", "affects", "drug"].edge_index[:, tt1_idx].size(1), dtype=torch.float),
        # Type 0 edges
        torch.ones(train_data["drug", "affects", "drug"].edge_index[:, tt2_idx].size(1), dtype=torch.float)
        # Type 1 edges
    ])
    test_edge_index = torch.cat([test_edge_index[m_type1], test_edge_index[m_type2]], dim=1)
    test_edge_label = torch.cat([
        torch.zeros(test_data["drug", "affects", "drug"].edge_index[:, tt1_idx].size(1), dtype=torch.float),
        # Type 0 edges
        torch.ones(test_data["drug", "affects", "drug"].edge_index[:, tt2_idx].size(1), dtype=torch.float)
        # Type 1 edges
    ])

    edge_split = {
        "train": {"edge_index": train_edge_index, "edge_label": train_edge_label},
        "test": {"edge_index": test_edge_index, "edge_label": test_edge_label},
    }

    model = HeteroGNN(hetero_graph, args, aggr="mean").to(args['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    best_model = None
    best_val = 0

    for epoch in range(args['epochs']):
        loss = train(model, optimizer, hetero_graph, edge_split["train"])
        accs, best_model, best_val = test(model, hetero_graph, edge_split, best_model, best_val)
        print(
            f"Epoch {epoch + 1}: loss {round(loss, 5)}, "
            f"train micro {round(accs[0][0] * 100, 2)}%, train macro {round(accs[0][1] * 100, 2)}%, "
            f"test micro {round(accs[1][0] * 100, 2)}%, test macro {round(accs[1][1] * 100, 2)}%"
        )
    best_accs, _, _ = test(best_model, hetero_graph, edge_split, save_preds=True, agg_type="Mean")
    print(
        f"Best model: "
        f"train micro {round(best_accs[0][0] * 100, 2)}%, train macro {round(best_accs[0][1] * 100, 2)}%, "
        f"test micro {round(best_accs[1][0] * 100, 2)}%, test macro {round(best_accs[1][1] * 100, 2)}%"
    )



if __name__ == '__main__':
    main()