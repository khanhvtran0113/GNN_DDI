import torch
import copy
import networkx as nx
import deepsnap
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import pandas as pd
from sklearn.metrics import f1_score
from deepsnap.hetero_graph import HeteroGraph
from torch_sparse import SparseTensor, matmul

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

        return self.propagate(edge_index=edge_index, size=size, node_feature_src=node_feature_src, node_feature_dst=node_feature_dst)

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
        super(HeteroConvWrapper, self).reset_parameters()
        if self.aggr == "attn":
            for layer in self.attn_proj.children():
                layer.reset_parameters()

    def forward(self, node_features, edge_indices):
        message_type_emb = {}
        for message_key, message_type in edge_indices.items():
            src_type, edge_type, dst_type = message_key
            node_feature_src = node_features[src_type]
            node_feature_dst = node_features[dst_type]
            edge_index = edge_indices[message_key]
            message_type_emb[message_key] = (
                self.convs[message_key](
                    node_feature_src,
                    node_feature_dst,
                    edge_index,
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

    # Get all message types
    message_types = hetero_graph.message_types

    for message_type in message_types:
        src_type, _, dst_type = message_type
        if first_layer:
          in_channels_src = hetero_graph.num_node_features(src_type)
          in_channels_dst = hetero_graph.num_node_features(dst_type)
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


    def forward(self, node_feature, edge_index):
        x = node_feature

        x = self.convs1(x, edge_index)
        x = deepsnap.hetero_gnn.forward_op(x, self.bns1)
        x = deepsnap.hetero_gnn.forward_op(x, self.relus1)
        x = self.convs2(x, edge_index)
        x = deepsnap.hetero_gnn.forward_op(x, self.bns2)
        x = deepsnap.hetero_gnn.forward_op(x, self.relus2)
        x = deepsnap.hetero_gnn.forward_op(x, self.post_mps)

        return x

    def loss(self, preds, y, indices):

        loss = 0
        loss_func = F.cross_entropy

        for node_type in preds.keys():
          supervised_indices = indices[node_type]
          loss += loss_func(preds[node_type][supervised_indices], y[node_type][supervised_indices])

        return loss

def train(model, optimizer, hetero_graph, train_idx):
    model.train()
    optimizer.zero_grad()
    preds = model(hetero_graph.node_feature, hetero_graph.edge_index)

    loss = None

    ############# Your code here #############
    ## Note:
    ## 1. Compute the loss here
    ## 2. `deepsnap.hetero_graph.HeteroGraph.node_label` is useful
    y = {}
    indices = {}
    for node_type in preds.keys():
      y[node_type] = hetero_graph.node_label[node_type]
      indices[node_type] = train_idx[node_type]

    loss = model.loss(preds, y, indices)


    ##########################################

    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, graph, indices, best_model=None, best_val=0, save_preds=False, agg_type=None):
    model.eval()
    accs = []
    for i, index in enumerate(indices):
        preds = model(graph.node_feature, graph.edge_index)
        num_node_types = 0
        micro = 0
        macro = 0
        for node_type in preds:
            idx = index[node_type]
            pred = preds[node_type][idx]
            pred = pred.max(1)[1]
            label_np = graph.node_label[node_type][idx].cpu().numpy()
            pred_np = pred.cpu().numpy()
            micro = f1_score(label_np, pred_np, average='micro')
            macro = f1_score(label_np, pred_np, average='macro')
            num_node_types += 1

        # Averaging f1 score might not make sense, but in our example we only
        # have one node type
        micro /= num_node_types
        macro /= num_node_types
        accs.append((micro, macro))

        # Only save the test set predictions and labels!
        if save_preds and i == 2:
          print ("Saving Heterogeneous Node Prediction Model Predictions with Agg:", agg_type)
          print()

          data = {}
          data['pred'] = pred_np
          data['label'] = label_np

          df = pd.DataFrame(data=data)
          # Save locally as csv
          df.to_csv('ACM-Node-' + agg_type + 'Agg.csv', sep=',', index=False)

    if accs[1][0] > best_val:
        best_val = accs[1][0]
        best_model = copy.deepcopy(model)
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

    # TODO: Load the data
    data = None

    # TODO: Load message types from G

    # TODO: Load dictionary of edge indices
    edge_index = {}

    # TODO: Load dictionary of node features
    node_feature = {}

    # TODO: Load dictionary of node labels
    node_label = {}

    # Load the train, validation and test indices
    train_idx = None
    val_idx = None
    test_idx = None

    # Construct a deepsnap tensor backend HeteroGraph
    hetero_graph = HeteroGraph(
        node_feature=node_feature,
        node_label=node_label,
        edge_index=edge_index,
        directed=True
    )

    print(f"GNN heterogeneous graph: {hetero_graph.num_nodes()} nodes, {hetero_graph.num_edges()} edges")

    # Node feature and node label to device
    for key in hetero_graph.node_feature:
        hetero_graph.node_feature[key] = hetero_graph.node_feature[key].to(args['device'])
    for key in hetero_graph.node_label:
        hetero_graph.node_label[key] = hetero_graph.node_label[key].to(args['device'])

    # Edge_index to sparse tensor and to device
    for key in hetero_graph.edge_index:
        edge_index = hetero_graph.edge_index[key]
        adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                           sparse_sizes=(hetero_graph.num_nodes('paper'), hetero_graph.num_nodes('paper')))
        hetero_graph.edge_index[key] = adj.t().to(args['device'])

    best_model = None
    best_val = 0

    model = HeteroGNN(hetero_graph, args, aggr="mean").to(args['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    for epoch in range(args['epochs']):
        loss = train(model, optimizer, hetero_graph, train_idx)
        accs, best_model, best_val = test(model, hetero_graph, [train_idx, val_idx, test_idx], best_model, best_val)
        print(
            f"Epoch {epoch + 1}: loss {round(loss, 5)}, "
            f"train micro {round(accs[0][0] * 100, 2)}%, train macro {round(accs[0][1] * 100, 2)}%, "
            f"valid micro {round(accs[1][0] * 100, 2)}%, valid macro {round(accs[1][1] * 100, 2)}%, "
            f"test micro {round(accs[2][0] * 100, 2)}%, test macro {round(accs[2][1] * 100, 2)}%"
        )
    best_accs, _, _ = test(best_model, hetero_graph, [train_idx, val_idx, test_idx], save_preds=True, agg_type="Mean")
    print(
        f"Best model: "
        f"train micro {round(best_accs[0][0] * 100, 2)}%, train macro {round(best_accs[0][1] * 100, 2)}%, "
        f"valid micro {round(best_accs[1][0] * 100, 2)}%, valid macro {round(best_accs[1][1] * 100, 2)}%, "
        f"test micro {round(best_accs[2][0] * 100, 2)}%, test macro {round(best_accs[2][1] * 100, 2)}%"
    )

if __name__ == '__main__':
    main()