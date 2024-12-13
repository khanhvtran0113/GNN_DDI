import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax
import math

# based on code from the 2021 heterogenous graph attention transformer
class HGTConv(MessagePassing):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.2, use_norm = True, use_RTE = True, **kwargs):
        super(HGTConv, self).__init__(node_dim=0, aggr='add', **kwargs)

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.num_types     = num_types
        self.num_relations = num_relations
        self.total_rel     = num_types * num_relations * num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.use_norm      = use_norm
        self.use_RTE       = use_RTE
        self.att           = None
        
        
        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        
        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
        self.relation_pri   = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(torch.ones(num_types))
        self.drop           = nn.Dropout(dropout)
        
        if self.use_RTE:
            self.emb        = RelTemporalEncoding(in_dim)
        
        glorot(self.relation_att)
        glorot(self.relation_msg)
        
    def forward(self, node_inp, node_type, edge_index, edge_type, edge_time):

        return self.propagate(edge_index, node_inp=node_inp, node_type=node_type,
                            edge_type=edge_type, edge_time = edge_time)

    def message(self, edge_index_i, node_inp_i, node_inp_j, node_type_i, node_type_j, edge_type, edge_time):
        data_size = edge_index_i.size(0)
        res_att     = torch.zeros(data_size, self.n_heads).to(node_inp_i.device)
        res_msg     = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)
        
        for source_type in range(self.num_types):
            sb = (node_type_j == int(source_type))
            k_linear = self.k_linears[source_type]
            v_linear = self.v_linears[source_type] 
            for target_type in range(self.num_types):
                tb = (node_type_i == int(target_type)) & sb
                q_linear = self.q_linears[target_type]
                for relation_type in range(self.num_relations):
                    idx = (edge_type == int(relation_type)) & tb
                    if idx.sum() == 0:
                        continue
                    
                    target_node_vec = node_inp_i[idx]
                    source_node_vec = node_inp_j[idx]
                    if self.use_RTE:
                        source_node_vec = self.emb(source_node_vec, edge_time[idx])
                    
                    q_mat = q_linear(target_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = k_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = torch.bmm(k_mat.transpose(1,0), self.relation_att[relation_type]).transpose(1,0)
                    res_att[idx] = (q_mat * k_mat).sum(dim=-1) * self.relation_pri[relation_type] / self.sqrt_dk
                    
                    v_mat = v_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    res_msg[idx] = torch.bmm(v_mat.transpose(1,0), self.relation_msg[relation_type]).transpose(1,0)   
        
        self.att = softmax(res_att, edge_index_i)
        res = res_msg * self.att.view(-1, self.n_heads, 1)
        del res_att, res_msg
        return res.view(-1, self.out_dim)


    def update(self, aggr_out, node_inp, node_type):
        
        aggr_out = F.gelu(aggr_out)
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)
        for target_type in range(self.num_types):
            idx = (node_type == int(target_type))
            if idx.sum() == 0:
                continue
            trans_out = self.drop(self.a_linears[target_type](aggr_out[idx]))
            
            alpha = torch.sigmoid(self.skip[target_type])
            if self.use_norm:
                res[idx] = self.norms[target_type](trans_out * alpha + node_inp[idx] * (1 - alpha))
            else:
                res[idx] = trans_out * alpha + node_inp[idx] * (1 - alpha)
        return res

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)
    
    
    
class DenseHGTConv(MessagePassing):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.2, use_norm = True, use_RTE = True, **kwargs):
        super(DenseHGTConv, self).__init__(node_dim=0, aggr='add', **kwargs)

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.num_types     = num_types
        self.num_relations = num_relations
        self.total_rel     = num_types * num_relations * num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.use_norm      = use_norm
        self.use_RTE       = use_RTE
        self.att           = None
        
        
        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()

        
        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
       
        self.relation_pri   = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.drop           = nn.Dropout(dropout)
        
        if self.use_RTE:
            self.emb            = RelTemporalEncoding(in_dim)
        
        glorot(self.relation_att)
        glorot(self.relation_msg)
        
        
        self.mid_linear  = nn.Linear(out_dim,  out_dim * 2)
        self.out_linear  = nn.Linear(out_dim * 2,  out_dim)
        self.out_norm    = nn.LayerNorm(out_dim)
        
    def forward(self, node_inp, node_type, edge_index, edge_type, edge_time):
        return self.propagate(edge_index, node_inp=node_inp, node_type=node_type, \
                              edge_type=edge_type, edge_time = edge_time)

    def message(self, edge_index_i, node_inp_i, node_inp_j, node_type_i, node_type_j, edge_type, edge_time):
        
        data_size = edge_index_i.size(0)
        
        res_att = torch.zeros(data_size, self.n_heads).to(node_inp_i.device)
        res_msg = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)
        
        for source_type in range(self.num_types):
            sb = (node_type_j == int(source_type))
            k_linear = self.k_linears[source_type]
            v_linear = self.v_linears[source_type] 
            for target_type in range(self.num_types):
                tb = (node_type_i == int(target_type)) & sb
                q_linear = self.q_linears[target_type]
                for relation_type in range(self.num_relations):
                    
                    idx = (edge_type == int(relation_type)) & tb
                    if idx.sum() == 0:
                        continue
                    
                    target_node_vec = node_inp_i[idx]
                    source_node_vec = node_inp_j[idx]
                    if self.use_RTE:
                        source_node_vec = self.emb(source_node_vec, edge_time[idx])
                    '''
                        Step 1: Heterogeneous Mutual Attention
                    '''
                    q_mat = q_linear(target_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = k_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = torch.bmm(k_mat.transpose(1,0), self.relation_att[relation_type]).transpose(1,0)
                    res_att[idx] = (q_mat * k_mat).sum(dim=-1) * self.relation_pri[relation_type] / self.sqrt_dk
                    '''
                        Step 2: Heterogeneous Message Passing
                    '''
                    v_mat = v_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    res_msg[idx] = torch.bmm(v_mat.transpose(1,0), self.relation_msg[relation_type]).transpose(1,0)   
        self.att = softmax(res_att, edge_index_i)
        res = res_msg * self.att.view(-1, self.n_heads, 1)
        del res_att, res_msg
        return res.view(-1, self.out_dim)


    def update(self, aggr_out, node_inp, node_type):
        
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)
        for target_type in range(self.num_types):
            idx = (node_type == int(target_type))
            if idx.sum() == 0:
                continue
            trans_out = self.drop(self.a_linears[target_type](aggr_out[idx])) + node_inp[idx]
            
            if self.use_norm:
                trans_out = self.norms[target_type](trans_out)
                
            trans_out     = self.drop(self.out_linear(F.gelu(self.mid_linear(trans_out)))) + trans_out
            res[idx]      = self.out_norm(trans_out)
        return res

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)


class RelTemporalEncoding(nn.Module):
    def __init__(self, n_hid, max_len = 240, dropout = 0.2):
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) *
                             -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)
    def forward(self, x, t):
        return x + self.lin(self.emb(t))
    
    
    
class GeneralConv(nn.Module):
    def __init__(self, conv_name, in_hid, out_hid, num_types, num_relations, n_heads, dropout, use_norm = True, use_RTE = True):
        super(GeneralConv, self).__init__()
        self.conv_name = conv_name
        if self.conv_name == 'hgt':
            self.base_conv = HGTConv(in_hid, out_hid, num_types, num_relations, n_heads, dropout, use_norm, use_RTE)
        elif self.conv_name == 'dense_hgt':
            self.base_conv = DenseHGTConv(in_hid, out_hid, num_types, num_relations, n_heads, dropout, use_norm, use_RTE)
        elif self.conv_name == 'gcn':
            self.base_conv = GCNConv(in_hid, out_hid)
        elif self.conv_name == 'gat':
            self.base_conv = GATConv(in_hid, out_hid // n_heads, heads=n_heads)
    def forward(self, meta_xs, node_type, edge_index, edge_type, edge_time):
        if self.conv_name == 'hgt':
            return self.base_conv(meta_xs, node_type, edge_index, edge_type, edge_time)
        elif self.conv_name == 'gcn':
            return self.base_conv(meta_xs, edge_index)
        elif self.conv_name == 'gat':
            return self.base_conv(meta_xs, edge_index)
        elif self.conv_name == 'dense_hgt':
            return self.base_conv(meta_xs, node_type, edge_index, edge_type, edge_time)

class Classifier(nn.Module):
    def __init__(self, n_hid, n_out):
        super(Classifier, self).__init__()
        self.n_hid    = n_hid
        self.n_out    = n_out
        self.linear   = nn.Linear(n_hid,  n_out)
    def forward(self, x):
        tx = self.linear(x)
        return torch.log_softmax(tx.squeeze(), dim=-1)
    def __repr__(self):
        return '{}(n_hid={}, n_out={})'.format(
            self.__class__.__name__, self.n_hid, self.n_out)

# this is needed for link prediction (what we aer doing)
class Matcher(nn.Module):
    def __init__(self, n_hid):
        super(Matcher, self).__init__()
        self.left_linear    = nn.Linear(n_hid,  n_hid)
        self.right_linear   = nn.Linear(n_hid,  n_hid)
        self.sqrt_hd  = math.sqrt(n_hid)
        self.cache      = None
    def forward(self, x, y, infer = False, pair = False):
        ty = self.right_linear(y)
        if infer:
            if self.cache != None:
                tx = self.cache
            else:
                tx = self.left_linear(x)
                self.cache = tx
        else:
            tx = self.left_linear(x)
        if pair:
            res = (tx * ty).sum(dim=-1)
        else:
            res = torch.matmul(tx, ty.transpose(0,1))
        return res / self.sqrt_hd
    def __repr__(self):
        return '{}(n_hid={})'.format(
            self.__class__.__name__, self.n_hid)
    


        
class GNN(nn.Module):
    def __init__(self, in_dim, n_hid, num_types, num_relations, n_heads, n_layers, dropout = 0.2, conv_name = 'hgt', prev_norm = False, last_norm = False, use_RTE = True):
        super(GNN, self).__init__()
        self.gcs = nn.ModuleList()
        self.num_types = num_types
        self.in_dim    = in_dim
        self.n_hid     = n_hid
        self.adapt_ws  = nn.ModuleList()
        self.drop      = nn.Dropout(dropout)
        for t in range(num_types):
            self.adapt_ws.append(nn.Linear(in_dim, n_hid))
        for l in range(n_layers - 1):
            self.gcs.append(GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm = prev_norm, use_RTE = use_RTE))
        self.gcs.append(GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm = last_norm, use_RTE = use_RTE))

    def forward(self, node_feature, node_type, edge_time, edge_index, edge_type):
        res = torch.zeros(node_feature.size(0), self.n_hid).to(node_feature.device)
        for t_id in range(self.num_types):
            idx = (node_type == int(t_id))
            if idx.sum() == 0:
                continue
            res[idx] = torch.tanh(self.adapt_ws[t_id](node_feature[idx]))
        meta_xs = self.drop(res)
        del res
        for gc in self.gcs:
            meta_xs = gc(meta_xs, node_type, edge_index, edge_type, edge_time)
        return meta_xs  

# use of this model; needed to be adapted for link prediction. We used the node embeddings from the HGAT for a link prediction task
import torch
from torch_geometric.utils import negative_sampling
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.transforms import RandomLinkSplit

DDI_graph = torch.load("../full_data/ddi_graph.pt")

label = 0 
DDI_graph['drug'].y = torch.full((DDI_graph['drug'].num_nodes,), label, dtype=torch.long)

# connection types
m_type1 = ("drug", 0, "drug")
m_type2 = ("drug", 1, "drug")
# 80-20 split
transform = RandomLinkSplit(
    num_val=0.0,           
    num_test=0.2,          
    is_undirected=False, 
    split_labels=True,    
    edge_types = (("drug", "affects", "drug"))
)
train_data, val_data, test_data = transform(DDI_graph)
train_edge_index = {}
tt1_idx = torch.argwhere(train_data["drug", "affects", "drug"].edge_attr == 0)
tt2_idx = torch.argwhere(train_data["drug", "affects", "drug"].edge_attr == 1)
num_tt1 = len(tt1_idx)
num_tt2 = len(tt2_idx)
perm = torch.randperm(num_tt1)
perm2 = torch.randperm(num_tt2)

train_size = int(num_tt1 * 0.8)
train_size2 = int(num_tt2 * .8)

train_tt1_idx = tt1_idx[perm[:train_size]]
test_tt1_idx = tt1_idx[perm[train_size:]]
train_tt2_idx = tt2_idx[perm2[:train_size2]]
test_tt2_idx = tt2_idx[perm2[train_size2:]]

train_edge_index[m_type1] = train_data["drug", "affects", "drug"].edge_index[:,train_tt1_idx]
train_edge_index[m_type2] = train_data["drug", "affects", "drug"].edge_index[:,train_tt2_idx]

test_edge_index = {}
test_edge_index[m_type1] = test_data["drug", "affects", "drug"].edge_index[:,test_tt1_idx]
test_edge_index[m_type2] = test_data["drug", "affects", "drug"].edge_index[:,test_tt2_idx]




node_feature = DDI_graph["drug"].x
num_nodes = node_feature.size(0)

node_type = torch.zeros(num_nodes, dtype=torch.long)

from torch.nn import Module

class HGAT(Module):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, n_layers, dropout):
        super(HGAT, self).__init__()
        self.gnn = GNN(
            in_dim=in_dim,
            n_hid=out_dim,
            num_types=num_types,
            num_relations=num_relations,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            conv_name='hgt',
            prev_norm=False,
            last_norm=False,
            use_RTE=True
        )
    
    def forward(self, node_feature, node_type, edge_index, edge_type, edge_time):
        return self.gnn(node_feature, node_type, edge_time, edge_index, edge_type)

# hyperparameters for our transformer
model = HGAT(
    in_dim=node_feature.size(1),  
    out_dim=32,                  
    num_types=1,                 
    num_relations=2,             
    n_heads=2,                   
    n_layers=2,                  
    dropout=0.05
)

train_edge_index = torch.cat([train_edge_index[m_type1], train_edge_index[m_type2]], dim=1)
train_edge_type = torch.cat([
    torch.zeros(train_data["drug", "affects", "drug"].edge_index[:,train_tt1_idx].size(1), dtype=torch.float),
    torch.ones(train_data["drug", "affects", "drug"].edge_index[:,train_tt2_idx].size(1), dtype=torch.float)   
])

test_edge_index = torch.cat([test_edge_index[m_type1], test_edge_index[m_type2]], dim=1)
test_edge_type = torch.cat([
    torch.zeros(test_data["drug", "affects", "drug"].edge_index[:,test_tt1_idx].size(1), dtype=torch.float),  
    torch.ones(test_data["drug", "affects", "drug"].edge_index[:,test_tt2_idx].size(1), dtype=torch.float)   
])

train_edge_time = torch.zeros(train_edge_index.size(1), dtype=torch.float)
test_edge_time = torch.zeros(test_edge_index.size(1), dtype=torch.float)
# used for link prediction
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

matcher = Matcher(32)  
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
# training step
for epoch in range(250):
    model.train()
    optimizer.zero_grad()

    train_edge_index = train_edge_index.squeeze(-1)
    node_embeddings = model(node_feature.to(dtype=torch.float), node_type.to(dtype=torch.float), train_edge_index.to(dtype=torch.int64), train_edge_type.to(dtype=torch.float), train_edge_time.to(dtype=torch.int64))

    src, dst = train_edge_index
    pos_scores = matcher(node_embeddings[src], node_embeddings[dst], pair=True)

    neg_edge_index = negative_sampling(
        edge_index=train_edge_index, num_nodes=num_nodes, num_neg_samples=src.size(0)
    )
    neg_src, neg_dst = neg_edge_index
    neg_scores = matcher(node_embeddings[neg_src], node_embeddings[neg_dst], pair=True)

    labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
    scores = torch.cat([pos_scores, neg_scores])

    loss = F.binary_cross_entropy_with_logits(scores, labels)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# evaluation step
model.eval()
with torch.no_grad():
    test_edge_index = test_edge_index.squeeze(-1)
    node_embeddings = model(node_feature.to(dtype=torch.float), node_type.to(dtype=torch.float), test_edge_index.to(dtype=torch.int64), test_edge_type.to(dtype=torch.float), test_edge_time.to(dtype=torch.int64))

    pos_scores = matcher(node_embeddings[src], node_embeddings[dst], pair=True)
    neg_scores = matcher(node_embeddings[neg_src], node_embeddings[neg_dst], pair=True)

    labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
    scores = torch.cat([pos_scores, neg_scores])

    auc = roc_auc_score(labels.cpu(), scores.cpu())
    predictions = (scores > 0.5).cpu().numpy()

    labels_np = labels.cpu().numpy()

    accuracy = accuracy_score(labels_np, predictions)
    precision = precision_score(labels_np, predictions)
    recall = recall_score(labels_np, predictions)
    f1 = f1_score(labels_np, predictions)

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

with open('../full_data/feature_encoders.json', 'r') as f:
    feature_encoders = json.load(f)

name_mapping = {v: k for k, v in feature_encoders['name'].items()} 

nx_graph = to_networkx(DDI_graph, edge_attrs=['edge_attr'], to_undirected=False)
edge_types = nx.get_edge_attributes(nx_graph, 'edge_attr')

unique_edge_types = set(edge_types.values())
color_map = {edge_type: plt.cm.tab10(i) for i, edge_type in enumerate(unique_edge_types)}

edge_colors = [color_map[edge_types[edge]] for edge in nx_graph.edges]

pos = nx.spring_layout(nx_graph, k=2.3)  
plt.figure(figsize=(15, 10))

nx.draw_networkx_nodes(nx_graph, pos, node_size=900, node_color='skyblue')
node_labels = {node: name_mapping.get(node, f"Node {node}") for node in nx_graph.nodes}
nx.draw_networkx_labels(nx_graph, pos, labels=node_labels, font_size=4, font_color='black')

arc_rad = 0.8  
for edge, color in zip(nx_graph.edges, edge_colors):
    if nx_graph.has_edge(edge[1], edge[0]):  
        nx.draw_networkx_edges(
            nx_graph, pos,
            edgelist=[edge],
            edge_color=[color],
            connectionstyle=f'arc3,rad={arc_rad}',
            arrowstyle='-|>',
            arrowsize=30,
            width=1
        )
    else:  
        nx.draw_networkx_edges(
            nx_graph, pos,
            edgelist=[edge],
            edge_color=[color],
            arrowstyle='-|>',
            arrowsize=30,
            width=1
        )

edge_type_labels = {0: "Increases", 1: "Decreases"}  
legend_labels = [edge_type_labels[edge_type] for edge_type in sorted(unique_edge_types)]
handles = [plt.Line2D([0], [0], color=color_map[edge_type], lw=2) for edge_type in sorted(unique_edge_types)]
plt.legend(handles, legend_labels, title="Edge Types", loc="upper right")

plt.title("Drug-Drug Interaction Graph")
plt.show()