# GNN_DDI

### Deep Learning for Drug-Drug Interaction Prediction Using Modern Heterogeneous Graph Architectures

Ishaan Singh, Khanh Tran, Thomas Hatcher

This repository contains various models designed for
link prediction on a directed heterogeneous graph, 
wherein there is a single node type representing drugs, 
and two edge types representing either an attenuating 
or exacerbating DDI. For a message type of 
(source node, attenuating, destination node), this 
means that the source node drug decreases the effect 
of the destination node drug. Similarly, for a message
type of (source node, exacerbating, destination node),
this means that the source node drug increases the 
effect of the destination node drug. Thus, the 
overarching goal is to design an optimal model to 
predict whether a network of unseen drugs will have 
DDIs, and if so, whether the effect of one drug will 
exacerbate or attenuate the effect of connected drugs. 
Such predictions hold significant medical importance, 
as they can guide healthcare providers in predicting 
adverse drug interaction in the absence of extensive 
clinical experimentation.

### Modules
In this repo, you'll find four main directories;
1. full_data: contains the software needed to parse the DrugBank database and create a graph. This also contains script
to make sure the outputted graph is consistent with uses for downstream tasks. In addition, non-float features are
encoded for machine use, so the feature encoders are also in this directory.
2. hetero_trans: contains software that trains and tests a model with hetergenous graph transformer (HGT) architecture.
HGT architecture is used for modeling heterogeneous graphs by using node- and edge-type dependent parameters to 
characterize the heterogeneous attention over each edge, which therefore allows HGT to maintain representations for 
different types of nodes and edges. To train/test the model on the ddi graph, simply run HGT.py.
3. heterogcn: contains software that trains and test a model with heterogeneous link prediction with relational graph 
convolutional network (HeteroLP-R-GCN). The HeteroLP-R-GCN architecture extends the traditional GCN by introducing 
relation-specific parameters, allowing them to effectively model heterogeneous graphs with multiple edge types. To train/test
this model, run gcn.py.
4. SeHGNN: Simple and Efficient Heterogeneous Graph Neural Network (SeHGNN) is an architecture originally proposed 
to easily capture structural information in heterogeneous graphs. This directory contains a model based on this architecture
with the added adaptation of complEx embeddings. To train/test this model, run SeHGNN.py

### More information
You can find a Medium draft with a more detailed description of the motivation and model architecture as well as test 
results at this [link](https://medium.com/@Ishaanksingh/deep-learning-for-drug-drug-interaction-prediction-using-modern-heterogeneous-graph-architectures-d369178299a4)


