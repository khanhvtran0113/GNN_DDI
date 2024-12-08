# GNN_DDI
Elucidating Drug-Drug Interaction Networks Using Deep Learning-Based Graph Neural Network with Automated Architecture Search

The rise of multidrug therapies has highlighted the critical need for accurate and
efficient detection of drug-drug interactions (DDIs), as unplanned interactions can
lead to severe side effects or reduced efficacy in treatment. Traditional in vitro and
in vivo methods for DDI detection are both costly and time-consuming, making
computational approaches a valuable alternative. In this project, we propose an
enhanced Graph Neural Network (GNN)-based model to predict DDIs using the
rich, multi-modal DrugBank dataset, which provides extensive biological context
beyond drug-drug interactions, including molecular structures, pharmacological
properties, and target proteins. Our approach builds on a baseline GNN-DDI
model by leveraging Graph Neural Architecture Search (GNAS) to automatically
optimize the GNN structure for each task. Additionally, we augment the model
with deeper GNN layers, bipartite graph construction for drug pairs, and novel
convolution methods for enhanced performance. By focusing on binary interaction
detection rather than predicting interaction event types, we simplify the model
while still capturing crucial interaction patterns. Our model’s performance will
be evaluated using metrics such as accuracy, F1 score, and AUROC, and we aim
to demonstrate superior generalization and interpretability compared to existing
models. This approach promises not only improved DDI prediction but also
potential applications in drug discovery and adverse reaction prediction.
