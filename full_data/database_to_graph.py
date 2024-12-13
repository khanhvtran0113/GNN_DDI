import xml.etree.ElementTree as ET
from torch_geometric.data import HeteroData
import torch
import json

def encode_items_as_set(feature_items, encoder):
    """Encodes a list of feature items (e.g., targets, enzymes) into a binary vector."""
    vector = torch.zeros(len(encoder), dtype=torch.long)
    for item in feature_items:
        if item in encoder:
            vector[encoder[item]] = 1
    return vector


def parse_drugbank_streaming(xml_file):
    """Memory-efficient XML parsing for drugs with interactions."""
    ns = None
    drugs = []
    drug_features = []
    interactions = []
    descriptions = []

    # Global dictionaries for encoding unique items across all drugs
    global_encoders = {
        "name": set(),
        "superclass": set(),
        "class": set(),
        "subclass": set(),
        "pathways": set(),
        "targets": set(),
        "enzymes": set(),
        "carriers": set(),
        "transporters": set(),
    }

    for event, elem in ET.iterparse(xml_file, events=("start", "end")):
        if event == "start" and ns is None and '}' in elem.tag:
            ns = {'db': elem.tag.split('}')[0].strip('{')}

        if event == "end" and elem.tag.endswith("drug"):
            drug_id = elem.find('db:drugbank-id', ns).text

            # Collect desired features
            features = {
                'name': elem.find('db:name', ns).text if elem.find('db:name', ns) is not None else None,
                'superclass': elem.find('db:classification/db:superclass', ns).text if elem.find(
                    'db:classification/db:superclass', ns) is not None else None,
                'class': elem.find('db:classification/db:class', ns).text if elem.find(
                    'db:classification/db:class', ns) is not None else None,
                'subclass': elem.find('db:classification/db:subclass', ns).text if elem.find(
                    'db:classification/db:subclass', ns) is not None else None,
                'pathways': [pathway.find('db:name', ns).text for pathway in elem.findall('db:pathways/db:pathway', ns)],
                'targets': [target.find('db:name', ns).text for target in elem.findall('db:targets/db:target', ns)],
                'enzymes': [enzyme.find('db:name', ns).text for enzyme in elem.findall('db:enzymes/db:enzyme', ns)],
                'carriers': [carrier.find('db:name', ns).text for carrier in elem.findall('db:carriers/db:carrier', ns)],
                'transporters': [transporter.find('db:name', ns).text for transporter in elem.findall('db:transporters/db:transporter', ns)],
            }

            # Check if all features are present and valid
            if any(value is None or (isinstance(value, list) and not value) for value in features.values()):
                elem.clear()
                continue

            # Add to global encoders
            for key in global_encoders.keys():
                if isinstance(features[key], list):
                    global_encoders[key].update(features[key])
                else:
                    global_encoders[key].add(features[key])

            # Check if the drug has any interactions and document interactions
            interactions_elem = elem.find('db:drug-interactions', ns)
            if interactions_elem is not None:
                for interaction in interactions_elem.findall('db:drug-interaction', ns):
                    target_id = interaction.find('db:drugbank-id', ns).text
                    description = interaction.find('db:description', ns).text
                    if "increased" in description or "decreased" in description:
                        if description not in descriptions:
                            interactions.append((drug_id, target_id, description))
                            descriptions.append(description)

            # Add the drug only if it has interactions
            if any(interaction[0] == drug_id or interaction[1] == drug_id for interaction in interactions):
                drugs.append(drug_id)
                drug_features.append(features)

            # Clear the element from memory
            elem.clear()

    # Finalize global encoders as dictionaries with unique integer mappings
    for key in global_encoders.keys():
        global_encoders[key] = {item: idx for idx, item in enumerate(sorted(global_encoders[key]))}

    return drugs, drug_features, interactions, global_encoders


def build_graph(drugs, drug_features, interactions, global_encoders):
    """Builds a heterogeneous graph from parsed drugs and interactions."""
    # Map drug IDs to indices
    drug_to_idx = {drug: idx for idx, drug in enumerate(drugs)}

    # Encode feature vectors
    feature_vectors = []
    for feature in drug_features:
        # Encode classification features
        classification_vector = torch.tensor(
            [global_encoders[key][feature[key]] for key in ['name', 'superclass', 'class', 'subclass']],
            dtype=torch.long
        )
        # Encode set-based features
        set_vectors = torch.cat([
            encode_items_as_set(feature[key], global_encoders[key])
            for key in ['pathways', 'targets', 'enzymes', 'carriers', 'transporters']
        ])
        # Combine classification and set-based vectors
        feature_vectors.append(torch.cat((classification_vector, set_vectors)))

    # Initialize the heterogeneous graph
    data = HeteroData()
    data['drug'].x = torch.stack(feature_vectors)

    # Add edges
    edge_index = []
    edge_types = []
    for source_id, target_id, description in interactions:
        if source_id in drug_to_idx and target_id in drug_to_idx:
            edge_index.append((drug_to_idx[source_id], drug_to_idx[target_id]))
            edge_types.append(0 if "increased" in description else 1)

    # Convert edge data to tensors
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_types_tensor = torch.tensor(edge_types, dtype=torch.long)

    data['drug', 'affects', 'drug'].edge_index = edge_index_tensor
    data['drug', 'affects', 'drug'].edge_attr = edge_types_tensor

    return data, global_encoders


def save_feature_encoders(encoders, file_path):
    """Save feature encoders to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(encoders, f, indent=4)
    print(f"Feature encoders saved to {file_path}")


def main():
    # Path to the XML file
    xml_file = "drugbank.xml"

    # Parse the database
    drugs, drug_features, interactions, global_encoders = parse_drugbank_streaming(xml_file)

    # Build the graph
    graph, encoders = build_graph(drugs, drug_features, interactions, global_encoders)

    # Save the graph
    graph_path = "ddi_graph.pt"
    torch.save(graph, graph_path)
    print(f"Graph saved to {graph_path}")

    # Save feature encoders
    encoder_path = "feature_encoders.json"
    save_feature_encoders(encoders, encoder_path)


if __name__ == "__main__":
    main()
