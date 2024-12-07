# Handle data preprocessing, including cleaning, feature extraction, and transformation. 
# For DDI, this could involve converting raw drug interaction data into a graph format 
# compatible with torch_geometric.
import sqlite3
import torch
from torch_geometric.data import Data
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(db_path):
    """Loads data from Event.db and preprocesses it into torch_geometric format."""
    conn = sqlite3.connect(db_path)
    
    # Load drug data and extract features
    drug_df = pd.read_sql_query("SELECT * FROM drug", conn)
    drug_ids = drug_df['id'].values       # Unique identifier for each drug
    drug_names = drug_df['name'].values    # Drug names
    
    # Encode categorical features using LabelEncoder
    label_encoder = LabelEncoder()

    # Encode 'target', 'enzyme', and 'pathway' columns
    drug_df['target_encoded'] = label_encoder.fit_transform(drug_df['target'].astype(str))
    drug_df['enzyme_encoded'] = label_encoder.fit_transform(drug_df['enzyme'].astype(str))
    drug_df['pathway_encoded'] = label_encoder.fit_transform(drug_df['pathway'].astype(str))

    # Create the feature vector for each drug
    drug_features = drug_df[['target_encoded', 'enzyme_encoded', 'pathway_encoded']].values

    # Convert drug features list to a tensor
    drug_features_tensor = torch.tensor(drug_features, dtype=torch.float)
    
    # Load DDI edges
    event_df = pd.read_sql_query("SELECT * FROM event", conn)
    edge_index = []
    for _, row in event_df.iterrows():
        source_id = row['id1'] 
        target_id = row['id2']
        edge_index.append((source_id, target_id))
    
    # Mapping drug IDs to node indices
    id_to_index = {drug_id: idx for idx, drug_id in enumerate(drug_ids)}
    edges = [(id_to_index[src], id_to_index[dst]) for src, dst in edge_index]
    edge_index_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Construct the torch_geometric data object, with names included as metadata
    data = Data(x=drug_features_tensor, edge_index=edge_index_tensor)
    data.drug_names = drug_names  # Add drug names as an attribute for reference
    
    conn.close()
    return data

def save_processed_data(data, output_path):
    """Save processed data object to a file for easy reloading."""
    torch.save(data, output_path)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    db_path = "event.db"
    output_path = "processed_data.pt"
    
    # Load and process the data
    data = load_data(db_path)
    
    # Save the processed data
    save_processed_data(data, output_path)

