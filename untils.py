import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset
from torch_geometric.data import Data

# Create a dynamic mapping of IP to nodes, which updates every time window
def create_ip_mapping_2017(time_slice):
    ip_list = pd.concat([time_slice[' Source IP'], time_slice[' Destination IP']]).unique()
    return {ip: i for i, ip in enumerate(ip_list)}

def create_ip_mapping_2012(time_slice):
    ip_list = pd.concat([time_slice['source'], time_slice['destination']]).unique()
    return {ip: i for i, ip in enumerate(ip_list)}

def create_ip_mapping_2020(time_slice):
    ip_list = pd.concat([time_slice['Src IP'], time_slice['Dst IP']]).unique()
    return {ip: i for i, ip in enumerate(ip_list)}

def create_graph_data_2017(time_slice, ip_to_id, time_window):
    
    label_encoder = LabelEncoder()
    
    # Replace Inf with NaN for imputation
    time_slice.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaN values forward and backward
    time_slice.fillna(method='ffill', inplace=True)
    time_slice.fillna(method='bfill', inplace=True)
    
    # Select numeric columns (strictly exclude non-numeric and string types)
    numeric_cols = time_slice.select_dtypes(include=['float64', 'int64']).columns.difference(['Flow ID', ' Source IP', ' Destination IP', ' Label'])
    
    # Standardize only the numeric columns
    scaler = StandardScaler()
    time_slice[numeric_cols] = scaler.fit_transform(time_slice[numeric_cols])
    
    # Convert Source IP and Destination IP to node IDs
    source_ids = time_slice[' Source IP'].map(ip_to_id).values
    destination_ids = time_slice[' Destination IP'].map(ip_to_id).values
    
    # Build edge index (Source -> Destination)
    edge_index = torch.tensor([source_ids, destination_ids], dtype=torch.long)

    # Get Source Port and Destination Port as node features
    source_ports = time_slice[' Source Port'].values
    destination_ports = time_slice[' Destination Port'].values
    
    # Build node feature matrix
    num_nodes = len(ip_to_id)  # Total number of nodes
    node_features = torch.zeros((num_nodes, 1), dtype=torch.float)  # Initialize node features

    # Populate each node's feature into the corresponding position
    for i, (src_id, dst_id, src_port, dst_port) in enumerate(zip(source_ids, destination_ids, source_ports, destination_ports)):
        node_features[src_id] = src_port  # Source IP node feature is Source Port
        node_features[dst_id] = dst_port  # Destination IP node feature is Destination Port
    
    # Edge features: drop 'Source IP', 'Destination IP', 'Label', 'Flow ID', ' Timestamp', use other numeric columns as edge features
    edge_attr_df = time_slice.drop(columns=[' Source IP', ' Destination IP', ' Label', 'Flow ID', ' Timestamp'])

    # Convert to tensor
    edge_attr = torch.tensor(edge_attr_df.values, dtype=torch.float)
    
    # Use LabelEncoder to encode 'Label' column as integers
    labels_encoded = label_encoder.fit_transform(time_slice[' Label'])

    # Only process labels if there are edges
    if edge_index.size(1) > 0:
        edge_labels = torch.tensor(labels_encoded, dtype=torch.long)
        graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, edge_labels=edge_labels)
    else:
        print(f"Skipping time window {time_window} as there are no edges")
        graph_data = None  # Do not create graph if there are no edges
    
    return graph_data

def create_graph_data_2012(time_slice, ip_to_id, time_window):
    
    label_encoder = LabelEncoder()
    
    # Replace Inf with NaN for imputation
    time_slice.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaN values forward and backward
    time_slice.fillna(method='ffill', inplace=True)
    time_slice.fillna(method='bfill', inplace=True)
    
    # Select numeric columns (strictly exclude non-numeric and string types)
    numeric_cols = time_slice.select_dtypes(include=['float64', 'int64']).columns.difference(['appName', 'source', 'destination', 'Label','generated'])
    
    # Standardize only the numeric columns
    scaler = StandardScaler()
    time_slice[numeric_cols] = scaler.fit_transform(time_slice[numeric_cols])
    
    # Convert Source IP and Destination IP to node IDs
    source_ids = time_slice['source'].map(ip_to_id).values
    destination_ids = time_slice['destination'].map(ip_to_id).values
    
    # Build edge index (Source -> Destination)
    edge_index = torch.tensor([source_ids, destination_ids], dtype=torch.long)

    # Get Source Port and Destination Port as node features
    source_ports = time_slice['sourcePort'].values
    destination_ports = time_slice['destinationPort'].values
    
    # Build node feature matrix
    num_nodes = len(ip_to_id)  # Total number of nodes
    node_features = torch.zeros((num_nodes, 1), dtype=torch.float)  # Initialize node features

    # Populate each node's feature into the corresponding position
    for i, (src_id, dst_id, src_port, dst_port) in enumerate(zip(source_ids, destination_ids, source_ports, destination_ports)):
        node_features[src_id] = src_port  # Source IP node feature is Source Port
        node_features[dst_id] = dst_port  # Destination IP node feature is Destination Port
    
    # Edge features: drop 'Source IP', 'Destination IP', 'Label', 'Flow ID', 'Timestamp', use other numeric columns as edge features
    edge_attr_df = time_slice.drop(columns=['appName', 'source', 'destination', 'Label','generated'])

    # Convert to tensor
    edge_attr = torch.tensor(edge_attr_df.values, dtype=torch.float)
    
    # Use LabelEncoder to encode 'Label' column as integers
    labels_encoded = label_encoder.fit_transform(time_slice['Label'])

    # Only process labels if there are edges
    if edge_index.size(1) > 0:
        edge_labels = torch.tensor(labels_encoded, dtype=torch.long)
        graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, edge_labels=edge_labels)
    else:
        print(f"Skipping time window {time_window} as there are no edges")
        graph_data = None  # Do not create graph if there are no edges
    
    return graph_data


def create_graph_data_2020(time_slice, ip_to_id, time_window):
    
    label_encoder = LabelEncoder()
    
    # Replace Inf with NaN for imputation
    time_slice.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaN values forward and backward
    time_slice.fillna(method='ffill', inplace=True)
    time_slice.fillna(method='bfill', inplace=True)
    
    # Select numeric columns (strictly exclude non-numeric and string types)
    numeric_cols = time_slice.select_dtypes(include=['float64', 'int64']).columns.difference(['Src IP', 'Dst IP', 'Flow ID', 'Label'])
    
    # Standardize only the numeric columns
    scaler = StandardScaler()
    time_slice[numeric_cols] = scaler.fit_transform(time_slice[numeric_cols])
    
    # Convert Source IP and Destination IP to node IDs
    source_ids = time_slice['Src IP'].map(ip_to_id).values
    destination_ids = time_slice['Dst IP'].map(ip_to_id).values
    
    # Build edge index (Source -> Destination)
    edge_index = torch.tensor([source_ids, destination_ids], dtype=torch.long)

    # Get Source Port and Destination Port as node features
    source_ports = time_slice['Src Port'].values
    destination_ports = time_slice['Dst Port'].values
    
    # Build node feature matrix
    num_nodes = len(ip_to_id)  # Total number of nodes
    node_features = torch.zeros((num_nodes, 1), dtype=torch.float)  # Initialize node features

    # Populate each node's feature into the corresponding position
    for i, (src_id, dst_id, src_port, dst_port) in enumerate(zip(source_ids, destination_ids, source_ports, destination_ports)):
        node_features[src_id] = src_port  # Source IP node feature is Source Port
        node_features[dst_id] = dst_port  # Destination IP node feature is Destination Port
    
    # Edge features: drop 'Source IP', 'Destination IP', 'Label', 'Flow ID', 'Timestamp', use other numeric columns as edge features
    edge_attr_df = time_slice.drop(columns=['Src IP', 'Dst IP', 'Flow ID','Label','Timestamp'])

    # Convert to tensor
    edge_attr = torch.tensor(edge_attr_df.values, dtype=torch.float)
    
    # Use LabelEncoder to encode 'Label' column as integers
    labels_encoded = label_encoder.fit_transform(time_slice['Label'])

    # Only process labels if there are edges
    if edge_index.size(1) > 0:
        edge_labels = torch.tensor(labels_encoded, dtype=torch.long)
        graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, edge_labels=edge_labels)
    else:
        print(f"Skipping time window {time_window} as there are no edges")
        graph_data = None  # Do not create graph if there are no edges
    
    return graph_data


# Custom dataset
class GraphDataset(Dataset):
    def __init__(self, graph_data_seq, device):
        self.graph_data_seq = graph_data_seq
        self.device = device  

    def __len__(self):
        return len(self.graph_data_seq)

    def __getitem__(self, idx):
        graph_data = self.graph_data_seq[idx]
        
        # Ensure each graph_data's data is moved to the specified device
        graph_data.x = graph_data.x.to(self.device)
        graph_data.edge_index = graph_data.edge_index.to(self.device)
        graph_data.edge_attr = graph_data.edge_attr.to(self.device)
        graph_data.edge_labels = graph_data.edge_labels.to(self.device)
        
        return graph_data

