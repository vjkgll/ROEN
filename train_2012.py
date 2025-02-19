import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, f1_score, roc_auc_score, precision_score
from untils import create_ip_mapping_2012, create_graph_data_2012, GraphDataset
from network import ROEN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from torch_geometric.loader import DataLoader 
import torch
import os
import torch.nn as nn
import torch.optim as optim

data1 = pd.read_csv("data/ISCXIDS2012/TestbedMonJun14Flows.csv")
data2 = pd.read_csv("data/ISCXIDS2012/TestbedSatJun12Flows.csv")
data3 = pd.read_csv("data/ISCXIDS2012/TestbedSunJun13Flows.csv")
# data4 = pd.read_csv("data/ISCXIDS2012/TestbedThuJun17Flows.csv")
# data5 = pd.read_csv("data/ISCXIDS2012/TestbedTueJun15Flows.csv")
# data6 = pd.read_csv("data/ISCXIDS2012/TestbedWedJun16Flows.csv")

# Combine all dataframes into a list
data_list = [data1, data2, data3]

# Use concat function to join all dataframes
data = pd.concat(data_list, ignore_index=True)

# Parse time stamps
data['generated'] = pd.to_datetime(data['generated'])
data['startDateTime'] = pd.to_datetime(data['startDateTime'])
data['stopDateTime'] = pd.to_datetime(data['stopDateTime'])

# Calculate duration of each session in minutes
data['Duration of time'] = data['stopDateTime'] - data['startDateTime']
data['Duration of time'] = data['Duration of time'].dt.total_seconds()/60
data.drop(columns=['stopDateTime','startDateTime'], inplace=True)

# Fill missing values and forward fill then backward fill
data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)

# Calculate non-empty value ratio for each column
non_null_ratio = data.notna().mean()

# Select columns to drop where the non-empty value ratio is less or equal to 0.3
columns_to_drop = non_null_ratio[non_null_ratio <= 0.3].index
data.drop(columns=columns_to_drop, inplace=True)

# One-hot encoding for TCP flags description
encoder = OneHotEncoder()
label_encoder = LabelEncoder()
tcp_flags = data[['sourceTCPFlagsDescription', 'destinationTCPFlagsDescription']].fillna('')
encoded_tcp_flags = encoder.fit_transform(tcp_flags)
encoded_tcp_flags_df = pd.DataFrame(encoded_tcp_flags.toarray(), columns=encoder.get_feature_names_out(['sourceTCPFlagsDescription', 'destinationTCPFlagsDescription']))
data.drop(columns=['sourceTCPFlagsDescription','destinationTCPFlagsDescription'], inplace=True)
data = pd.concat([data, encoded_tcp_flags_df], axis=1)

# Label encoding for protocol name
data['protocolName'] = label_encoder.fit_transform(data['protocolName'])
data['sourcePayloadAsUTF'] = label_encoder.fit_transform(data['sourcePayloadAsUTF'])
data['destinationPayloadAsUTF'] = label_encoder.fit_transform(data['destinationPayloadAsUTF'])

# Label encoding for direction
data['direction'] = label_encoder.fit_transform(data['direction'])

# Process load data, here we take length as example
data['sourcePayloadLength'] = data['sourcePayloadAsBase64'].apply(len)
data['destinationPayloadLength'] = data['destinationPayloadAsBase64'].apply(len)
data.drop(columns=['sourcePayloadAsBase64','destinationPayloadAsBase64'], inplace=True)

# Create time window by minute
data['generated'] = data['generated'].dt.floor('T')

# Group data by time window
grouped_data = data.groupby('generated')

# Assign dynamic IP mappings for each time window and build graph dynamically
graph_data_seq = []
for name, group in grouped_data:
    ip_mapping = create_ip_mapping_2012(group)
    graph_data = create_graph_data_2012(group, ip_mapping, time_window=name)
    graph_data_seq.append(graph_data)

# Initialize device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Split the dataset into training set and test set
train_data_seq, test_data_seq = train_test_split(graph_data_seq, test_size=0.2, random_state=42)

# Create DataLoader for training and test sets
train_dataset = GraphDataset(train_data_seq, device=device)
test_dataset = GraphDataset(test_data_seq, device=device)
train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=0)

def evaluate(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_labels = []
    all_preds = []

    with torch.no_grad():  # Turn off gradient calculation
        for graph_data in dataloader:
            graph_data = graph_data.to(device)

            # Get model predictions
            edge_predictions = model(graphs=[graph_data], seq_len=1)
            edge_labels_batch = graph_data.edge_labels.to(device)
            
            # Get predicted class for each edge
            _, predicted = torch.max(edge_predictions[0], dim=1)

            # Store true labels and predictions
            all_labels.extend(edge_labels_batch.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # Calculate accuracy
    accuracy = 100 * (np.array(all_preds) == np.array(all_labels)).sum() / len(all_labels)

    # Calculate recall
    recall = recall_score(all_labels, all_preds, average='weighted')

    # Calculate F1 score
    f1 = f1_score(all_labels, all_preds, average='weighted')
       
    # Calculate precision
    precision = 100 * precision_score(all_labels, all_preds, average='weighted')

    # Calculate AUC value (for multi-class tasks, need to One-Hot encode labels)
    try:
        auc = roc_auc_score(all_labels, all_preds, multi_class='ovo')
    except ValueError:
        auc = float('nan')  # If AUC cannot be calculated, return NaN
        
    model.train()  # Revert model to training mode
 
    # Return evaluation metrics and confusion matrix values
    return accuracy, precision, recall, f1, auc

def train(model, train_dataloader, test_dataloader, optimizer, criterion, num_epochs, eval_interval, save_dir):
    model.train()  # Set the model to training mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      
    for epoch in range(num_epochs):
        total_loss = 0

        # Training phase
        for graph_data in train_dataloader:  # Iterate through training data
            graph_data = graph_data.to(device)

            optimizer.zero_grad()

            # Get predictions
            edge_predictions = model(graphs=[graph_data], seq_len=1)

            # Get edge labels for current batch
            edge_labels_batch = graph_data.edge_labels.to(device)

            # Calculate loss
            loss = criterion(edge_predictions[0], edge_labels_batch)

            # Backpropagation
            loss.backward()

            # Gradient clipping to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

            # Update model parameters
            optimizer.step()

            total_loss += loss.item()

        # Print average loss for current epoch
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_dataloader):.4f}')
        
        # Evaluate every eval_interval epochs
        if (epoch + 1) % eval_interval == 0:
            accuracy, precision, recall, f1, auc = evaluate(model, test_dataloader)
            print(f'Epoch {epoch+1}/{num_epochs}, Test Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}%, Recall: {recall:.2f}, F1 Score: {f1:.2f}, AUC: {auc:.2f}')

            save_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)
            print(f'Model saved to {save_path}')
            
# Initialize and train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ROEN(node_in_channels=1, 
            edge_in_channels=55, 
            hidden_channels_node=128,
            hidden_channels_edge=128, 
            mlp_hidden_channels=128, 
            num_edge_classes=2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train(model, train_dataloader, test_dataloader, optimizer, criterion, 150, 10, 'models/2012')