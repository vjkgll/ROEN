import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, f1_score, roc_auc_score, precision_score
from untils import create_ip_mapping_2020, create_graph_data_2020, GraphDataset
from network import ROEN
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader 
import torch
import os
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("data/Darknet/Darknet.csv")
data.drop(columns=['Label.1'], inplace=True)

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(data['Label'])

# Parse timestamps
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Create time windows in minutes
data['Timestamp'] = data['Timestamp'].dt.floor('T')

# Group data by time window
grouped_data = data.groupby('Timestamp')

# Dynamically build graph for each time window and store to list
graph_data_seq = []
for name, group in grouped_data:
    
    # Dynamically generate IP mapping for each time window
    ip_to_id = create_ip_mapping_2020(group)
    
    # Build graph for current time window and pass time window name
    graph_data_seq.append(create_graph_data_2020(group, ip_to_id, label_encoder, time_window=name))
    
# Initialize device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Split dataset into training and testing sets
train_data_seq, test_data_seq = train_test_split(graph_data_seq, test_size=0.2, random_state=42)

# Create DataLoader for training and testing sets
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
            edge_in_channels=79, 
            hidden_channels_node=128,
            hidden_channels_edge=128, 
            mlp_hidden_channels=128, 
            num_edge_classes=9).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train(model, train_dataloader, test_dataloader, optimizer, criterion, 150, 10, 'models/2020')
