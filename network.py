import torch.nn as nn
import torch
from torch_geometric.nn import GCNConv

class ROEN(nn.Module):
    def __init__(self, node_in_channels, edge_in_channels, hidden_channels_node, hidden_channels_edge, mlp_hidden_channels, num_edge_classes):
        super(ROEN, self).__init__()

        # Add two MLP layers before GCN for processing node features
        self.mlp_node_fc1 = nn.Linear(node_in_channels, hidden_channels_node)
        self.mlp_node_fc2 = nn.Linear(hidden_channels_node, hidden_channels_node)
        
        # Add two MLP layers before GCN for processing edge features
        self.mlp_edge_fc1 = nn.Linear(edge_in_channels, hidden_channels_edge)
        self.mlp_edge_fc2 = nn.Linear(hidden_channels_edge, hidden_channels_edge)

        # GCN layers for processing node features
        self.gcn_node_layers1 = GCNConv(hidden_channels_node, hidden_channels_node)
        self.gcn_node_layers2 = GCNConv(hidden_channels_node, hidden_channels_node)
        
        # Fully connected layers for processing edge features
        self.edge_fc1 = nn.Linear(hidden_channels_edge, hidden_channels_edge)
        self.edge_fc2 = nn.Linear(hidden_channels_edge, hidden_channels_edge)

        # LSTM for processing temporal information of node features
        self.lstm_node = nn.LSTM(input_size=hidden_channels_node, hidden_size=hidden_channels_node, batch_first=True)
        
        # LSTM for processing temporal information of edge features
        self.lstm_edge = nn.LSTM(input_size=hidden_channels_edge, hidden_size=hidden_channels_edge, batch_first=True)

        # Final MLP for edge classification
        self.mlp_classifier_fc1 = nn.Linear(2 * hidden_channels_node + hidden_channels_edge, mlp_hidden_channels)  
        self.mlp_classifier_fc2 = nn.Linear(mlp_hidden_channels, num_edge_classes)

        # Store lstm_hidden_dim for forward pass
        self.lstm_hidden_node = hidden_channels_node  
        self.lstm_hidden_edge = hidden_channels_edge
    
    def forward(self, graphs, seq_len):
        batch_edge_predictions = []
        
        # Initialize LSTM hidden states
        h_0_node = torch.zeros(1, seq_len, self.lstm_hidden_node).to(graphs[0].x.device)
        c_0_node = torch.zeros(1, seq_len, self.lstm_hidden_node).to(graphs[0].x.device)
        
        h_0_edge = torch.zeros(1, seq_len, self.lstm_hidden_edge).to(graphs[0].x.device)
        c_0_edge = torch.zeros(1, seq_len, self.lstm_hidden_edge).to(graphs[0].x.device)
        
        node_features_seq = []
        edge_features_seq = []

        for t in range(seq_len):
            x_t = graphs[t].x  # Node features
            edge_index_t = graphs[t].edge_index  # Edge index
            edge_attr_t = graphs[t].edge_attr  # Edge attributes
            
            # Process node features through MLP
            x_t = torch.relu(self.mlp_node_fc1(x_t))
            x_t = torch.relu(self.mlp_node_fc2(x_t)) 
            
            # Process edge features through MLP
            edge_attr_t = torch.relu(self.mlp_edge_fc1(edge_attr_t))
            edge_attr_t = torch.relu(self.mlp_edge_fc2(edge_attr_t))  

            # Process node features through GCN
            x_t = torch.relu(self.gcn_node_layers1(x_t, edge_index_t))
            x_t = torch.relu(self.gcn_node_layers2(x_t, edge_index_t))
            
            # Process edge features through fully connected layers
            edge_attr_t = torch.relu(self.edge_fc1(edge_attr_t))
            edge_attr_t = torch.relu(self.edge_fc2(edge_attr_t))  
            
            # Save node and edge features for each time step
            node_features_seq.append(x_t)
            edge_features_seq.append(edge_attr_t)

        # Pass node feature sequence and edge feature sequence to LSTM for temporal processing
        node_features_seq = torch.stack(node_features_seq, dim=0) 
        edge_features_seq = torch.stack(edge_features_seq, dim=0)  
        
        lstm_out_node, (h_n_node, c_n_node) = self.lstm_node(node_features_seq, (h_0_node, c_0_node))
        
        lstm_out_edge, (h_n_edge, c_n_edge) = self.lstm_edge(edge_features_seq, (h_0_edge, c_0_edge))

        # Classify edges at each time step
        for t in range(seq_len):
            edge_index_t = graphs[t].edge_index
            
            # Concatenate source node and target node features
            edge_src = edge_index_t[0]  # Source node index of edges
            edge_dst = edge_index_t[1]  # Target node index of edges
            node_pair_features = torch.cat([lstm_out_node[t][edge_src], lstm_out_node[t][edge_dst]], dim=1) 
            
            # Concatenate node pair features and LSTM output edge features
            edge_features = torch.cat([node_pair_features, lstm_out_edge[t]], dim=1)  
            
            # Use the final MLP for edge classification
            edge_preds = torch.relu(self.mlp_classifier_fc1(edge_features)) 
            edge_preds = self.mlp_classifier_fc2(edge_preds) 
            
            batch_edge_predictions.append(edge_preds)
        
        return batch_edge_predictions 
