import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool
from torch_geometric.data import Data
import numpy as np

class DropoutPredictionGNN(nn.Module):
    """Graph Neural Network for MOOC dropout prediction"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1, 
                 num_layers: int = 2, dropout_rate: float = 0.5, conv_type: str = 'GCN'):
        super(DropoutPredictionGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        
        if conv_type == 'GCN':
            self.convs.append(GCNConv(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        elif conv_type == 'SAGE':
            self.convs.append(SAGEConv(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def get_embeddings(self, x, edge_index):
        """Get node embeddings without classification"""
        # Graph convolution layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # No activation after last conv layer
                x = F.relu(x)
                x = self.dropout(x)
        return x
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass through the GNN"""
        # Get embeddings
        x = self.get_embeddings(x, edge_index)
        
        # Global pooling for graph-level prediction (if batch is provided)
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # Classification
        out = self.classifier(x)
        return out

class LinkPredictionGNN(nn.Module):
    """GNN for link prediction between users and activities"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super(LinkPredictionGNN, self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x, edge_index):
        """Generate node embeddings"""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        return x
    
    def predict_link(self, node_embeddings, edge_index):
        """Predict link probability between nodes"""
        # Get embeddings for source and target nodes
        src_embeddings = node_embeddings[edge_index[0]]
        dst_embeddings = node_embeddings[edge_index[1]]
        
        # Compute dot product similarity
        link_logits = (src_embeddings * dst_embeddings).sum(dim=1)
        link_probs = torch.sigmoid(link_logits)
        
        return link_probs

class GraphDataProcessor:
    """Process NetworkX graphs for PyTorch Geometric"""
    
    @staticmethod
    def networkx_to_pyg(G, node_features=None, edge_labels=None):
        """Convert NetworkX graph to PyTorch Geometric data"""
        # Create node mapping
        node_list = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        
        # Create edge index
        edge_list = []
        edge_attr = []
        
        for src, dst, data in G.edges(data=True):
            src_idx = node_to_idx[src]
            dst_idx = node_to_idx[dst]
            edge_list.append([src_idx, dst_idx])
            
            # Extract edge attributes
            if edge_labels and 'label' in data:
                edge_attr.append(data['label'])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Create node features
        if node_features is None:
            # Use one-hot encoding based on node type or degree
            num_nodes = len(node_list)
            x = torch.eye(num_nodes, dtype=torch.float)
        else:
            x = torch.tensor(node_features, dtype=torch.float)
        
        # Create edge attributes
        if edge_attr:
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        else:
            data = Data(x=x, edge_index=edge_index)
        
        return data
    
    @staticmethod
    def create_node_features(G, feature_type='degree'):
        """Create node features from graph structure"""
        nodes = list(G.nodes())
        
        if feature_type == 'degree':
            # Use degree as feature
            degrees = [G.degree(node) for node in nodes]
            features = np.array(degrees).reshape(-1, 1)
        elif feature_type == 'centrality':
            # Use multiple centrality measures
            import networkx as nx
            degree_cent = nx.degree_centrality(G)
            between_cent = nx.betweenness_centrality(G, k=1000)
            
            features = []
            for node in nodes:
                features.append([
                    degree_cent.get(node, 0),
                    between_cent.get(node, 0),
                    G.degree(node)
                ])
            features = np.array(features)
        else:
            # Default: one-hot encoding
            features = np.eye(len(nodes))
        
        return features

if __name__ == "__main__":
    # Example model creation
    model = DropoutPredictionGNN(input_dim=10, hidden_dim=64)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Example forward pass
    x = torch.randn(100, 10)  # 100 nodes, 10 features each
    edge_index = torch.randint(0, 100, (2, 200))  # 200 edges
    
    output = model(x, edge_index)
    print(f"Output shape: {output.shape}")