import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Tuple, Dict, Any
import networkx as nx

from .gnn_model import DropoutPredictionGNN, GraphDataProcessor

class MOOCGNNTrainer:
    """Trainer class for MOOC Dropout Prediction GNN"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        
        # Create model directory if it doesn't exist
        Path(config.model.model_save_dir).mkdir(parents=True, exist_ok=True)
    
    def prepare_graph_data(self, graph: nx.Graph, df: pd.DataFrame) -> Tuple[Data, np.ndarray]:
        """Prepare graph data for PyTorch Geometric"""
        self.logger.info("Preparing graph data for PyTorch Geometric...")
        
        # Create node features
        processor = GraphDataProcessor()
        node_features = processor.create_node_features(
            graph, 
            feature_type=self.config.model.node_feature_type
        )
        
        # Convert NetworkX graph to PyG format
        pyg_data = processor.networkx_to_pyg(graph, node_features)
        
        # Create labels for edges based on dropout information
        # Map action IDs to labels
        action_to_label = dict(zip(df['ACTIONID'], df['LABEL']))
        
        # For bipartite graph, we need to create labels for user-activity pairs
        edge_labels = []
        user_nodes = [n for n in graph.nodes() if str(n).startswith('user_')]
        activity_nodes = [n for n in graph.nodes() if str(n).startswith('activity_')]
        
        # Create node mapping for easier processing
        node_list = list(graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        
        # Extract labels for existing edges
        labels = []
        for src, dst in graph.edges():
            # For bipartite graph, find corresponding action and get its label
            src_idx = node_to_idx[src]
            dst_idx = node_to_idx[dst]
            
            # Try to find the action ID for this user-activity pair
            # This is a simplification - in practice you'd need more sophisticated mapping
            if str(src).startswith('user_') and str(dst).startswith('activity_'):
                user_id = int(str(src).replace('user_', ''))
                activity_id = int(str(dst).replace('activity_', ''))
                
                # Find actions for this user-activity pair
                matching_actions = df[(df['USERID'] == user_id) & (df['TARGETID'] == activity_id)]
                if not matching_actions.empty:
                    # Use the most recent action's label
                    label = matching_actions.iloc[-1]['LABEL']
                    labels.append(label)
                else:
                    labels.append(0)  # Default to no dropout
            else:
                labels.append(0)  # Default for non-user-activity edges
        
        labels = np.array(labels, dtype=np.float32)
        
        self.logger.info(f"Prepared graph with {pyg_data.x.shape[0]} nodes, {pyg_data.edge_index.shape[1]} edges")
        self.logger.info(f"Node features shape: {pyg_data.x.shape}")
        self.logger.info(f"Label distribution: {np.bincount(labels.astype(int))}")
        
        return pyg_data, labels
    
    def create_train_test_split(self, pyg_data: Data, labels: np.ndarray) -> Tuple[Data, Data, np.ndarray, np.ndarray]:
        """Create train/test split for the graph data"""
        num_edges = pyg_data.edge_index.shape[1]
        edge_indices = np.arange(num_edges)
        
        train_idx, test_idx = train_test_split(
            edge_indices, 
            test_size=self.config.evaluation.test_size,
            random_state=self.config.evaluation.random_state,
            stratify=labels
        )
        
        # Create train data
        train_edge_index = pyg_data.edge_index[:, train_idx]
        train_data = Data(x=pyg_data.x, edge_index=train_edge_index)
        
        # Create test data  
        test_edge_index = pyg_data.edge_index[:, test_idx]
        test_data = Data(x=pyg_data.x, edge_index=test_edge_index)
        
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]
        
        self.logger.info(f"Train edges: {len(train_idx)}, Test edges: {len(test_idx)}")
        
        return train_data, test_data, train_labels, test_labels
    
    def initialize_model(self, input_dim: int):
        """Initialize the GNN model"""
        self.model = DropoutPredictionGNN(
            input_dim=input_dim,
            hidden_dim=self.config.model.hidden_dim,
            output_dim=1,
            num_layers=self.config.model.num_layers,
            dropout_rate=self.config.model.dropout_rate,
            conv_type=self.config.model.conv_type
        )
        
        self.model.to(self.device)
        
        self.logger.info(f"Initialized {self.config.model.conv_type} model with {sum(p.numel() for p in self.model.parameters())} parameters")
        
        return self.model
    
    def train_model(self, train_data: Data, train_labels: np.ndarray, 
                   val_data: Data = None, val_labels: np.ndarray = None) -> Dict[str, list]:
        """Train the GNN model"""
        self.logger.info("Starting model training...")
        
        # Move data to device
        train_data = train_data.to(self.device)
        train_labels = torch.tensor(train_labels, dtype=torch.float32).to(self.device)
        
        if val_data is not None:
            val_data = val_data.to(self.device)
            val_labels = torch.tensor(val_labels, dtype=torch.float32).to(self.device)
        
        # Initialize optimizer and loss function
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config.model.learning_rate
        )
        criterion = nn.BCELoss()
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.model.num_epochs):
            # Training phase
            self.model.train()
            optimizer.zero_grad()
            
            # Forward pass - get node embeddings
            node_embeddings = self.model.get_embeddings(train_data.x, train_data.edge_index)
            
            # For edge prediction, we need to compute edge-level predictions
            # Get embeddings for source and target nodes of each edge
            src_embeddings = node_embeddings[train_data.edge_index[0]]
            dst_embeddings = node_embeddings[train_data.edge_index[1]]
            
            # Compute edge predictions (concatenate or dot product)
            edge_features = torch.cat([src_embeddings, dst_embeddings], dim=1)
            
            # Use a simple linear layer to predict edge labels
            if not hasattr(self, 'edge_predictor'):
                # Use the model's hidden_dim instead of node_embeddings.shape[1] to ensure consistency
                hidden_dim = self.config.model.hidden_dim
                self.edge_predictor = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid()
                ).to(self.device)
                
                # Add to optimizer
                optimizer.add_param_group({'params': self.edge_predictor.parameters()})
            
            train_out = self.edge_predictor(edge_features).squeeze()
            train_loss = criterion(train_out, train_labels)
            
            # Backward pass
            train_loss.backward()
            optimizer.step()
            
            # Calculate training accuracy
            train_pred = (train_out > 0.5).float()
            train_acc = (train_pred == train_labels).float().mean().item()
            
            history['train_loss'].append(train_loss.item())
            history['train_acc'].append(train_acc)
            
            # Validation phase
            if val_data is not None:
                self.model.eval()
                with torch.no_grad():
                    val_node_embeddings = self.model.get_embeddings(val_data.x, val_data.edge_index)
                    val_src_embeddings = val_node_embeddings[val_data.edge_index[0]]
                    val_dst_embeddings = val_node_embeddings[val_data.edge_index[1]]
                    val_edge_features = torch.cat([val_src_embeddings, val_dst_embeddings], dim=1)
                    val_out = self.edge_predictor(val_edge_features).squeeze()
                    val_loss = criterion(val_out, val_labels)
                    val_pred = (val_out > 0.5).float()
                    val_acc = (val_pred == val_labels).float().mean().item()
                    
                    history['val_loss'].append(val_loss.item())
                    history['val_acc'].append(val_acc)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if self.config.model.save_best_model:
                        self.save_model('best_model.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.model.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        self.logger.info("Training completed!")
        return history
    
    def evaluate_model(self, test_data: Data, test_labels: np.ndarray) -> Dict[str, float]:
        """Evaluate the trained model"""
        self.logger.info("Evaluating model...")
        
        self.model.eval()
        test_data = test_data.to(self.device)
        test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            # Get node embeddings and compute edge predictions
            test_node_embeddings = self.model.get_embeddings(test_data.x, test_data.edge_index)
            test_src_embeddings = test_node_embeddings[test_data.edge_index[0]]
            test_dst_embeddings = test_node_embeddings[test_data.edge_index[1]]
            test_edge_features = torch.cat([test_src_embeddings, test_dst_embeddings], dim=1)
            test_out = self.edge_predictor(test_edge_features).squeeze()
            test_pred = (test_out > 0.5).float()
            
            # Move to CPU for sklearn metrics
            test_out_cpu = test_out.cpu().numpy()
            test_pred_cpu = test_pred.cpu().numpy()
            
        # Calculate metrics
        accuracy = accuracy_score(test_labels, test_pred_cpu)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, test_pred_cpu, average='binary'
        )
        
        # ROC AUC (handle case where all labels are the same)
        try:
            roc_auc = roc_auc_score(test_labels, test_out_cpu)
        except ValueError:
            roc_auc = 0.5  # Random performance when only one class
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        self.logger.info("Evaluation Results:")
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value:.4f}")
        
        return metrics
    
    def save_model(self, filename: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        save_path = Path(self.config.model.model_save_dir) / filename
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'model_params': {
                'input_dim': self.model.input_dim,
                'hidden_dim': self.model.hidden_dim,
                'output_dim': self.model.output_dim,
                'num_layers': self.model.num_layers,
                'dropout_rate': self.model.dropout_rate
            }
        }
        
        # Save edge predictor if it exists
        if hasattr(self, 'edge_predictor'):
            save_dict['edge_predictor_state_dict'] = self.edge_predictor.state_dict()
            save_dict['edge_predictor_input_dim'] = self.model.hidden_dim * 2
        
        torch.save(save_dict, save_path)
        self.logger.info(f"Model saved to {save_path}")
    
    def load_model(self, filename: str):
        """Load a trained model"""
        load_path = Path(self.config.model.model_save_dir) / filename
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
        model_params = checkpoint['model_params']
        
        # Initialize model with saved parameters
        self.model = DropoutPredictionGNN(**model_params)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        # Load edge predictor if it exists
        if 'edge_predictor_state_dict' in checkpoint:
            # Always use the correct dimensions based on model_params
            hidden_dim = model_params['hidden_dim']
            
            self.edge_predictor = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ).to(self.device)
            
            # Only load state dict if dimensions match, otherwise reinitialize
            try:
                self.edge_predictor.load_state_dict(checkpoint['edge_predictor_state_dict'])
                self.logger.info("Edge predictor loaded from checkpoint")
            except RuntimeError as e:
                self.logger.warning(f"Edge predictor dimensions mismatch: {e}")
                self.logger.info("Reinitializing edge predictor with correct dimensions")
        
        self.logger.info(f"Model loaded from {load_path}")
        return self.model