import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class DataConfig:
    """Configuration for data processing"""
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    dataset_urls: Dict[str, str] = None
    
    def __post_init__(self):
        if self.dataset_urls is None:
            self.dataset_urls = {
                "mooc_actions": "https://snap.stanford.edu/data/act-mooc.tar.gz",
                "mooc_labels": "https://snap.stanford.edu/data/act-mooc.tar.gz"
            }

@dataclass
class GraphConfig:
    """Configuration for graph analysis"""
    graph_type: str = "bipartite"  # bipartite, directed, undirected
    visualization_node_limit: int = 50
    centrality_k_samples: int = 1000  # For approximate betweenness centrality
    save_graph_plots: bool = True
    plot_dir: str = "logs/plots"

@dataclass
class ModelConfig:
    """Configuration for GNN models"""
    # Model architecture
    hidden_dim: int = 64
    num_layers: int = 2
    dropout_rate: float = 0.5
    conv_type: str = "GCN"  # GCN, SAGE
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    early_stopping_patience: int = 10
    
    # Feature engineering
    node_feature_type: str = "degree"  # degree, centrality, onehot
    
    # Model saving
    model_save_dir: str = "models/"
    save_best_model: bool = True

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    
    # Metrics to compute
    compute_roc_auc: bool = True
    compute_pr_curve: bool = True
    save_plots: bool = True
    
    # Output directories
    results_dir: str = "logs/results"
    plots_dir: str = "logs/plots"

@dataclass
class ProjectConfig:
    """Main project configuration"""
    project_name: str = "MOOC Dropout Prediction"
    version: str = "1.0.0"
    
    # Sub-configurations
    data: DataConfig = None
    graph: GraphConfig = None
    model: ModelConfig = None
    evaluation: EvaluationConfig = None
    
    # Logging
    log_level: str = "INFO"
    log_dir: str = "logs/"
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.graph is None:
            self.graph = GraphConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
        
        # Create necessary directories
        self.create_directories()
    
    def create_directories(self):
        """Create all necessary directories"""
        dirs_to_create = [
            self.data.raw_data_dir,
            self.data.processed_data_dir,
            self.graph.plot_dir,
            self.model.model_save_dir,
            self.evaluation.results_dir,
            self.evaluation.plots_dir,
            self.log_dir,
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

# Default configuration instance
DEFAULT_CONFIG = ProjectConfig()

def load_config_from_file(config_path: str) -> ProjectConfig:
    """Load configuration from file (future enhancement)"""
    # TODO: Implement loading from YAML/JSON file
    return DEFAULT_CONFIG

def get_config() -> ProjectConfig:
    """Get project configuration"""
    config_path = os.environ.get("MOOC_CONFIG_PATH")
    
    if config_path and os.path.exists(config_path):
        return load_config_from_file(config_path)
    else:
        return DEFAULT_CONFIG

if __name__ == "__main__":
    config = get_config()
    print(f"Project: {config.project_name} v{config.version}")
    print(f"Data directory: {config.data.raw_data_dir}")
    print(f"Model hidden dim: {config.model.hidden_dim}")
    print(f"Graph type: {config.graph.graph_type}")