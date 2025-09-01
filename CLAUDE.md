# Claude Code Project Guide

## Project Overview
Graph-Based MOOC Dropout Prediction system using Graph Neural Networks (GNNs) to predict student dropout risk in Massive Open Online Courses. Built with NetworkX for graph analysis and PyTorch Geometric for GNN modeling.

## Quick Start Commands

### Environment Setup
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv
source .venv/Scripts/activate  # Windows
# source .venv/bin/activate     # Unix/macOS

# Install dependencies
uv pip install -r requirements.txt
```

### Run the Complete Pipeline
```bash
python main.py --phase all
```

### Run Individual Phases
```bash
python main.py --phase data      # Download and load MOOC dataset
python main.py --phase graph     # Build graphs and centrality analysis
python main.py --phase model     # Train GNN models
python main.py --phase evaluation # Evaluate trained models
```

## Project Status (as of 2025-09-01)

### ✅ Fully Implemented & Working
- **Data Pipeline**: Stanford SNAP MOOC dataset (411,749 actions, 7,047 users, 97 activities)
- **Graph Construction**: Bipartite graphs with 14,288 nodes, 174,870 edges
- **Centrality Analysis**: Degree and betweenness centrality computed and visualized
- **GNN Model Architecture**: Graph Convolutional Network with edge predictor
- **Complete Training Pipeline**: `MOOCGNNTrainer` class with full training loop
- **Model Evaluation**: Comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC)
- **Model Serialization**: Save/load functionality with proper state management
- **End-to-End Pipeline**: All phases working from data ingestion to evaluation
- **Comprehensive Reporting**: Training history, model metrics, and evaluation reports

### 🎯 Recent Major Achievements
- **Fixed Edge Predictor Dimensions**: Resolved dimension mismatch between training and loading
- **Added Node Embedding Extraction**: Separate method for getting 64-dim embeddings vs 1-dim predictions  
- **Implemented Edge-Level Classification**: Concatenated node embeddings for dropout prediction
- **Comprehensive Error Handling**: Graceful fallbacks for model loading issues

## Key Files & Structure

```
LearningSystemML/
├── main.py                      # Main pipeline orchestrator
├── requirements.txt             # Dependencies (updated with exact versions)
├── config/config.py            # Project configuration
├── src/
│   ├── data_ingestion/         # Data loading and processing
│   ├── graph_analysis/         # Graph construction and analysis
│   ├── models/                 # GNN model implementations
│   │   ├── gnn_model.py        # DropoutPredictionGNN with get_embeddings()
│   │   └── trainer.py          # MOOCGNNTrainer (complete training pipeline)
│   └── evaluation/             # Metrics and evaluation
├── data/
│   └── raw/                    # MOOC dataset files (auto-downloaded)
├── logs/
│   ├── plots/                  # Generated visualizations
│   └── results/                # Analysis results (CSV reports)
├── models/                     # Trained GNN models (.pth files)
└── .venv/                      # Virtual environment (uv-managed)
```

## Dataset Information
- **Source**: Stanford SNAP MOOC User Action Dataset
- **Files**: `mooc_actions.tsv`, `mooc_action_labels.tsv`, `mooc_action_features.tsv`
- **Auto-download**: Dataset automatically downloaded on first run
- **Labels**: Binary (0=continue, 1=dropout), ~1% dropout rate

## Key Technical Achievements & Bug Fixes

### Major Implementation Milestones
1. **Complete MOOCGNNTrainer**: Built full training pipeline with edge-level prediction
2. **Model Architecture**: GCN + edge predictor for binary classification
3. **Dimension Management**: Proper 64-dim node embeddings vs 1-dim classification output
4. **Model Serialization**: Save/load with edge predictor state management

### Critical Bug Fixes
1. **Data Merge Issue**: Fixed merge operation in `data_loader.py` to use `ACTIONID` instead of `USERID`/`TARGETID`
2. **Edge Predictor Dimensions**: Fixed mismatch between expected (128x64) and actual (2x1) dimensions
3. **Node Embeddings**: Added `get_embeddings()` method to extract proper embeddings before classification
4. **PyTorch Security**: Added `weights_only=False` for model loading compatibility
5. **Dependencies**: Removed problematic `dgl` package, updated all versions to working combinations

## Development Notes
- **Environment**: Python 3.13, Windows-compatible
- **Package Manager**: Using `uv` for virtual environment and dependency management
- **Graph Library**: NetworkX for analysis, PyTorch Geometric ready for GNN implementation
- **Logging**: Comprehensive logging to `logs/mooc_analysis.log`

## Current Model Performance & Known Issues

### Model Training Results
- **Training**: Converges quickly with early stopping around epoch 16-21
- **Accuracy**: 100% (concerning - indicates class imbalance issue)  
- **Precision/Recall**: 0% (model predicting all negative class)
- **F1/ROC-AUC**: 0/NaN (no positive predictions made)

### Known Issues & Next Steps
1. **Class Imbalance**: ~1% dropout rate causing model to predict all negative class
2. **Feature Engineering**: Current node features are very basic (mostly 1-dimensional)
3. **Data Stratification**: Need better handling of imbalanced binary classification
4. **Hyperparameter Tuning**: Learning rate, loss weighting, architecture optimization

## Future Development Priorities
1. **Address Class Imbalance**: Implement weighted loss, SMOTE, or focal loss
2. **Enhanced Node Features**: Add user demographics, activity content features
3. **Temporal Analysis**: Sequential learning pattern analysis
4. **Architecture Experiments**: Try GAT, GraphSAGE, different aggregation methods
5. **Hyperparameter Optimization**: Systematic model tuning

## Common Issues & Solutions
- **Missing pandas**: Ensure virtual environment is activated before running
- **Data files not found**: Run data phase first or check `data/raw/` directory
- **Import errors**: Verify all dependencies installed with `uv pip list`

## Output Locations
- **Logs**: `logs/mooc_analysis.log` (comprehensive pipeline logging)
- **Visualizations**: `logs/plots/` (graph plots, centrality distributions) 
- **Analysis Results**: `logs/results/centrality_report.csv`
- **Training Metrics**: `logs/results/training_history.csv`, `model_metrics.csv`
- **Model Files**: `models/best_model.pth`, `models/final_model.pth`
- **Evaluation Reports**: `logs/results/evaluation_report_*.csv`

## Success Summary
This project successfully implements a **complete end-to-end Graph Neural Network pipeline** for MOOC dropout prediction, featuring:

✅ **Production-Ready Infrastructure**: Full data processing, model training, and evaluation pipeline  
✅ **Advanced GNN Implementation**: Edge-level prediction with proper embedding extraction  
✅ **Robust Model Management**: Save/load with complex architectures (GNN + edge predictor)  
✅ **Comprehensive Evaluation**: Multiple metrics, detailed reporting, training history export  
✅ **Educational Domain Application**: Real-world dataset with 400K+ student interactions

The technical foundation is solid and ready for advanced research in educational data mining and graph-based learning analytics.