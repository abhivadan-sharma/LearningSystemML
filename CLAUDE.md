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
python main.py --phase model     # Train GNN models (TODO)
python main.py --phase evaluation # Evaluate performance (TODO)
```

## Project Status (as of 2025-09-01)

### ✅ Working Components
- **Data Pipeline**: Stanford SNAP MOOC dataset (411,749 actions, 7,047 users, 97 targets)
- **Graph Construction**: Bipartite graphs with 14,288 nodes, 174,870 edges
- **Centrality Analysis**: Degree and betweenness centrality computed
- **Visualizations**: Graph plots and centrality distributions saved to `logs/plots/`
- **Reporting**: Centrality analysis results in `logs/results/centrality_report.csv`

### ⏳ TODO Components
- **GNN Model Training**: Framework exists in `src/models/gnn_model.py` but not integrated
- **Model Evaluation**: Framework exists in `src/evaluation/metrics.py` but not integrated

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
│   └── evaluation/             # Metrics and evaluation
├── data/
│   └── raw/                    # MOOC dataset files (auto-downloaded)
├── logs/
│   ├── plots/                  # Generated visualizations
│   └── results/                # Analysis results
└── .venv/                      # Virtual environment (uv-managed)
```

## Dataset Information
- **Source**: Stanford SNAP MOOC User Action Dataset
- **Files**: `mooc_actions.tsv`, `mooc_action_labels.tsv`, `mooc_action_features.tsv`
- **Auto-download**: Dataset automatically downloaded on first run
- **Labels**: Binary (0=continue, 1=dropout), ~1% dropout rate

## Key Bug Fixes Applied
1. **Data Merge Issue**: Fixed merge operation in `data_loader.py` to use `ACTIONID` instead of `USERID`/`TARGETID`
2. **Dependencies**: Removed problematic `dgl` package, updated all versions to working combinations
3. **Environment**: Switched to `uv` for faster, more reliable package management

## Development Notes
- **Environment**: Python 3.13, Windows-compatible
- **Package Manager**: Using `uv` for virtual environment and dependency management
- **Graph Library**: NetworkX for analysis, PyTorch Geometric ready for GNN implementation
- **Logging**: Comprehensive logging to `logs/mooc_analysis.log`

## Next Steps for Development
1. **Implement GNN Training**: Connect existing model architecture to main pipeline
2. **Add Evaluation Pipeline**: Integrate metrics framework with trained models
3. **Temporal Analysis**: Add time-based graph analysis features
4. **Hyperparameter Optimization**: Add automated model tuning

## Common Issues & Solutions
- **Missing pandas**: Ensure virtual environment is activated before running
- **Data files not found**: Run data phase first or check `data/raw/` directory
- **Import errors**: Verify all dependencies installed with `uv pip list`

## Output Locations
- **Logs**: `logs/mooc_analysis.log`
- **Visualizations**: `logs/plots/` (graph plots, centrality distributions)
- **Results**: `logs/results/centrality_report.csv`
- **Models**: `models/` (when training is implemented)

This project successfully demonstrates graph-based analysis of educational data and provides a solid foundation for GNN-based dropout prediction.