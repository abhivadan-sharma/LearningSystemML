# Graph-Based MOOC Dropout Prediction

A machine learning project that uses Graph Neural Networks (GNNs) to predict student dropout risk in Massive Open Online Courses (MOOCs). This project analyzes learner behavior patterns using graph-based approaches to identify at-risk students early.

## Project Overview

- **Objective**: Build a graph-based model to predict which learners are at risk of dropping out of MOOC courses
- **Dataset**: Stanford SNAP MOOC User Action Dataset
- **Approach**: Two-phase analysis using NetworkX for exploratory graph analysis and PyTorch Geometric for GNN-based prediction
- **Business Impact**: Enable early intervention strategies to improve course completion rates

## Project Structure

```
LearningSystemML/
├── data/
│   ├── raw/                    # Raw dataset files
│   └── processed/              # Processed data for modeling
├── src/
│   ├── data_ingestion/         # Data download and loading utilities
│   │   ├── download_data.py    # Download MOOC dataset from Stanford SNAP
│   │   └── data_loader.py      # Load and preprocess data
│   ├── graph_analysis/         # Graph construction and analysis
│   │   ├── graph_builder.py    # Build bipartite and directed graphs
│   │   └── centrality_analysis.py # Centrality measures and analysis
│   ├── models/                 # GNN model implementations
│   │   └── gnn_model.py        # PyTorch Geometric GNN models
│   └── evaluation/             # Model evaluation and metrics
│       └── metrics.py          # Evaluation metrics and visualization
├── notebooks/                  # Jupyter notebooks for exploration
├── config/
│   └── config.py              # Project configuration
├── logs/                      # Logs and outputs
│   ├── plots/                 # Generated visualizations
│   └── results/               # Analysis results
├── requirements.txt           # Python dependencies
├── main.py                    # Main pipeline script
└── README.md                  # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd LearningSystemML
```

2. Install uv (if not already installed):
```bash
# On Unix/macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

3. Create virtual environment and install dependencies:
```bash
uv venv
source .venv/Scripts/activate  # On Unix: source .venv/bin/activate
uv pip install -r requirements.txt
```

### Alternative: Traditional pip installation
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Complete Pipeline

Run the entire analysis pipeline:
```bash
python main.py --phase all
```

### Individual Phases

Run specific phases of the analysis:

1. **Data Phase**: Download and preprocess the MOOC dataset
```bash
python main.py --phase data
```

2. **Graph Phase**: Build graphs and perform centrality analysis
```bash
python main.py --phase graph
```

3. **Model Phase**: Train GNN models (TODO)
```bash
python main.py --phase model
```

4. **Evaluation Phase**: Evaluate model performance (TODO)
```bash
python main.py --phase evaluation
```

### Interactive Analysis

Use Jupyter notebooks for interactive exploration:
```bash
jupyter notebook
```

## Dataset

The project uses the Stanford SNAP MOOC User Action Dataset:
- **Nodes**: Users and Course Activities ("targets")
- **Edges**: Actions users take on course activities
- **Labels**: Binary indicators of student dropout after specific actions
- **Source**: https://snap.stanford.edu/data/act-mooc.html

## Technical Approach

### Phase 1: Exploratory Graph Analysis (NetworkX)
- **Graph Construction**: Build bipartite and directed graphs from user-activity interactions
- **Pathfinding**: Identify common learning paths and bottlenecks
- **Centrality Analysis**: Use degree, betweenness, and other centrality measures to find influential activities
- **Visualization**: Generate graph visualizations for understanding user-activity relationships

### Phase 2: GNN-Based Prediction (PyTorch Geometric)
- **Problem Formulation**: Link prediction to estimate dropout likelihood
- **Model Architecture**: Graph Convolutional Networks (GCN) and GraphSAGE
- **Features**: Node embeddings capturing both content and network position
- **Training**: Supervised learning using dropout labels

## Key Features

- **Modular Architecture**: Separate modules for data processing, graph analysis, modeling, and evaluation
- **Configurable**: Centralized configuration system for easy parameter tuning
- **Visualization**: Comprehensive plotting capabilities for graphs and model results
- **Logging**: Detailed logging for debugging and progress tracking
- **Extensible**: Easy to add new graph analysis methods or model architectures

## Configuration

Modify `config/config.py` to customize:
- Data processing parameters
- Graph construction options
- Model hyperparameters
- Evaluation metrics
- Output directories

## Results

The analysis generates several outputs:
- **Graph Statistics**: Node/edge counts, connectivity metrics
- **Centrality Reports**: Most influential activities and users
- **Visualizations**: Graph plots and centrality distributions
- **Model Metrics**: Classification performance (accuracy, precision, recall, F1, AUC)

## Dependencies

Key packages used:
- **NetworkX**: Graph construction and analysis
- **PyTorch Geometric**: Graph Neural Networks
- **Pandas/NumPy**: Data manipulation
- **Scikit-learn**: Evaluation metrics
- **Matplotlib/Seaborn**: Visualization
- **Jupyter**: Interactive analysis

See `requirements.txt` for complete dependency list.

## Development Status

- ✅ Project structure and configuration
- ✅ Data ingestion and loading (Stanford SNAP MOOC dataset)
- ✅ Graph construction (bipartite and directed graphs)
- ✅ Centrality analysis (degree and betweenness centrality)
- ✅ Graph visualization and reporting
- ✅ End-to-end data pipeline working
- ✅ GNN model architecture (PyTorch Geometric ready)
- ✅ Evaluation metrics framework
- ⏳ Model training pipeline (TODO - framework ready)
- ⏳ End-to-end GNN evaluation (TODO - framework ready)

## Future Enhancements

- [ ] Implement temporal graph analysis for sequential learning patterns
- [ ] Add more sophisticated node feature engineering
- [ ] Experiment with different GNN architectures (GAT, GraphSAINT)
- [ ] Implement hyperparameter optimization
- [ ] Add model interpretability features
- [ ] Create web interface for predictions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational and research purposes.

## Acknowledgments

- Stanford SNAP for providing the MOOC dataset
- PyTorch Geometric team for the excellent graph deep learning library
- NetworkX developers for graph analysis tools