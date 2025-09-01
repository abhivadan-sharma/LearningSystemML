#!/usr/bin/env python3
"""
MOOC Dropout Prediction - Main Entry Point

This script orchestrates the complete pipeline for MOOC dropout prediction:
1. Data ingestion and preprocessing
2. Graph construction and analysis
3. GNN model training
4. Evaluation and results

Usage:
    python main.py --phase all
    python main.py --phase data
    python main.py --phase graph
    python main.py --phase model
"""

import argparse
import logging
from pathlib import Path
import sys

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from config.config import get_config
from src.data_ingestion.data_loader import MOOCDataLoader
from src.data_ingestion.download_data import MOOCDataDownloader
from src.graph_analysis.graph_builder import MOOCGraphBuilder
from src.graph_analysis.centrality_analysis import CentralityAnalyzer

def setup_logging(config):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(config.log_dir) / "mooc_analysis.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def run_data_phase(config, logger):
    """Run data ingestion and preprocessing phase"""
    logger.info("Starting data phase...")
    
    # Download data
    downloader = MOOCDataDownloader(config.data.raw_data_dir)
    downloader.download_all()
    
    # Load and summarize data
    loader = MOOCDataLoader(config.data.raw_data_dir)
    summary = loader.get_data_summary()
    logger.info(f"Data summary: {summary}")
    
    return True

def run_graph_phase(config, logger):
    """Run graph construction and analysis phase"""
    logger.info("Starting graph analysis phase...")
    
    # Load data
    loader = MOOCDataLoader(config.data.raw_data_dir)
    try:
        df = loader.load_combined_data()
    except FileNotFoundError:
        logger.error("Data files not found. Please run data phase first.")
        return False
    
    # Build graph
    builder = MOOCGraphBuilder()
    
    if config.graph.graph_type == "bipartite":
        graph = builder.build_bipartite_graph(df)
    elif config.graph.graph_type == "directed":
        graph = builder.build_directed_graph(df)
    else:
        logger.error(f"Unsupported graph type: {config.graph.graph_type}")
        return False
    
    # Graph statistics
    stats = builder.get_graph_statistics()
    logger.info(f"Graph statistics: {stats}")
    
    # Visualize subgraph
    if config.graph.save_graph_plots:
        plot_path = Path(config.graph.plot_dir) / "graph_visualization.png"
        builder.visualize_subgraph(
            node_limit=config.graph.visualization_node_limit,
            save_path=str(plot_path)
        )
    
    # Centrality analysis
    analyzer = CentralityAnalyzer(graph)
    centrality_results = analyzer.analyze_all_centralities(top_k=10)
    
    # Create centrality report
    report = analyzer.create_centrality_report(centrality_results)
    report_path = Path(config.evaluation.results_dir) / "centrality_report.csv"
    report.to_csv(report_path, index=False)
    logger.info(f"Centrality report saved to {report_path}")
    
    # Plot centrality distributions
    if config.graph.save_graph_plots:
        for measure, data in centrality_results.items():
            plot_path = Path(config.graph.plot_dir) / f"{measure}_distribution.png"
            analyzer.plot_centrality_distribution(
                data['scores'], measure, str(plot_path)
            )
    
    return True

def run_model_phase(config, logger):
    """Run GNN model training phase"""
    logger.info("Starting model training phase...")
    
    try:
        from src.models.trainer import MOOCGNNTrainer
        
        # Load data and graph
        loader = MOOCDataLoader(config.data.raw_data_dir)
        df = loader.load_combined_data()
        
        # Build graph
        builder = MOOCGraphBuilder()
        if config.graph.graph_type == "bipartite":
            graph = builder.build_bipartite_graph(df)
        elif config.graph.graph_type == "directed":
            graph = builder.build_directed_graph(df)
        else:
            logger.error(f"Unsupported graph type: {config.graph.graph_type}")
            return False
        
        # Initialize trainer
        trainer = MOOCGNNTrainer(config)
        
        # Prepare graph data for PyTorch Geometric
        pyg_data, labels = trainer.prepare_graph_data(graph, df)
        
        # Create train/test split
        train_data, test_data, train_labels, test_labels = trainer.create_train_test_split(
            pyg_data, labels
        )
        
        # Further split training data for validation
        train_data_final, val_data, train_labels_final, val_labels = trainer.create_train_test_split(
            train_data, train_labels
        )
        
        # Initialize model
        input_dim = pyg_data.x.shape[1]
        model = trainer.initialize_model(input_dim)
        
        # Train model
        history = trainer.train_model(
            train_data_final, train_labels_final,
            val_data, val_labels
        )
        
        # Save final model
        trainer.save_model('final_model.pth')
        
        # Quick evaluation on test set
        test_metrics = trainer.evaluate_model(test_data, test_labels)
        
        # Save training history and metrics
        import pandas as pd
        from pathlib import Path
        
        history_df = pd.DataFrame(history)
        history_path = Path(config.evaluation.results_dir) / "training_history.csv"
        history_df.to_csv(history_path, index=False)
        
        metrics_df = pd.DataFrame([test_metrics])
        metrics_path = Path(config.evaluation.results_dir) / "model_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        
        logger.info(f"Model training completed successfully!")
        logger.info(f"Final test accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Final test F1-score: {test_metrics['f1_score']:.4f}")
        logger.info(f"Training history saved to: {history_path}")
        logger.info(f"Model metrics saved to: {metrics_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_evaluation_phase(config, logger):
    """Run model evaluation phase"""
    logger.info("Starting evaluation phase...")
    
    try:
        from src.models.trainer import MOOCGNNTrainer
        from src.evaluation.metrics import ModelEvaluator
        from pathlib import Path
        
        # Check if model exists
        model_path = Path(config.model.model_save_dir) / 'best_model.pth'
        if not model_path.exists():
            model_path = Path(config.model.model_save_dir) / 'final_model.pth'
            if not model_path.exists():
                logger.warning("No trained model found. Please run model training first.")
                return True
        
        # Load data and graph (same as training)
        loader = MOOCDataLoader(config.data.raw_data_dir)
        df = loader.load_combined_data()
        
        builder = MOOCGraphBuilder()
        if config.graph.graph_type == "bipartite":
            graph = builder.build_bipartite_graph(df)
        elif config.graph.graph_type == "directed":
            graph = builder.build_directed_graph(df)
        else:
            logger.error(f"Unsupported graph type: {config.graph.graph_type}")
            return False
        
        # Initialize trainer and load model
        trainer = MOOCGNNTrainer(config)
        trainer.load_model(model_path.name)
        
        # Prepare data
        pyg_data, labels = trainer.prepare_graph_data(graph, df)
        _, test_data, _, test_labels = trainer.create_train_test_split(pyg_data, labels)
        
        # Evaluate model
        metrics = trainer.evaluate_model(test_data, test_labels)
        
        # Create detailed evaluation report
        evaluator = ModelEvaluator(config)
        evaluator.create_evaluation_report(metrics, test_labels, model_path.name)
        
        logger.info("Model evaluation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return True  # Don't fail the pipeline for evaluation issues

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="MOOC Dropout Prediction Pipeline")
    parser.add_argument(
        "--phase",
        choices=["all", "data", "graph", "model", "evaluation"],
        default="all",
        help="Which phase to run"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config()
    logger = setup_logging(config)
    
    logger.info(f"Starting MOOC Dropout Prediction Pipeline")
    logger.info(f"Project: {config.project_name} v{config.version}")
    logger.info(f"Running phase: {args.phase}")
    
    success = True
    
    # Run requested phases
    if args.phase in ["all", "data"]:
        success &= run_data_phase(config, logger)
        if not success:
            logger.error("Data phase failed")
            return 1
    
    if args.phase in ["all", "graph"]:
        success &= run_graph_phase(config, logger)
        if not success:
            logger.error("Graph phase failed")
            return 1
    
    if args.phase in ["all", "model"]:
        success &= run_model_phase(config, logger)
        if not success:
            logger.error("Model phase failed")
            return 1
    
    if args.phase in ["all", "evaluation"]:
        success &= run_evaluation_phase(config, logger)
        if not success:
            logger.error("Evaluation phase failed")
            return 1
    
    logger.info("Pipeline completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())