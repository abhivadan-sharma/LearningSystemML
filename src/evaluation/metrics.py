import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, roc_curve,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

class ModelEvaluator:
    """Evaluate dropout prediction models"""
    
    def __init__(self):
        self.results = {}
    
    def compute_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                     y_pred_proba: np.ndarray = None, 
                                     model_name: str = "model") -> Dict:
        """Compute comprehensive classification metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
        
        # Class-specific metrics
        metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None)
        metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None)
        metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None)
        
        # AUC metrics (if probabilities provided)
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except ValueError:
                metrics['roc_auc'] = None
        
        # Store results
        self.results[model_name] = {
            'metrics': metrics,
            'predictions': y_pred,
            'true_labels': y_true,
            'probabilities': y_pred_proba
        }
        
        return metrics
    
    def print_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  model_name: str = "Model"):
        """Print detailed classification report"""
        print(f"\n{model_name} Classification Report")
        print("=" * 50)
        print(classification_report(y_true, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                       model_name: str = "Model", save_path: str = None):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        else:
            plt.show()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   model_name: str = "Model", save_path: str = None):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label=f'{model_name}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add baseline (proportion of positive class)
        baseline = np.sum(y_true) / len(y_true)
        plt.axhline(y=baseline, color='k', linestyle='--', label=f'Baseline ({baseline:.3f})')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PR curve saved to {save_path}")
        else:
            plt.show()
    
    def compare_models(self, model_results: Dict) -> pd.DataFrame:
        """Compare multiple models"""
        comparison_data = []
        
        for model_name, results in model_results.items():
            metrics = results['metrics']
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'ROC-AUC': metrics.get('roc_auc', 'N/A')
            })
        
        df = pd.DataFrame(comparison_data)
        return df.round(4)
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame, save_path: str = None):
        """Plot model comparison"""
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            bars = ax.bar(comparison_df['Model'], comparison_df[metric])
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        else:
            plt.show()
    
    def analyze_predictions_by_user_activity(self, predictions: np.ndarray, 
                                           user_ids: List, target_ids: List,
                                           true_labels: np.ndarray) -> pd.DataFrame:
        """Analyze prediction performance by user activity patterns"""
        analysis_data = []
        
        for i, (user_id, target_id, pred, true_label) in enumerate(
            zip(user_ids, target_ids, predictions, true_labels)
        ):
            analysis_data.append({
                'user_id': user_id,
                'target_id': target_id,
                'prediction': pred,
                'true_label': true_label,
                'correct': pred == true_label
            })
        
        df = pd.DataFrame(analysis_data)
        
        # Summary statistics
        print("Prediction Analysis Summary:")
        print(f"Overall Accuracy: {df['correct'].mean():.4f}")
        print(f"Predictions by User (top 10):")
        user_accuracy = df.groupby('user_id')['correct'].agg(['count', 'mean']).sort_values('count', ascending=False)
        print(user_accuracy.head(10))
        
        print(f"\nPredictions by Target (top 10):")
        target_accuracy = df.groupby('target_id')['correct'].agg(['count', 'mean']).sort_values('count', ascending=False)
        print(target_accuracy.head(10))
        
        return df

if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator()
    
    # Simulate some predictions
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_pred = np.random.randint(0, 2, 1000)
    y_pred_proba = np.random.random(1000)
    
    # Compute metrics
    metrics = evaluator.compute_classification_metrics(y_true, y_pred, y_pred_proba, "Test Model")
    print("Computed metrics:", metrics)