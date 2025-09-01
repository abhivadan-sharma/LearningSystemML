import networkx as nx
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

class CentralityAnalyzer:
    """Analyze centrality measures in MOOC graphs"""
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
    
    def compute_degree_centrality(self) -> Dict:
        """Compute degree centrality for all nodes"""
        return nx.degree_centrality(self.graph)
    
    def compute_betweenness_centrality(self, k: int = None) -> Dict:
        """Compute betweenness centrality (approximate for large graphs)"""
        return nx.betweenness_centrality(self.graph, k=k)
    
    def compute_closeness_centrality(self) -> Dict:
        """Compute closeness centrality"""
        if nx.is_connected(self.graph):
            return nx.closeness_centrality(self.graph)
        else:
            # For disconnected graphs
            return nx.closeness_centrality(self.graph)
    
    def compute_eigenvector_centrality(self, max_iter: int = 1000) -> Dict:
        """Compute eigenvector centrality"""
        try:
            return nx.eigenvector_centrality(self.graph, max_iter=max_iter)
        except nx.PowerIterationFailedConvergence:
            print("Eigenvector centrality failed to converge")
            return {}
    
    def get_top_central_nodes(self, centrality_scores: Dict, top_k: int = 10) -> List[Tuple]:
        """Get top-k nodes by centrality score"""
        sorted_nodes = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:top_k]
    
    def analyze_all_centralities(self, top_k: int = 10) -> Dict:
        """Compute all centrality measures and return top nodes"""
        print("Computing centrality measures...")
        
        results = {}
        
        # Degree centrality
        print("- Degree centrality")
        degree_cent = self.compute_degree_centrality()
        results['degree'] = {
            'scores': degree_cent,
            'top_nodes': self.get_top_central_nodes(degree_cent, top_k)
        }
        
        # Betweenness centrality (approximate for large graphs)
        print("- Betweenness centrality")
        between_cent = self.compute_betweenness_centrality(k=min(1000, len(self.graph.nodes())))
        results['betweenness'] = {
            'scores': between_cent,
            'top_nodes': self.get_top_central_nodes(between_cent, top_k)
        }
        
        # Closeness centrality
        if len(self.graph.nodes()) < 10000:  # Only for smaller graphs
            print("- Closeness centrality")
            close_cent = self.compute_closeness_centrality()
            results['closeness'] = {
                'scores': close_cent,
                'top_nodes': self.get_top_central_nodes(close_cent, top_k)
            }
        
        # Eigenvector centrality
        if len(self.graph.nodes()) < 5000:  # Only for smaller graphs
            print("- Eigenvector centrality")
            eigen_cent = self.compute_eigenvector_centrality()
            if eigen_cent:
                results['eigenvector'] = {
                    'scores': eigen_cent,
                    'top_nodes': self.get_top_central_nodes(eigen_cent, top_k)
                }
        
        return results
    
    def create_centrality_report(self, results: Dict) -> pd.DataFrame:
        """Create a summary report of centrality analysis"""
        report_data = []
        
        for measure, data in results.items():
            for i, (node, score) in enumerate(data['top_nodes']):
                report_data.append({
                    'rank': i + 1,
                    'measure': measure,
                    'node': node,
                    'score': score,
                    'node_type': 'user' if 'user_' in node else 'target'
                })
        
        return pd.DataFrame(report_data)
    
    def plot_centrality_distribution(self, centrality_scores: Dict, measure_name: str, 
                                   save_path: str = None):
        """Plot distribution of centrality scores"""
        scores = list(centrality_scores.values())
        
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel(f'{measure_name.title()} Centrality Score')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {measure_name.title()} Centrality Scores')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_score = sum(scores) / len(scores)
        plt.axvline(mean_score, color='red', linestyle='--', label=f'Mean: {mean_score:.4f}')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def identify_influential_activities(self, centrality_results: Dict) -> Dict:
        """Identify most influential activities (targets) based on centrality"""
        influential_activities = {}
        
        for measure, data in centrality_results.items():
            target_nodes = [(node, score) for node, score in data['top_nodes'] 
                          if 'target_' in node]
            influential_activities[measure] = target_nodes[:5]  # Top 5 targets
        
        return influential_activities

if __name__ == "__main__":
    # Example usage would go here
    print("Centrality analysis module loaded. Use with a built graph.")