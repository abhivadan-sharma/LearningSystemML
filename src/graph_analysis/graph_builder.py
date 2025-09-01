import networkx as nx
import pandas as pd
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

class MOOCGraphBuilder:
    """Build and analyze graphs from MOOC data"""
    
    def __init__(self):
        self.graph = None
        self.user_nodes = set()
        self.target_nodes = set()
    
    def build_bipartite_graph(self, df: pd.DataFrame) -> nx.Graph:
        """Build bipartite graph with users and targets as different node types"""
        G = nx.Graph()
        
        # Add user nodes
        users = df['USERID'].unique()
        G.add_nodes_from([f"user_{u}" for u in users], bipartite=0, node_type='user')
        self.user_nodes = set([f"user_{u}" for u in users])
        
        # Add target nodes
        targets = df['TARGETID'].unique()
        G.add_nodes_from([f"target_{t}" for t in targets], bipartite=1, node_type='target')
        self.target_nodes = set([f"target_{t}" for t in targets])
        
        # Add edges
        for _, row in df.iterrows():
            user_node = f"user_{row['USERID']}"
            target_node = f"target_{row['TARGETID']}"
            
            # Add edge attributes if available
            edge_attrs = {}
            if 'TIMESTAMP' in df.columns:
                edge_attrs['timestamp'] = row['TIMESTAMP']
            if 'LABEL' in df.columns:
                edge_attrs['label'] = row['LABEL']
            
            G.add_edge(user_node, target_node, **edge_attrs)
        
        self.graph = G
        print(f"Built bipartite graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def build_directed_graph(self, df: pd.DataFrame) -> nx.DiGraph:
        """Build directed graph for sequential analysis"""
        G = nx.DiGraph()
        
        # Sort by user and timestamp for sequential edges
        if 'TIMESTAMP' in df.columns:
            df_sorted = df.sort_values(['USERID', 'TIMESTAMP'])
        else:
            df_sorted = df.sort_values(['USERID'])
        
        # Group by user to create sequential paths
        for user_id, user_data in df_sorted.groupby('USERID'):
            targets = user_data['TARGETID'].tolist()
            
            # Create sequential edges between targets for this user
            for i in range(len(targets) - 1):
                current_target = f"target_{targets[i]}"
                next_target = f"target_{targets[i+1]}"
                
                edge_attrs = {'user_id': user_id}
                if 'TIMESTAMP' in df.columns:
                    edge_attrs['timestamp'] = user_data.iloc[i]['TIMESTAMP']
                if 'LABEL' in df.columns:
                    edge_attrs['label'] = user_data.iloc[i]['LABEL']
                
                if G.has_edge(current_target, next_target):
                    G[current_target][next_target]['weight'] = G[current_target][next_target].get('weight', 0) + 1
                else:
                    G.add_edge(current_target, next_target, weight=1, **edge_attrs)
        
        self.graph = G
        print(f"Built directed graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def get_graph_statistics(self) -> Dict:
        """Get basic graph statistics"""
        if self.graph is None:
            return {"error": "No graph built yet"}
        
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'is_directed': isinstance(self.graph, nx.DiGraph),
            'is_connected': nx.is_connected(self.graph) if not isinstance(self.graph, nx.DiGraph) else nx.is_weakly_connected(self.graph)
        }
        
        if hasattr(self, 'user_nodes') and hasattr(self, 'target_nodes'):
            stats['num_users'] = len(self.user_nodes)
            stats['num_targets'] = len(self.target_nodes)
        
        return stats
    
    def visualize_subgraph(self, node_limit: int = 50, save_path: Optional[str] = None):
        """Visualize a subgraph for exploration"""
        if self.graph is None:
            print("No graph to visualize. Build graph first.")
            return
        
        # Take subgraph of most connected nodes
        nodes = list(self.graph.nodes())[:node_limit]
        subgraph = self.graph.subgraph(nodes)
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(subgraph, k=1, iterations=50)
        
        # Color nodes by type if bipartite
        if hasattr(self, 'user_nodes') and hasattr(self, 'target_nodes'):
            user_nodes_in_sub = [n for n in nodes if n in self.user_nodes]
            target_nodes_in_sub = [n for n in nodes if n in self.target_nodes]
            
            nx.draw_networkx_nodes(subgraph, pos, nodelist=user_nodes_in_sub, 
                                 node_color='lightblue', label='Users', node_size=100)
            nx.draw_networkx_nodes(subgraph, pos, nodelist=target_nodes_in_sub, 
                                 node_color='lightcoral', label='Targets', node_size=100)
        else:
            nx.draw_networkx_nodes(subgraph, pos, node_color='lightblue', node_size=100)
        
        nx.draw_networkx_edges(subgraph, pos, alpha=0.5, width=0.5)
        
        plt.title(f"MOOC Graph Subgraph ({len(nodes)} nodes)")
        plt.legend()
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graph saved to {save_path}")
        else:
            plt.show()

if __name__ == "__main__":
    # Example usage
    from ..data_ingestion.data_loader import MOOCDataLoader
    
    loader = MOOCDataLoader()
    try:
        df = loader.load_combined_data()
        
        builder = MOOCGraphBuilder()
        graph = builder.build_bipartite_graph(df)
        
        stats = builder.get_graph_statistics()
        print("Graph Statistics:", stats)
        
    except FileNotFoundError:
        print("Dataset not found. Please download the dataset first.")