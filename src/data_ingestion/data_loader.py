import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

class MOOCDataLoader:
    """Load and preprocess MOOC dataset files"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
    
    def load_actions(self, filename: str = "mooc_actions.tsv") -> pd.DataFrame:
        """Load MOOC actions data"""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} not found. Please download the dataset first.")
        
        print(f"Loading {filename}...")
        df = pd.read_csv(filepath, sep='\t')
        
        print(f"Loaded {len(df)} actions")
        print(f"Columns: {list(df.columns)}")
        print(f"Unique users: {df['USERID'].nunique()}")
        print(f"Unique targets: {df['TARGETID'].nunique()}")
        
        return df
    
    def load_labels(self, filename: str = "mooc_action_labels.tsv") -> pd.DataFrame:
        """Load MOOC action labels"""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} not found. Please download the dataset first.")
        
        print(f"Loading {filename}...")
        df = pd.read_csv(filepath, sep='\t')
        
        print(f"Loaded {len(df)} labeled actions")
        print(f"Columns: {list(df.columns)}")
        print(f"Label distribution:")
        print(df['LABEL'].value_counts())
        
        return df
    
    def load_combined_data(self) -> pd.DataFrame:
        """Load and merge actions with labels"""
        actions = self.load_actions()
        labels = self.load_labels()
        
        # Merge on ACTIONID (the common identifier)
        combined = actions.merge(labels, on='ACTIONID', how='inner')
        
        print(f"Combined dataset: {len(combined)} records")
        return combined
    
    def get_data_summary(self) -> dict:
        """Get summary statistics of the dataset"""
        try:
            actions = self.load_actions()
            labels = self.load_labels()
            
            summary = {
                'total_actions': len(actions),
                'total_labeled_actions': len(labels),
                'unique_users': actions['USERID'].nunique(),
                'unique_targets': actions['TARGETID'].nunique(),
                'label_distribution': labels['LABEL'].value_counts().to_dict()
            }
            
            return summary
        except FileNotFoundError as e:
            return {'error': str(e)}

if __name__ == "__main__":
    loader = MOOCDataLoader()
    summary = loader.get_data_summary()
    print("Dataset Summary:", summary)