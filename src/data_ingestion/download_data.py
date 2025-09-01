import os
import requests
from pathlib import Path
from tqdm import tqdm

class MOOCDataDownloader:
    """Download MOOC dataset from Stanford SNAP"""
    
    BASE_URL = "https://snap.stanford.edu/data/"
    FILES = {
        "mooc_actions.tsv": "act-mooc.tar.gz",
        "mooc_action_labels.tsv": "act-mooc.tar.gz"
    }
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, filename: str, url: str) -> None:
        """Download a file with progress bar"""
        filepath = self.data_dir / filename
        
        if filepath.exists():
            print(f"{filename} already exists, skipping...")
            return
        
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    def download_all(self) -> None:
        """Download all required MOOC dataset files"""
        for filename, archive in self.FILES.items():
            url = f"{self.BASE_URL}{archive}"
            self.download_file(archive, url)
        
        print("Download complete. Extract the files manually to get the .tsv files.")

if __name__ == "__main__":
    downloader = MOOCDataDownloader()
    downloader.download_all()