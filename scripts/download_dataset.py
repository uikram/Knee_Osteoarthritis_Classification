"""
Script to download knee OA dataset from Mendeley (requires manual download link)
"""
import os
from pathlib import Path

def download_dataset():
    print("Download the dataset manually from Mendeley Data:")
    print("DOI: 10.17632/56rmx5bjcr.1 or https://data.mendeley.com/datasets/56rmx5bjcr/1")
    print("Place the extracted folders into ./data/raw/")

if __name__ == "__main__":
    download_dataset()
