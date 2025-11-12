"""
Preprocess raw data into standardized format
"""
import os
from pathlib import Path
import pandas as pd

DATA_DIR = Path('data')
SPLITS = ['train', 'val', 'test']

for split in SPLITS:
    split_dir = DATA_DIR / split
    meta_list = []
    for grade in range(5):  # Only 0–4
        grade_dir = split_dir / str(grade)
        if grade_dir.exists():
            for img_path in grade_dir.glob('*.png'):
                meta_list.append({
                    'image_path': str(img_path.resolve()),
                    'grade': grade,
                    'split': split,
                    'filename': img_path.name
                })
    df = pd.DataFrame(meta_list)
    outdir = DATA_DIR / 'processed'
    outdir.mkdir(exist_ok=True)
    df.to_csv(outdir / f'{split}_metadata.csv', index=False)
    print(f"Saved {len(df)} samples for {split} to {outdir / f'{split}_metadata.csv'}")