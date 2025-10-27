import os
import pandas as pd
from PIL import Image
from glob import glob
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# --- Augmentation Transforms ---
# From notebook cell [9]
# These are applied ON-THE-FLY (in memory)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- DataFrame Creation ---
def create_dataframe(data_dir):
    """Scans the data_dir for class subfolders and creates a DataFrame."""
    print(f"[INFO] Creating initial DataFrame from {data_dir}...")
    filepaths = []
    labels = []
    
    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for class_label in class_dirs:
        class_path = os.path.join(data_dir, class_label)
        image_paths = glob(f"{class_path}/*.jpg") + glob(f"{class_path}/*.png")
        
        for img_path in image_paths:
            filepaths.append(img_path)
            labels.append(class_label)
            
    df = pd.DataFrame({'filepaths': filepaths, 'labels': labels})
    print("Original counts:\n", df['labels'].value_counts())
    return df

# --- Class Balancing Functions (from notebook) ---
def trim(df, max_samples, min_samples, column):
    """Trims over-represented classes."""
    print(f"[INFO] Trimming classes to max={max_samples}, min={min_samples}...")
    df = df.copy()
    groups = df.groupby(column)    
    trimmed_df_list = []
    
    for label in df[column].unique(): 
        group = groups.get_group(label)
        count = len(group)    
        if count > max_samples:
            sampled_group = group.sample(n=max_samples, random_state=123, axis=0)
            trimmed_df_list.append(sampled_group)
        elif count >= min_samples:
            sampled_group = group        
            trimmed_df_list.append(sampled_group)
            
    trimmed_df = pd.concat(trimmed_df_list, axis=0)
    print("Counts after trimming:\n", trimmed_df['labels'].value_counts())
    return trimmed_df

def generate_balanced_dataframe(df, n, random_state=123):
    """
    Balances the DataFrame by duplicating rows (paths) of under-represented
    classes to be augmented on-the-fly.
    """
    print(f"[INFO] Balancing DataFrame to {n} samples per class...")
    df = df.copy()
    groups = df.groupby('labels')
    balanced_df_list = []

    for label in df['labels'].unique():
        group = groups.get_group(label)
        count = len(group)
        if count < n:
            # Sample with replacement to duplicate rows
            sampled_group = group.sample(n=n, replace=True, random_state=random_state)
            balanced_df_list.append(sampled_group)
        else:
            # Group is already at or above target, no change
            balanced_df_list.append(group)
            
    balanced_df = pd.concat(balanced_df_list, axis=0)
    print("Final balanced DataFrame counts:\n", balanced_df['labels'].value_counts())
    return balanced_df

# --- Custom Dataset (from notebook cell [9]) ---
class KneeDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        # Ensure labels are integers for the loss function
        self.labels = df['labels'].astype(int).values
        self.filepaths = df['filepaths'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        label = self.labels[idx]
        
        # Load image as PIL
        img = Image.open(img_path).convert("RGB")
        
        # Apply on-the-fly transforms
        if self.transform:
            img = self.transform(img)
            
        return img, label