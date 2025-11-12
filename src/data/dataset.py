"""
Dataset module for Knee OA Classification
Handles data loading, preprocessing, and augmentation
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Callable, List
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')


class KneeOADataset(Dataset):
    """
    Knee Osteoarthritis Dataset for KL grading
    
    Args:
        root_dir: Path to dataset root directory
        split: 'train', 'val', or 'test'
        transform: Albumentations transform pipeline
        return_paths: If True, return image paths along with data
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        return_paths: bool = False,
        cache_images: bool = False
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.return_paths = return_paths
        self.cache_images = cache_images
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Cache for faster loading (optional)
        self.image_cache = {} if cache_images else None
        
        # Class distribution info
        self.class_counts = self._compute_class_distribution()
        
    def _load_metadata(self) -> pd.DataFrame:
        """Load and filter metadata for the specified split"""
        metadata_path = self.root_dir / f'{self.split}_metadata.csv'
        
        if metadata_path.exists():
            df = pd.read_csv(metadata_path)
        else:
            # Create metadata if not exists
            df = self._create_metadata()
            df.to_csv(metadata_path, index=False)
        
        return df
    
    def _create_metadata(self) -> pd.DataFrame:
        """Create metadata from directory structure"""
        data = []
        
        # Assume directory structure: root_dir/split/grade/image.png
        split_dir = self.root_dir / self.split
        
        for grade_dir in sorted(split_dir.glob('*')):
            if not grade_dir.is_dir():
                continue
            grade = int(grade_dir.name)
            for img_path in grade_dir.glob('*.png'):
                data.append({
                    'image_path': str(img_path),
                    'grade': grade,
                })
        
        return pd.DataFrame(data)
    
    def _compute_class_distribution(self) -> dict:
        """Compute class distribution for weighted sampling"""
        class_counts = self.metadata['grade'].value_counts().sort_index().to_dict()
        return class_counts
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for loss balancing"""
        total = sum(self.class_counts.values())
        weights = [total / (len(self.class_counts) * count) 
                   for count in self.class_counts.values()]
        return torch.tensor(weights, dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Get a sample from the dataset
        
        Returns:
            image: (C, H, W) tensor
            label: int (0-4)
            (optional) path: str
        """
        row = self.metadata.iloc[idx]
        img_path = row['image_path']
        label = row['grade']
        
        # Load image
        if self.cache_images and img_path in self.image_cache:
            image = self.image_cache[img_path]
        else:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.cache_images:
                self.image_cache[img_path] = image
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # Default: just convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        if self.return_paths:
            return image, label, img_path
        
        return image, label
    
    def get_sample_weights(self) -> List[float]:
        """Get sample weights for WeightedRandomSampler"""
        class_weights = self.get_class_weights()
        sample_weights = [class_weights[label].item() 
                          for label in self.metadata['grade']]
        return sample_weights


def get_transforms(image_size: int = 224, augment: bool = True) -> A.Compose:
    """
    Get augmentation pipeline
    
    Args:
        image_size: Target image size
        augment: Whether to apply augmentation
    
    Returns:
        Albumentations composition
    """
    
    if augment:
        # Training augmentations
        transform = A.Compose([
            # Geometric transforms
            A.Resize(image_size, image_size),
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=15,
                p=0.5
            ),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            
            # Intensity transforms
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.CLAHE(clip_limit=4.0, p=0.4),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            
            # Noise and blur
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.MotionBlur(blur_limit=5, p=0.2),
            
            # Cutout augmentation
            A.CoarseDropout(
                max_holes=8,
                max_height=int(image_size * 0.1),
                max_width=int(image_size * 0.1),
                p=0.3
            ),
            
            # Normalization (ImageNet stats)
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        # Validation/Test transforms (no augmentation)
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    return transform


def create_data_loaders(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    use_weighted_sampling: bool = True,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders
    
    Args:
        root_dir: Path to dataset root
        batch_size: Batch size
        num_workers: Number of worker processes
        image_size: Image size
        use_weighted_sampling: Use weighted random sampling for class imbalance
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Get transforms
    train_transform = get_transforms(image_size, augment=True)
    val_transform = get_transforms(image_size, augment=False)
    
    # Create datasets
    train_dataset = KneeOADataset(
        root_dir=root_dir,
        split='train',
        transform=train_transform
    )
    
    val_dataset = KneeOADataset(
        root_dir=root_dir,
        split='val',
        transform=val_transform
    )
    
    test_dataset = KneeOADataset(
        root_dir=root_dir,
        split='test',
        transform=val_transform
    )
    
    # Create samplers
    if use_weighted_sampling:
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    print(f"Class distribution (train): {train_dataset.class_counts}")
    
    return train_loader, val_loader, test_loader


def create_kfold_loaders(
    root_dir: str,
    n_splits: int = 5,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    random_state: int = 42
):
    """
    Create K-Fold cross-validation data loaders
    
    Yields:
        fold_idx, train_loader, val_loader
    """
    
    # Load full training dataset
    full_dataset = KneeOADataset(
        root_dir=root_dir,
        split='train',
        transform=None  # Will be set per fold
    )
    
    # Get labels for stratification
    labels = full_dataset.metadata['grade'].values
    indices = np.arange(len(labels))
    
    # Create stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
        print(f"\n=== Fold {fold_idx + 1}/{n_splits} ===")
        
        # Create fold datasets
        train_transform = get_transforms(image_size, augment=True)
        val_transform = get_transforms(image_size, augment=False)
        
        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_dataset, val_idx)
        
        # Set transforms
        train_subset.dataset.transform = train_transform
        val_subset.dataset.transform = val_transform
        
        # Create loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f"Train: {len(train_subset)} | Val: {len(val_subset)}")
        
        yield fold_idx, train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading
    root_dir = "data/processed"
    
    # Test single dataset
    dataset = KneeOADataset(
        root_dir=root_dir,
        split='train',
        transform=get_transforms(224, augment=True)
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Class distribution: {dataset.class_counts}")
    print(f"Class weights: {dataset.get_class_weights()}")
    
    # Test loading a sample
    image, label = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Label: {label}")
    
    # Test data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        root_dir=root_dir,
        batch_size=16,
        num_workers=2
    )
    
    # Test batch loading
    images, labels = next(iter(train_loader))
    print(f"\nBatch - Images shape: {images.shape}")
    print(f"Batch - Labels: {labels}")
