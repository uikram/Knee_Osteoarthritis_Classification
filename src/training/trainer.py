"""
Training module for Knee OA Classification
Handles model training, validation, and checkpointing
"""

import os
import time
import copy
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from ..evaluation.metrics import compute_metrics, mean_absolute_error
from ..utils.helpers import AverageMeter, EarlyStopping, save_checkpoint, load_checkpoint


class Trainer:
    """
    Trainer class for model training and evaluation
    
    Args:
        model: PyTorch model
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        config: Training configuration
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[object] = None,
        device: torch.device = torch.device('cuda'),
        config: Optional[Dict] = None
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config or {}
        
        # Training settings
        self.epochs = self.config.get('epochs', 100)
        self.use_amp = self.config.get('use_amp', True)
        self.gradient_clip = self.config.get('gradient_clip', 1.0)
        self.log_interval = self.config.get('log_interval', 10)
        
        # Mixed precision training
        self.scaler = GradScaler() if self.use_amp else None
        
        # Early stopping
        patience = self.config.get('early_stopping_patience', 15)
        self.early_stopping = EarlyStopping(patience=patience, mode='min')
        
        # Checkpoint directory
        self.checkpoint_dir = Path(self.config.get('checkpoint_dir', 'models/checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # History tracking
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_mae': [],
            'learning_rate': []
        }
        
        # Wandb logging
        self.use_wandb = self.config.get('use_wandb', False)
        if self.use_wandb:
            wandb.watch(self.model, log='all', log_freq=100)
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        pbar = tqdm(train_loader, desc='Training')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            batch_size = images.size(0)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                self.optimizer.step()
            
            # Compute accuracy
            _, preds = torch.max(outputs, 1)
            acc = (preds == labels).float().mean()
            
            # Update meters
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc.item(), batch_size)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{acc_meter.avg:.4f}'
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % self.log_interval == 0:
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/batch_acc': acc.item()
                })
        
        return {
            'loss': loss_meter.avg,
            'acc': acc_meter.avg
        }
    
    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        
        loss_meter = AverageMeter()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(val_loader, desc='Validation')
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            batch_size = images.size(0)
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Update meters
            loss_meter.update(loss.item(), batch_size)
            
            # Store predictions
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
        
        # Compute metrics
        metrics = compute_metrics(
            y_true=np.array(all_labels),
            y_pred=np.array(all_preds),
            y_prob=np.array(all_probs)
        )
        
        metrics['loss'] = loss_meter.avg
        
        return metrics
    
    def train(
        self,
        train_loader,
        val_loader,
        start_epoch: int = 0
    ) -> Dict[str, List]:
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            start_epoch: Starting epoch (for resuming)
        
        Returns:
            Training history
        """
        
        print(f"\n{'='*50}")
        print(f"Starting training for {self.epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"{'='*50}\n")
        
        for epoch in range(start_epoch, self.epochs):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print("-" * 50)
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['learning_rate'].append(current_lr)
            
            # Print metrics
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"Time: {epoch_time:.2f}s | LR: {current_lr:.6f}")
            print(f"Train - Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['acc']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | MAE: {val_metrics['mae']:.4f}")
            print(f"Val   - Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}")
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/loss': train_metrics['loss'],
                    'train/acc': train_metrics['acc'],
                    'val/loss': val_metrics['loss'],
                    'val/acc': val_metrics['accuracy'],
                    'val/mae': val_metrics['mae'],
                    'val/precision': val_metrics['precision'],
                    'val/recall': val_metrics['recall'],
                    'val/f1': val_metrics['f1'],
                    'learning_rate': current_lr
                })
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                
                save_path = self.checkpoint_dir / 'best_model.pth'
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    loss=val_metrics['loss'],
                    metrics=val_metrics,
                    path=save_path
                )
                print(f"✓ Saved best model (val_loss: {val_metrics['loss']:.4f})")
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                save_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pth'
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    loss=val_metrics['loss'],
                    metrics=val_metrics,
                    path=save_path
                )
            
            # Early stopping
            self.early_stopping(val_metrics['loss'])
            if self.early_stopping.early_stop:
                print(f"\n⚠ Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("\n✓ Loaded best model weights")
        
        return self.history
    
    def test(self, test_loader) -> Dict[str, float]:
        """
        Evaluate on test set
        
        Args:
            test_loader: Test data loader
        
        Returns:
            Test metrics
        """
        print("\n" + "="*50)
        print("Evaluating on test set...")
        print("="*50)
        
        test_metrics = self.validate(test_loader)
        
        print("\nTest Set Results:")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"MAE: {test_metrics['mae']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall: {test_metrics['recall']:.4f}")
        print(f"F1-Score: {test_metrics['f1']:.4f}")
        
        if self.use_wandb:
            wandb.log({
                'test/acc': test_metrics['accuracy'],
                'test/mae': test_metrics['mae'],
                'test/precision': test_metrics['precision'],
                'test/recall': test_metrics['recall'],
                'test/f1': test_metrics['f1']
            })
        
        return test_metrics


def mixup_data(x, y, alpha=1.0):
    """
    Apply MixUp augmentation
    
    Args:
        x: Input batch
        y: Target labels
        alpha: MixUp parameter
    
    Returns:
        Mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute MixUp loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


if __name__ == "__main__":
    # Test trainer
    from ..models.custom_architectures import build_model
    from ..models.losses import get_loss_function
    from ..data.dataset import create_data_loaders
    
    # Configuration
    config = {
        'model_type': 'efficientnet',
        'model_name': 'efficientnet_b4',
        'num_classes': 5,
        'pretrained': True,
        'use_ordinal': False,
        'loss_type': 'ce',
        'epochs': 2,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'use_amp': True,
        'checkpoint_dir': 'models/checkpoints'
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build model
    model = build_model(config)
    
    # Loss and optimizer
    criterion = get_loss_function(config)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs']
    )
    
    # Data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        root_dir='data/processed',
        batch_size=config['batch_size']
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config
    )
    
    # Train
    history = trainer.train(train_loader, val_loader)
    
    # Test
    test_metrics = trainer.test(test_loader)
