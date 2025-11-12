#!/usr/bin/env python3
"""
Training script for Knee OA Classification
Usage: python scripts/train.py --config experiments/configs/efficientnet_b4.yaml
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

from src.models.custom_architectures import build_model
from src.models.losses import get_loss_function
from src.data.dataset import create_data_loaders
from src.training.trainer import Trainer


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_optimizer(config, model):
    """Create optimizer from config"""
    opt_config = config['training']['optimizer']
    opt_type = opt_config['type']
    
    if opt_type == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=opt_config['lr'],
            weight_decay=opt_config.get('weight_decay', 0)
        )
    elif opt_type == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=opt_config['lr'],
            weight_decay=opt_config.get('weight_decay', 0)
        )
    elif opt_type == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=opt_config['lr'],
            momentum=opt_config.get('momentum', 0.9),
            weight_decay=opt_config.get('weight_decay', 0)
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")
    
    return optimizer


def get_scheduler(config, optimizer):
    """Create scheduler from config"""
    sched_config = config['training']['scheduler']
    sched_type = sched_config['type']
    
    if sched_type == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_config['T_max']
        )
    elif sched_type == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=sched_config['T_0'],
            T_mult=sched_config.get('T_mult', 2)
        )
    elif sched_type == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=sched_config.get('patience', 5),
            factor=sched_config.get('factor', 0.1)
        )
    elif sched_type == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_config['step_size'],
            gamma=sched_config.get('gamma', 0.1)
        )
    else:
        scheduler = None
    
    return scheduler


def main(args):
    """Main training function"""
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded config from: {args.config}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_data_loaders(
        root_dir=config['data']['root_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        image_size=config['data']['image_size'],
        use_weighted_sampling=config['data']['use_weighted_sampling']
    )
    
    # Build model
    print("\nBuilding model...")
    model = build_model(config['model']).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    criterion = get_loss_function(config['training']['loss'])
    
    # Create optimizer
    optimizer = get_optimizer(config, model)
    
    # Create scheduler
    scheduler = get_scheduler(config, optimizer)
    
    # Training configuration
    train_config = {
        'epochs': config['training']['epochs'],
        'use_amp': config['training']['use_amp'],
        'gradient_clip': config['training']['gradient_clip'],
        'early_stopping_patience': config['training']['early_stopping_patience'],
        'checkpoint_dir': config['checkpoint']['save_dir'],
        'use_wandb': config['logging']['use_wandb'],
        'log_interval': config['logging']['log_interval']
    }
    
    # Create trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=train_config
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    history = trainer.train(train_loader, val_loader)
    
    # Test
    print("\n" + "="*60)
    print("Evaluating on Test Set")
    print("="*60)
    
    test_metrics = trainer.test(test_loader)
    
    # Save final results
    results_dir = Path('experiments/results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'results_{config["model"]["name"]}_{timestamp}.txt'
    
    with open(results_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"Model: {config['model']['name']}\n")
        f.write(f"Training completed: {timestamp}\n")
        f.write("="*60 + "\n\n")
        f.write("Test Metrics:\n")
        for key, value in test_metrics.items():
            if not isinstance(value, (list, dict)):
                f.write(f"  {key}: {value:.4f}\n")
    
    print(f"\n✅ Results saved to: {results_file}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Knee OA Classification Model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    args = parser.parse_args()
    
    main(args)
