#!/usr/bin/env python3
"""
Evaluation script for trained models
Usage: python scripts/evaluate.py --model models/saved_models/best_model.pth --data data/processed
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.custom_architectures import build_model
from src.data.dataset import create_data_loaders
from src.evaluation.metrics import (
    compute_metrics, plot_confusion_matrix, 
    plot_per_class_metrics, print_classification_report
)


def main(args):
    """Main evaluation function"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading model from: {args.model}")
    checkpoint = torch.load(args.model, map_location=device)
    
    # Build model
    config = checkpoint.get('config', checkpoint)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✅ Model loaded successfully")
    
    # Load data
    print("\nLoading test data...")
    _, _, test_loader = create_data_loaders(
        root_dir=args.data,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    # Evaluate
    print("\nEvaluating model...")
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"MAE:               {metrics['mae']:.4f}")
    print(f"Precision:         {metrics['precision']:.4f}")
    print(f"Recall:            {metrics['recall']:.4f}")
    print(f"F1-Score:          {metrics['f1']:.4f}")
    print(f"Cohen's Kappa:     {metrics['kappa']:.4f}")
    print(f"AUC (OvR):         {metrics.get('auc_ovr', 'N/A'):.4f}")
    print(f"\nBinary Classification (OA vs No OA):")
    print(f"Sensitivity:       {metrics['sensitivity_binary']:.4f}")
    print(f"Specificity:       {metrics['specificity_binary']:.4f}")
    print(f"False Neg Rate:    {metrics['false_negative_rate']:.4f}")
    print(f"PPV:               {metrics['ppv']:.4f}")
    print(f"NPV:               {metrics['npv']:.4f}")
    print("="*60)
    
    # Classification report
    class_names = ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']
    print_classification_report(all_labels, all_preds, class_names)
    
    # Plot confusion matrix
    if args.plot:
        plot_confusion_matrix(
            metrics['confusion_matrix'],
            class_names=class_names,
            normalize=True,
            title='Normalized Confusion Matrix',
            save_path='evaluation_confusion_matrix.png'
        )
        
        plot_per_class_metrics(
            metrics,
            num_classes=5,
            save_path='evaluation_per_class_metrics.png'
        )
    
    # Save results
    if args.output:
        results_df = pd.DataFrame([{
            'Model': Path(args.model).stem,
            'Accuracy': metrics['accuracy'],
            'MAE': metrics['mae'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'Kappa': metrics['kappa'],
            'Sensitivity': metrics['sensitivity_binary'],
            'Specificity': metrics['specificity_binary'],
            'FNR': metrics['false_negative_rate']
        }])
        
        results_df.to_csv(args.output, index=False)
        print(f"\n✅ Results saved to: {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Knee OA Classification Model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to data directory')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save results CSV')
    
    args = parser.parse_args()
    main(args)
