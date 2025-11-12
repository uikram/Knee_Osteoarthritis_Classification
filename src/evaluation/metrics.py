"""
Evaluation metrics for Knee OA classification
"""

import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    roc_auc_score, cohen_kappa_score,
    mean_absolute_error as mae
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    num_classes: int = 5
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)
        num_classes: Number of classes
    
    Returns:
        Dictionary of metrics
    """
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    
    # MAE (important for ordinal classification)
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    
    # Cohen's Kappa
    metrics['kappa'] = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, per_class_support = \
        precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    for i in range(num_classes):
        metrics[f'precision_class_{i}'] = per_class_precision[i]
        metrics[f'recall_class_{i}'] = per_class_recall[i]
        metrics[f'f1_class_{i}'] = per_class_f1[i]
        metrics[f'support_class_{i}'] = per_class_support[i]
    
    # AUC (if probabilities provided)
    if y_prob is not None:
        try:
            # One-vs-rest AUC
            metrics['auc_ovr'] = roc_auc_score(
                y_true, y_prob, multi_class='ovr', average='weighted'
            )
        except ValueError:
            metrics['auc_ovr'] = 0.0
    
    # Confusion matrix metrics
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    # Sensitivity and Specificity for binary (OA vs No OA)
    # Grade 0 = No OA, Grades 1-4 = OA
    y_true_binary = (y_true > 0).astype(int)
    y_pred_binary = (y_pred > 0).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
    
    metrics['sensitivity_binary'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['specificity_binary'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # PPV and NPV
    metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    return metrics


def mean_absolute_error(y_true, y_pred):
    """Compute mean absolute error"""
    return mae(y_true, y_pred)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list = None,
    normalize: bool = False,
    title: str = 'Confusion Matrix',
    cmap: str = 'Blues',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        normalize: Whether to normalize
        title: Plot title
        cmap: Colormap
        figsize: Figure size
        save_path: Path to save figure
    """
    
    if class_names is None:
        class_names = [f'Grade {i}' for i in range(cm.shape[0])]
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_per_class_metrics(
    metrics: Dict[str, float],
    num_classes: int = 5,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
):
    """
    Plot per-class precision, recall, and F1-score
    
    Args:
        metrics: Dictionary containing per-class metrics
        num_classes: Number of classes
        figsize: Figure size
        save_path: Path to save figure
    """
    
    classes = [f'Grade {i}' for i in range(num_classes)]
    
    precision = [metrics[f'precision_class_{i}'] for i in range(num_classes)]
    recall = [metrics[f'recall_class_{i}'] for i in range(num_classes)]
    f1 = [metrics[f'f1_class_{i}'] for i in range(num_classes)]
    support = [metrics[f'support_class_{i}'] for i in range(num_classes)]
    
    x = np.arange(num_classes)
    width = 0.25
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot metrics
    ax1.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax1.bar(x, recall, width, label='Recall', alpha=0.8)
    ax1.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1.1])
    
    # Plot support
    ax2.bar(x, support, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Class', fontsize=12)
    ax2.set_ylabel('Support (# samples)', fontsize=12)
    ax2.set_title('Class Distribution', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list] = None
):
    """
    Print detailed classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    
    if class_names is None:
        class_names = [f'Grade {i}' for i in range(5)]
    
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4
    )
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(report)
    print("="*60 + "\n")


def create_metrics_table(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create comparison table of metrics across multiple models
    
    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        save_path: Path to save CSV
    
    Returns:
        DataFrame with metrics comparison
    """
    
    # Select key metrics to display
    key_metrics = [
        'accuracy', 'precision', 'recall', 'f1',
        'mae', 'kappa', 'auc_ovr',
        'sensitivity_binary', 'specificity_binary',
        'false_negative_rate'
    ]
    
    data = []
    for model_name, metrics in metrics_dict.items():
        row = {'Model': model_name}
        for metric in key_metrics:
            if metric in metrics:
                row[metric.replace('_', ' ').title()] = f"{metrics[metric]:.4f}"
        data.append(row)
    
    df = pd.DataFrame(data)
    
    if save_path:
        df.to_csv(save_path, index=False)
    
    return df


if __name__ == "__main__":
    # Test metrics computation
    np.random.seed(42)
    
    # Generate dummy predictions
    n_samples = 1000
    num_classes = 5
    
    y_true = np.random.randint(0, num_classes, n_samples)
    y_pred = np.random.randint(0, num_classes, n_samples)
    y_prob = np.random.dirichlet(np.ones(num_classes), n_samples)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob)
    
    print("Computed Metrics:")
    for key, value in metrics.items():
        if not isinstance(value, np.ndarray):
            print(f"{key}: {value:.4f}")
    
    # Plot confusion matrix
    cm = metrics['confusion_matrix']
    plot_confusion_matrix(cm, normalize=True)
    
    # Plot per-class metrics
    plot_per_class_metrics(metrics)
    
    # Print classification report
    print_classification_report(y_true, y_pred)
