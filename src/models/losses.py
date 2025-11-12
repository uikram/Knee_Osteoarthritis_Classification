"""
Custom loss functions for knee OA classification
Includes ordinal, focal, and class-balanced losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class OrdinalRegressionLoss(nn.Module):
    """
    Ordinal regression loss for KL grading
    Based on Chen et al. 2019 paper
    """
    def __init__(self, num_classes=5, reduction='mean'):
        super().__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, num_thresholds) - ordinal logits
            targets: (B,) - class labels [0, num_classes-1]
        """
        batch_size = logits.size(0)
        
        # Convert targets to ordinal encoding
        # For target k, we want: [1,1,...,1,0,0,...,0] with k ones
        ordinal_targets = torch.zeros(batch_size, self.num_thresholds, 
                                       device=logits.device)
        
        for i in range(batch_size):
            target = targets[i].item()
            ordinal_targets[i, :target] = 1.0
        
        # Binary cross-entropy for each threshold
        loss = F.binary_cross_entropy_with_logits(
            logits, ordinal_targets, reduction='none'
        )
        
        # Average across thresholds and batch
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C) - class logits
            targets: (B,) - class labels
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha = torch.tensor(self.alpha, device=logits.device)
            else:
                alpha = self.alpha
            
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """Cross-entropy with class weights for imbalanced data"""
    def __init__(self, class_weights=None, reduction='mean'):
        super().__init__()
        self.class_weights = class_weights
        self.reduction = reduction
    
    def forward(self, logits, targets):
        if self.class_weights is not None:
            if isinstance(self.class_weights, (list, tuple)):
                weights = torch.tensor(self.class_weights, device=logits.device)
            else:
                weights = self.class_weights
        else:
            weights = None
        
        return F.cross_entropy(logits, targets, weight=weights, 
                               reduction=self.reduction)


class LabelSmoothingLoss(nn.Module):
    """Label smoothing for better generalization"""
    def __init__(self, num_classes=5, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C) - class logits
            targets: (B,) - class labels
        """
        log_probs = F.log_softmax(logits, dim=1)
        
        # Create smoothed targets
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        loss = -true_dist * log_probs
        
        if self.reduction == 'mean':
            return loss.sum(dim=1).mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """Combination of multiple losses with weights"""
    def __init__(self, losses, weights=None):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        
        if weights is None:
            weights = [1.0] * len(losses)
        self.weights = weights
    
    def forward(self, logits, targets):
        total_loss = 0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(logits, targets)
        return total_loss


def get_loss_function(config):
    """Factory function to create loss function from config"""
    loss_type = config.get('loss_type', 'ce')
    num_classes = config.get('num_classes', 5)
    
    if loss_type == 'ce':
        # Standard cross-entropy
        class_weights = config.get('class_weights', None)
        if class_weights:
            return WeightedCrossEntropyLoss(class_weights)
        return nn.CrossEntropyLoss()
    
    elif loss_type == 'focal':
        # Focal loss for class imbalance
        alpha = config.get('focal_alpha', None)
        gamma = config.get('focal_gamma', 2.0)
        return FocalLoss(alpha, gamma)
    
    elif loss_type == 'ordinal':
        # Ordinal regression loss
        return OrdinalRegressionLoss(num_classes)
    
    elif loss_type == 'label_smoothing':
        # Label smoothing
        smoothing = config.get('smoothing', 0.1)
        return LabelSmoothingLoss(num_classes, smoothing)
    
    elif loss_type == 'combined':
        # Combination of losses
        loss_configs = config.get('combined_losses', [])
        losses = [get_loss_function(cfg) for cfg in loss_configs]
        weights = config.get('combined_weights', None)
        return CombinedLoss(losses, weights)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test loss functions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dummy data
    batch_size = 16
    num_classes = 5
    logits = torch.randn(batch_size, num_classes).to(device)
    targets = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    # Test standard CE
    ce_loss = nn.CrossEntropyLoss()
    print(f"CE Loss: {ce_loss(logits, targets).item():.4f}")
    
    # Test focal loss
    focal_loss = FocalLoss(gamma=2.0)
    print(f"Focal Loss: {focal_loss(logits, targets).item():.4f}")
    
    # Test ordinal loss
    ordinal_logits = torch.randn(batch_size, num_classes - 1).to(device)
    ordinal_loss = OrdinalRegressionLoss(num_classes)
    print(f"Ordinal Loss: {ordinal_loss(ordinal_logits, targets).item():.4f}")
    
    # Test label smoothing
    smooth_loss = LabelSmoothingLoss(num_classes, smoothing=0.1)
    print(f"Label Smoothing Loss: {smooth_loss(logits, targets).item():.4f}")
