"""
Custom CNN Architectures for Knee OA Classification
Implements multiple state-of-the-art architectures with ordinal loss support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import timm
from typing import Optional, Tuple


class CoordinateAttention(nn.Module):
    """Coordinate Attention mechanism for enhanced feature representation"""
    def __init__(self, channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, channels // reduction)
        
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU(inplace=True)
        
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        # Pooling in both directions
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        # Concatenate and transform
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # Split and apply attention
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        out = identity * a_h * a_w
        return out


class OrdinalHead(nn.Module):
    """Ordinal classification head for KL grading (0-4)"""
    def __init__(self, in_features, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        # Each binary classifier predicts P(y > k) for k=0,1,2,3
        self.num_thresholds = num_classes - 1
        
        self.fc = nn.Linear(in_features, self.num_thresholds)
        
    def forward(self, x):
        # Returns logits for each threshold
        return self.fc(x)
    
    def predict_probabilities(self, logits):
        """Convert ordinal logits to class probabilities"""
        # Apply sigmoid to get P(y > k) for each threshold
        cumulative_probs = torch.sigmoid(logits)
        
        # Calculate P(y = k) from cumulative probabilities
        batch_size = logits.size(0)
        probs = torch.zeros(batch_size, self.num_classes, device=logits.device)
        
        probs[:, 0] = 1 - cumulative_probs[:, 0]
        for k in range(1, self.num_thresholds):
            probs[:, k] = cumulative_probs[:, k-1] - cumulative_probs[:, k]
        probs[:, -1] = cumulative_probs[:, -1]
        
        return probs


class EfficientNetOA(nn.Module):
    """EfficientNet backbone with custom OA classification head"""
    def __init__(self, model_name='efficientnet_b4', num_classes=5, 
                 pretrained=True, use_ordinal=True, use_attention=True):
        super().__init__()
        self.use_ordinal = use_ordinal
        
        # Load pretrained EfficientNet
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.backbone.classifier.in_features
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Add coordinate attention
        if use_attention:
            self.attention = CoordinateAttention(in_features)
        else:
            self.attention = None
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Classification head
        if use_ordinal:
            self.head = OrdinalHead(in_features, num_classes)
        else:
            self.head = nn.Linear(in_features, num_classes)
    
    def forward(self, x, return_features=False):
        # Extract features
        features = self.backbone.forward_features(x)
        
        # Apply attention if enabled
        if self.attention is not None:
            features = self.attention(features)
        
        # Global pooling
        pooled = self.global_pool(features)
        pooled = torch.flatten(pooled, 1)
        pooled = self.dropout(pooled)
        
        # Classification
        logits = self.head(pooled)
        
        if return_features:
            return logits, features
        return logits
    
    def predict(self, x):
        """Get class predictions and probabilities"""
        logits = self.forward(x)
        
        if self.use_ordinal:
            probs = self.head.predict_probabilities(logits)
            preds = torch.argmax(probs, dim=1)
        else:
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        
        return preds, probs


class VisionTransformerOA(nn.Module):
    """Vision Transformer for OA classification"""
    def __init__(self, model_name='vit_base_patch16_224', num_classes=5, 
                 pretrained=True, use_ordinal=True):
        super().__init__()
        self.use_ordinal = use_ordinal
        
        # Load pretrained ViT
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.backbone.head.in_features
        
        # Remove original head
        self.backbone.head = nn.Identity()
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
        
        # Classification head
        if use_ordinal:
            self.head = OrdinalHead(in_features, num_classes)
        else:
            self.head = nn.Linear(in_features, num_classes)
    
    def forward(self, x, return_attention=False):
        # ViT forward pass
        features = self.backbone.forward_features(x)
        
        # Extract CLS token (first token)
        cls_token = features[:, 0]
        cls_token = self.dropout(cls_token)
        
        # Classification
        logits = self.head(cls_token)
        
        if return_attention:
            # Return attention weights for visualization
            attention = self.backbone.blocks[-1].attn.get_attention_map()
            return logits, attention
        
        return logits
    
    def predict(self, x):
        """Get class predictions and probabilities"""
        logits = self.forward(x)
        
        if self.use_ordinal:
            probs = self.head.predict_probabilities(logits)
            preds = torch.argmax(probs, dim=1)
        else:
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        
        return preds, probs


class ConvNeXtOA(nn.Module):
    """ConvNeXt architecture for OA classification"""
    def __init__(self, model_name='convnext_base', num_classes=5, 
                 pretrained=True, use_ordinal=True):
        super().__init__()
        self.use_ordinal = use_ordinal
        
        # Load pretrained ConvNeXt
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.backbone.head.fc.in_features
        
        # Remove original head
        self.backbone.head.fc = nn.Identity()
        
        # Dropout
        self.dropout = nn.Dropout(0.25)
        
        # Classification head
        if use_ordinal:
            self.head = OrdinalHead(in_features, num_classes)
        else:
            self.head = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        # Extract features
        features = self.backbone.forward_features(x)
        features = self.backbone.head.global_pool(features)
        features = self.backbone.head.norm(features)
        features = self.backbone.head.flatten(features)
        features = self.dropout(features)
        
        # Classification
        logits = self.head(features)
        return logits
    
    def predict(self, x):
        """Get class predictions and probabilities"""
        logits = self.forward(x)
        
        if self.use_ordinal:
            probs = self.head.predict_probabilities(logits)
            preds = torch.argmax(probs, dim=1)
        else:
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        
        return preds, probs


class EnsembleModel(nn.Module):
    """Ensemble of multiple models with weighted averaging"""
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = nn.Parameter(torch.tensor(weights), requires_grad=False)
    
    def forward(self, x):
        # Get predictions from all models
        outputs = []
        for model in self.models:
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            outputs.append(probs)
        
        # Weighted average
        ensemble_probs = torch.stack(outputs, dim=0)
        weighted_probs = ensemble_probs * self.weights.view(-1, 1, 1)
        final_probs = weighted_probs.sum(dim=0)
        
        # Convert back to logits for loss computation
        final_logits = torch.log(final_probs + 1e-8)
        return final_logits
    
    def predict(self, x):
        """Get ensemble predictions"""
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds, probs


def build_model(config):
    """Factory function to build models from config"""
    model_type = config.get('model_type', 'efficientnet')
    num_classes = config.get('num_classes', 5)
    pretrained = config.get('pretrained', True)
    use_ordinal = config.get('use_ordinal', True)
    if model_type == 'resnet':
        model_name = config.get('model_name', 'resnet50')
        num_classes = config.get('num_classes', 5)
        pretrained = config.get('pretrained', True)
        
        # Load from torchvision
        import torchvision.models as models
        
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
        elif model_name == 'resnet101':
            model = models.resnet101(pretrained=pretrained)
        elif model_name == 'resnet152':
            model = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown ResNet variant: {model_name}")
        
        # Replace final layer
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
        return model
    
    if model_type == 'efficientnet':
        model_name = config.get('model_name', 'efficientnet_b4')
        use_attention = config.get('use_attention', True)
        model = EfficientNetOA(model_name, num_classes, pretrained, 
                               use_ordinal, use_attention)
    
    elif model_type == 'vit':
        model_name = config.get('model_name', 'vit_base_patch16_224')
        model = VisionTransformerOA(model_name, num_classes, pretrained, use_ordinal)
    
    elif model_type == 'convnext':
        model_name = config.get('model_name', 'convnext_base')
        model = ConvNeXtOA(model_name, num_classes, pretrained, use_ordinal)
    
    elif model_type == 'ensemble':
        # Build individual models for ensemble
        model_configs = config.get('ensemble_models', [])
        models = [build_model(cfg) for cfg in model_configs]
        weights = config.get('ensemble_weights', None)
        model = EnsembleModel(models, weights)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


if __name__ == "__main__":
    # Test model building
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test EfficientNet
    config = {
        'model_type': 'efficientnet',
        'model_name': 'efficientnet_b4',
        'num_classes': 5,
        'pretrained': True,
        'use_ordinal': True,
        'use_attention': True
    }
    
    model = build_model(config).to(device)
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"EfficientNet output shape: {output.shape}")
    
    preds, probs = model.predict(dummy_input)
    print(f"Predictions: {preds}")
    print(f"Probabilities shape: {probs.shape}")
