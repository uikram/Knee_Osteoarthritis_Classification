import torch.nn as nn
from torchvision import models

def build_model(model_name='resnet18', pretrained=True, num_classes=5):
    """Builds a ResNet model with fine-tuning on layer3 and layer4."""
    print(f"[INFO] Building {model_name} model...")
    model = None
    
    # Use modern 'weights' argument
    weights = 'IMAGENET1K_V1' if pretrained else None
    
    if model_name == 'resnet18':
        model = models.resnet18(weights=weights)
    elif model_name == 'resnet101':
        model = models.resnet101(weights=weights)
    
    if model is not None:
        # Freeze all layers first, then unfreeze layer3 and layer4
        for name, child in model.named_children():
            if name in ['layer3', 'layer4']:
                print(f"Unfreezing {name}...")
                for param in child.parameters():
                    param.requires_grad = True
            else:
                # Freeze other layers
                for param in child.parameters():
                    param.requires_grad = False
        
        # Replace the final fully connected layer (which will be trainable by default)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    else:
        raise ValueError(f"Model '{model_name}' not supported.")
        
    return model