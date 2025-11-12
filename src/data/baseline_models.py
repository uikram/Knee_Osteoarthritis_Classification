"""
Classic baseline CNN models for knee OA classification
"""
import torch.nn as nn
import torchvision.models as models

def get_resnet50_baseline(num_classes=5, pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_densenet121_baseline(num_classes=5, pretrained=True):
    model = models.densenet121(pretrained=pretrained)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model
