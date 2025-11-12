"""
Ensemble model for weighted averaging of predictions
"""
import torch
import torch.nn as nn

class EnsembleModel(nn.Module):
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights or [1.0 / len(models)] * len(models)

    def forward(self, x):
        probs = [torch.softmax(model(x), dim=1) for model in self.models]
        stacked = torch.stack(probs)
        weights_tensor = torch.tensor(self.weights, device=x.device).view(-1, 1, 1)
        combined = (stacked * weights_tensor).sum(dim=0)
        return torch.log(combined + 1e-8)
