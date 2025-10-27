import torch
import torch.nn as nn
from tqdm import tqdm

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Performs one training epoch."""
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    # Wrap dataloader with tqdm for a progress bar
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_preds += torch.sum(preds == labels.data)
        total_preds += inputs.size(0)
        
        # Update progress bar
        progress_bar.set_postfix(loss=running_loss/total_preds, acc=correct_preds.double().item()/total_preds)

    epoch_loss = running_loss / total_preds
    epoch_acc = correct_preds.double() / total_preds
    return epoch_loss, epoch_acc.item()

def validate_epoch(model, dataloader, criterion, device):
    """Performs one validation epoch."""
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    # Wrap dataloader with tqdm for a progress bar
    progress_bar = tqdm(dataloader, desc="Validating", leave=False)
    
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data)
            total_preds += inputs.size(0)
            
            # Update progress bar
            progress_bar.set_postfix(loss=running_loss/total_preds, acc=correct_preds.double().item()/total_preds)

    epoch_loss = running_loss / total_preds
    epoch_acc = correct_preds.double() / total_preds
    return epoch_loss, epoch_acc.item()