import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import config
from tqdm import tqdm

def plot_metrics(history, save_path):
    """Plots training and validation loss and accuracy."""
    print(f"[INFO] Plotting metrics to {save_path}...")
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    train_acc = history['train_acc']
    val_acc = history['val_acc']
    
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'bo-', label='Training acc')
    plt.plot(epochs, val_acc, 'ro-', label='Validation acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig(save_path)
    print(f"[INFO] Metrics plot saved to {save_path}")

@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """Gets predictions and labels from the validation set."""
    print("[INFO] Evaluating model on validation data...")
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    for inputs, labels in tqdm(dataloader, desc="Evaluating"):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plots and saves a confusion matrix."""
    print(f"[INFO] Plotting confusion matrix to {save_path}...")
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    # --- START OF FIX ---
    # Corrected 'np.new_axis' to 'np.newaxis'
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # --- END OF FIX ---

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_percent, 
        annot=True, 
        fmt='.2%', 
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.title('Confusion Matrix (Percentages)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(save_path)
    print(f"[INFO] Confusion matrix saved to {save_path}")

def get_classification_report(y_true, y_pred, class_names):
    """Prints a classification report and handles warnings."""
    print("\n--- Classification Report ---")
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=class_names, 
        digits=4,
        zero_division=0 
    )
    print(report)