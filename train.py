# --- START OF GPU/CONFIG SETUP ---
import os
import config
os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_ID
print(f"[INFO] Running on GPU: {config.GPU_ID}")
# --- END OF GPU/CONFIG SETUP ---

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import time

# Import from our custom modules
from data_setup import (
    create_dataframe, 
    trim,
    generate_balanced_dataframe,
    KneeDataset,
    train_transforms,
    val_transforms
)
from model import build_model
from engine import train_epoch, validate_epoch
from utils import (
    plot_metrics, 
    evaluate_model, 
    plot_confusion_matrix,
    get_classification_report
)

def main():
    # --- Setup ---
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Set seed for reproducibility
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if config.DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(config.SEED)

    # --- 1. Data Preprocessing & Balancing (Notebook Logic) ---
    df = create_dataframe(config.ORIG_DATA_DIR)
    trimmed_df = trim(df, config.MAX_SAMPLES, config.MIN_SAMPLES, 'labels')
    
    # Generate balanced DataFrame by duplicating paths
    balanced_df = generate_balanced_dataframe(
        trimmed_df, 
        config.BALANCE_N,
        random_state=config.SEED
    )
    
    # --- 2. Create Train/Val Split (from balanced DataFrame) ---
    print(f"[INFO] Splitting DataFrame into {1-config.VAL_SPLIT:.0%} train and {config.VAL_SPLIT:.0%} val...")
    train_df, val_df = train_test_split(
        balanced_df,
        test_size=config.VAL_SPLIT,
        random_state=config.SEED,
        stratify=balanced_df['labels'] # Ensure classes are balanced in splits
    )

    # --- 3. Create Datasets and DataLoaders ---
    print("[INFO] Creating Datasets and DataLoaders...")
    
    # Use KneeDataset with on-the-fly augmentations
    train_dataset = KneeDataset(train_df, transform=train_transforms)
    val_dataset = KneeDataset(val_df, transform=val_transforms)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=os.cpu_count(),
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=os.cpu_count(),
        pin_memory=True
    )
    
    print(f"Found {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    # --- 4. Model, Loss, Optimizer ---
    model = build_model(
        model_name=config.MODEL_NAME, 
        pretrained=True, 
        num_classes=config.NUM_CLASSES
    )
    model = model.to(config.DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    
    print("[INFO] Params to train:")
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print(f"\t{name}")
            
    optimizer = optim.Adam(params_to_update, lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # --- 5. Training Loop ---
    print(f"[INFO] Starting training for {config.NUM_EPOCHS} epochs on {config.DEVICE}...")
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, config.DEVICE)
        
        scheduler.step()
        
        end_time = time.time()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} - "
              f"Time: {end_time-start_time:.2f}s - "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"Saved best model to {config.MODEL_SAVE_PATH}")

    print("[INFO] Training complete.")

    # --- 6. Plot Metrics ---
    plot_metrics(history, config.METRICS_PLOT_PATH)

    # --- 7. Final Evaluation & Reporting ---
    print(f"[INFO] Loading best model from {config.MODEL_SAVE_PATH} for final evaluation...")
    
    model = build_model(
        model_name=config.MODEL_NAME, 
        pretrained=False,
        num_classes=config.NUM_CLASSES
    )
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    model = model.to(config.DEVICE)

    y_true, y_pred, _ = evaluate_model(model, val_loader, config.DEVICE)
    
    # Get class names from the dataset (they are sorted alphabetically)
    class_names = sorted(balanced_df['labels'].unique())
    
    get_classification_report(y_true, y_pred, class_names)
    
    plot_confusion_matrix(y_true, y_pred, class_names, config.CM_PLOT_PATH)
    
    print(f"[INFO] All outputs saved to {config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()