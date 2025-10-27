import torch
import os

# --- Machine & Model Config ---
GPU_ID = "2"  # GPU to use (e.g., "0", "1", "2")
MODEL_NAME = "resnet18"  # Options: "resnet18", "resnet101"

# --- Directory Config ---
ORIG_DATA_DIR = "Data/train"
# This directory will be created by data_setup.py
BAL_DIR_BASE = "./balanced_data" 
# These directories will also be created
TRAIN_DIR = "./balanced_data_train"
VAL_DIR = "./balanced_data_val"
OUTPUT_DIR = "./output"

# --- File Path Config ---
# --- THIS IS UPDATED ---
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_best.pth")
METRICS_PLOT_PATH = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_metrics.png")
CM_PLOT_PATH = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_confusion_matrix.png")

# --- Model & Training Params ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_EPOCHS = 10  # From notebook
LEARNING_RATE = 0.001
NUM_CLASSES = 5  # 0, 1, 2, 3, 4

# --- Device ---
# This is now flexible and will respect the GPU_ID set above
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Class Balancing Params ---
MAX_SAMPLES = 500
MIN_SAMPLES = 173
BALANCE_N = 500  # Number of samples per class after augmentation
VAL_SPLIT = 0.2  # 20% for validation

# --- Reproducibility ---
SEED = 42