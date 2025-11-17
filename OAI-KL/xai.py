"""
Notebook 5: Explainable AI Integration
Grad-CAM, Grad-CAM++, LIME visualizations for Knee X-ray Classification

Requirements:
pip install grad-cam opencv-python lime matplotlib seaborn torch torchvision albumentations pandas numpy scikit-image

Usage:
1. Update paths in XAIConfig class
2. Ensure trained models are available in model_base_path
3. Run the script to generate visualizations
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Grad-CAM imports
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# LIME imports
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

#=============================================================================
# CONFIGURATION
#=============================================================================

class XAIConfig:
    """Configuration for XAI Analysis"""
    
    # Paths - UPDATE THESE
    test_csv = "../KneeXray/test/test_correct.csv"
    train_csv = "../KneeXray/train/train.csv"
    model_base_path = "./models"
    output_dir = "./xai_visualizations"
    
    # Model configurations to analyze
    # model_configs = [
    #     # {'name': 'resnet50', 'size': 224},
    #     {'name': 'densenet_161', 'size': 224},
    #     # 
    # ]
    model_configs = [
    {
        'name': 'densenet_161',
        'size': 224,
        'folder': '(224, 224)'
    },
    {
        'name': 'efficientnet_b5',
        'size': 224,
        'folder': '(224, 224)'
    },
    {
        'name': 'efficientnet_v2_s',
        'size': 224,
        'folder': '(224, 224)'
    },
    {
        'name': 'regnet_y_8gf',
        'size': 224,
        'folder': '(224, 224)'
    },
    {
        'name': 'resnet_101',
        'size': 224,
        'folder': '(224, 224)'
    },
    {
        'name': 'resnext_50_32x4d',
        'size': 224,
        'folder': '(224, 224)'
    },
    {
        'name': 'shufflenet_v2_x2_0',
        'size': 224,
        'folder': '(224, 224)'
    },
    {
        'name': 'wide_resnet_50_2',
        'size': 224,
        'folder': '(224, 224)'
    }
]

    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Class information
    class_names = ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 4', 'Grade 5']
    num_classes = 5
    grade_descriptions = {
        0: "Grade 0 (Normal): No signs of osteoarthritis. Joint space is normal.",
        1: "Grade 1 (Doubtful): Minimal osteophytes (small bony growths) may be present. Joint space is normal.",
        2: "Grade 2 (Mild): Definite small osteophytes. Possible joint space narrowing.",
        4: "Grade 4 (Moderate): Multiple osteophytes, definite joint space narrowing, some bone deformity (sclerosis).",
        5: "Grade 5 (Severe): Large osteophytes, severe joint space narrowing, significant bone deformity, and sclerosis."
    }   
    # Sampling
    num_samples_per_class = 5  # Samples to visualize per class
    fold_to_analyze = 1  # Which fold's model to use
    
    # LIME parameters
    lime_num_samples = 1000
    lime_num_features = 10
    
    # Visualization
    figsize = (20, 12)
    cam_alpha = 0.5
    dpi = 150

config = XAIConfig()
os.makedirs(config.output_dir, exist_ok=True)

print(f"Device: {config.device}")
print(f"Output directory: {config.output_dir}")
print(f"CUDA available: {torch.cuda.is_available()}")

#=============================================================================
# DATASET CLASS
#=============================================================================

class ImageDataset(Dataset):
    """Custom dataset for knee X-ray images"""
    
    def __init__(self, csv_file, transforms=None):
        if isinstance(csv_file, str):
            self.data = pd.read_csv(csv_file)
        else:
            self.data = csv_file
        self.transforms = transforms
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['data']
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target = self.data.iloc[idx]['label']
        
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']
        
        return {
            'image': image,
            'target': torch.tensor(target, dtype=torch.long),
            'path': img_path
        }
    
    def get_labels(self):
        return self.data['label'].values

#=============================================================================
# MODEL LOADING UTILITIES
#=============================================================================

def create_model(model_name, num_classes=5):
    """
    Create model architecture based on model name
    Supports multiple architectures from torchvision
    """
    from torchvision import models
    
    model_name_lower = model_name.lower()
    
    # ResNet family
    if model_name_lower == 'resnet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name_lower == 'resnet_101' or model_name_lower == 'resnet101':
        model = models.resnet101(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name_lower == 'resnet34':
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name_lower == 'resnext_50_32x4d' or model_name_lower == 'resnext50_32x4d':
        model = models.resnext50_32x4d(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name_lower == 'wide_resnet_50_2' or model_name_lower == 'wide_resnet50_2':
        model = models.wide_resnet50_2(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # EfficientNet family
    elif model_name_lower == 'efficientnet_b0' or model_name_lower == 'efficientnetb0':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name_lower == 'efficientnet_b1' or model_name_lower == 'efficientnetb1':
        model = models.efficientnet_b1(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name_lower == 'efficientnet_b5' or model_name_lower == 'efficientnetb5':
        model = models.efficientnet_b5(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name_lower == 'efficientnet_v2_s' or model_name_lower == 'efficientnetv2_s':
        model = models.efficientnet_v2_s(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    # DenseNet family
    elif model_name_lower == 'densenet121':
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name_lower == 'densenet_161' or model_name_lower == 'densenet161':
        model = models.densenet161(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    
    # RegNet
    elif model_name_lower == 'regnet_y_8gf' or model_name_lower == 'regnety_8gf':
        model = models.regnet_y_8gf(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # ShuffleNet
    elif model_name_lower == 'shufflenet_v2_x2_0' or model_name_lower == 'shufflenetv2_x2_0':
        model = models.shufflenet_v2_x2_0(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # MobileNet
    elif model_name_lower == 'mobilenet_v2' or model_name_lower == 'mobilenetv2':
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    # VGG
    elif model_name_lower == 'vgg16':
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    
    else:
        raise ValueError(f"Model '{model_name}' not supported. Please add it to create_model function.")
    
    return model

def get_target_layer(model, model_name):
    """
    Get the target layer for Grad-CAM based on model architecture
    Returns the last convolutional layer before the classifier
    """
    model_name_lower = model_name.lower()
    
    if 'resnet' in model_name_lower or 'resnext' in model_name_lower or 'wide_resnet' in model_name_lower:
        return [model.layer4[-1]]
    elif 'efficientnet' in model_name_lower:
        return [model.features[-1]]
    elif 'densenet' in model_name_lower:
        return [model.features[-1]]
    elif 'regnet' in model_name_lower:
        return [model.trunk_output[-1]]
    elif 'shufflenet' in model_name_lower:
        return [model.conv5]
    elif 'vgg' in model_name_lower:
        return [model.features[-1]]
    elif 'mobilenet' in model_name_lower:
        return [model.features[-1]]
    else:
        # Generic approach: find last Conv2d layer
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Conv2d):
                return [module]
        raise ValueError(f"Could not find target layer for {model_name}")

def load_model_weights(model_config, fold=5):
    """
    Load trained model weights for a given configuration and fold
    """
    model_name = model_config['name']
    img_size = model_config['size']
    folder = model_config.get('folder', str(img_size))  # Use folder if specified

    # Construct full model path
    model_path = os.path.join(config.model_base_path, model_name, folder)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}\nAvailable models: {os.listdir(os.path.join(config.model_base_path, model_name))}")

    # Find model files for the specified fold
    model_files = [f for f in os.listdir(model_path) 
                   if f.startswith(f'{fold}fold_') and f.endswith('.pt')]

    if not model_files:
        available_files = [f for f in os.listdir(model_path) if f.endswith('.pt')]
        raise FileNotFoundError(
            f"No model files found for fold {fold} in {model_path}\n"
            f"Available files: {available_files}\n"
            f"Looking for pattern: {fold}fold_epoch*.pt"
        )

    model_file = sorted(model_files)[-1]  # Use latest epoch
    model_path_full = os.path.join(model_path, model_file)

    print(f"   Loading: {model_file}")
    print(f"   From: {model_path_full}")

    # Create model architecture
    model = create_model(model_name, config.num_classes)

    # Load weights
    state_dict = torch.load(model_path_full, map_location=config.device, weights_only=True)

    # Handle DataParallel prefix
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    model.to(config.device)

    print(f"   ✓ Successfully loaded: {model_file}")
    return model, model_name

#=============================================================================
# GRAD-CAM IMPLEMENTATION
#=============================================================================

def generate_gradcam(model, img_tensor, target_layer, target_class=None, method='gradcam'):
    """
    Generate Grad-CAM visualization
    
    Args:
        model: PyTorch model
        img_tensor: Input image tensor (C, H, W)
        target_layer: Layer to visualize
        target_class: Target class for CAM (None = predicted class)
        method: 'gradcam', 'gradcam++', or 'scorecam'
    
    Returns:
        cam: CAM heatmap
        prediction: Model prediction
        confidence: Prediction confidence
    """
    input_tensor = img_tensor.unsqueeze(0).to(config.device)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        prediction = output.argmax(dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    if target_class is None:
        target_class = prediction
    
    # Initialize CAM algorithm
    if method == 'gradcam':
        cam_algorithm = GradCAM(model=model, target_layers=target_layer)
    elif method == 'gradcam++':
        cam_algorithm = GradCAMPlusPlus(model=model, target_layers=target_layer)
    elif method == 'scorecam':
        cam_algorithm = ScoreCAM(model=model, target_layers=target_layer)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Generate CAM
    targets = [ClassifierOutputTarget(target_class)]
    cam = cam_algorithm(input_tensor=input_tensor, targets=targets)
    cam = cam[0, :]
    
    return cam, prediction, confidence

#=============================================================================
# LIME IMPLEMENTATION
#=============================================================================

def generate_lime_explanation(model, image_np, img_size):
    """
    Generate LIME explanation for the model prediction
    
    Args:
        model: PyTorch model
        image_np: Numpy image (H, W, C) in range [0, 1]
        img_size: Image size for model input
    
    Returns:
        explanation: LIME explanation object
        prediction: Model prediction
    """
    def predict_fn(images):
        """Prediction function for LIME"""
        transform = A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        batch_tensors = []
        for img in images:
            img_uint8 = (img * 255).astype(np.uint8)
            augmented = transform(image=img_uint8)
            batch_tensors.append(augmented['image'])
        
        batch = torch.stack(batch_tensors).to(config.device)
        
        with torch.no_grad():
            output = model(batch)
            probabilities = torch.nn.functional.softmax(output, dim=1)
        
        return probabilities.cpu().numpy()
    
    # Initialize LIME explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Generate explanation
    explanation = explainer.explain_instance(
        image_np,
        predict_fn,
        top_labels=5,
        hide_color=0,
        num_samples=config.lime_num_samples
    )
    
    # Get prediction
    probs = predict_fn(np.array([image_np]))[0]
    prediction = probs.argmax()
    
    return explanation, prediction

#=============================================================================
# COMPREHENSIVE VISUALIZATION
#=============================================================================

def create_comprehensive_xai_visualization(model, model_name, img_tensor, image_np, 
                                            target_layer, true_label, img_path, img_size):
    """
    Create a comprehensive visualization combining all XAI methods
    """
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
    
    # 1. Original Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image_np)
    ax1.set_title(f'Original Image\nTrue: {config.class_names[true_label]}', 
                  fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. Grad-CAM
    try:
        cam_gradcam, prediction, confidence = generate_gradcam(
            model, img_tensor, target_layer, method='gradcam'
        )
        cam_image_gradcam = show_cam_on_image(image_np, cam_gradcam, use_rgb=True)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(cam_image_gradcam)
        ax2.set_title(f'Grad-CAM\nPred: {config.class_names[prediction]} ({confidence:.2%})', 
                      fontsize=12, fontweight='bold')
        ax2.axis('off')
    except Exception as e:
        print(f"  Grad-CAM failed: {e}")
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.text(0.5, 0.5, 'Grad-CAM Failed', ha='center', va='center')
        ax2.axis('off')
        prediction = -1
        confidence = 0
    
    # 3. Grad-CAM++
    try:
        cam_gradcam_plus, _, _ = generate_gradcam(
            model, img_tensor, target_layer, method='gradcam++'
        )
        cam_image_gradcam_plus = show_cam_on_image(image_np, cam_gradcam_plus, use_rgb=True)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(cam_image_gradcam_plus)
        ax3.set_title('Grad-CAM++', fontsize=12, fontweight='bold')
        ax3.axis('off')
    except Exception as e:
        print(f"  Grad-CAM++ failed: {e}")
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.text(0.5, 0.5, 'Grad-CAM++ Failed', ha='center', va='center')
        ax3.axis('off')
    
    # 4. Heatmap overlay
    try:
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(image_np, alpha=0.5)
        ax4.imshow(cam_gradcam, cmap='jet', alpha=0.5)
        ax4.set_title('Grad-CAM Overlay', fontsize=12, fontweight='bold')
        ax4.axis('off')
    except:
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.axis('off')
    
    # 5-7. LIME explanations
    try:
        explanation, lime_pred = generate_lime_explanation(model, image_np, img_size)
        
        # Positive regions
        temp, mask = explanation.get_image_and_mask(
            prediction if prediction >= 0 else lime_pred, 
            positive_only=True, num_features=10, hide_rest=False
        )
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.imshow(mark_boundaries(temp, mask))
        ax5.set_title('LIME: Supporting Evidence', fontsize=12, fontweight='bold')
        ax5.axis('off')
        
        # Positive & Negative
        temp, mask = explanation.get_image_and_mask(
            prediction if prediction >= 0 else lime_pred,
            positive_only=False, num_features=10, hide_rest=False
        )
        ax6 = fig.add_subplot(gs[1, 1])
        ax6.imshow(mark_boundaries(temp, mask))
        ax6.set_title('LIME: Pos & Neg', fontsize=12, fontweight='bold')
        ax6.axis('off')
        
        # Key features only
        temp, mask = explanation.get_image_and_mask(
            prediction if prediction >= 0 else lime_pred,
            positive_only=True, num_features=5, hide_rest=True
        )
        ax7 = fig.add_subplot(gs[1, 2])
        ax7.imshow(mark_boundaries(temp, mask))
        ax7.set_title('LIME: Key Features', fontsize=12, fontweight='bold')
        ax7.axis('off')
        
    except Exception as e:
        print(f"  LIME failed: {e}")
        for i in range(3):
            ax = fig.add_subplot(gs[1, i])
            ax.text(0.5, 0.5, 'LIME Failed', ha='center', va='center')
            ax.axis('off')
    
    # 8. Prediction confidence
    try:
        with torch.no_grad():
            input_tensor = img_tensor.unsqueeze(0).to(config.device)
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()
        
        ax8 = fig.add_subplot(gs[1, 3])
        colors = ['green' if i == prediction else 'skyblue' for i in range(5)]
        bars = ax8.barh(config.class_names, probs, color=colors)
        ax8.set_xlabel('Confidence', fontsize=10)
        ax8.set_title('Class Probabilities', fontsize=12, fontweight='bold')
        ax8.set_xlim([0, 1])
        
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax8.text(prob + 0.02, i, f'{prob:.1%}', va='center', fontsize=9)
    except:
        pass
    
    # Overall title
    correct = '✓' if prediction == true_label else '✗'
    fig.suptitle(f'{model_name} - XAI Analysis {correct}\n{os.path.basename(img_path)}', 
                 fontsize=14, fontweight='bold')
    
    # Save
    filename = os.path.basename(img_path).replace('.png', f'_{model_name}_xai.png')
    save_path = os.path.join(config.output_dir, filename)
    plt.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {filename}")
    return save_path

#=============================================================================
# SAMPLE SELECTION
#=============================================================================

def select_samples_for_visualization(test_csv, num_samples_per_class=5):
    """Select representative samples from each class"""
    df = pd.read_csv(test_csv)
    selected_samples = []
    
    for class_idx in range(config.num_classes):
        class_samples = df[df['label'] == class_idx]
        
        if len(class_samples) == 0:
            print(f"⚠ Warning: No samples for class {class_idx}")
            continue
        
        if len(class_samples) < num_samples_per_class:
            print(f"⚠ Warning: Class {class_idx} has only {len(class_samples)} samples")
            selected = class_samples
        else:
            indices = np.linspace(0, len(class_samples)-1, num_samples_per_class, dtype=int)
            selected = class_samples.iloc[indices]
        
        selected_samples.append(selected)
    
    return pd.concat(selected_samples, ignore_index=True) if selected_samples else pd.DataFrame()

#=============================================================================
# MAIN EXECUTION
#=============================================================================

def run_xai_analysis():
    """Main function to run XAI analysis"""
    print("="*70)
    print(" "*20 + "EXPLAINABLE AI ANALYSIS")
    print("="*70)
    
    # Load test data
    print("\n📊 Loading test dataset...")
    test_df = pd.read_csv(config.test_csv)
    print(f"   Total test samples: {len(test_df)}")
    
    # Select samples
    print(f"\n🎯 Selecting {config.num_samples_per_class} samples per class...")
    selected_df = select_samples_for_visualization(
        config.test_csv, 
        config.num_samples_per_class
    )
    print(f"Selected {len(selected_df)} samples total")
    
    # Process each model
    for idx, model_config in enumerate(config.model_configs, 1):
        model_name = model_config['name']
        img_size = model_config['size']
        
        print(f"\n{'='*70}")
        print(f"🔍 Model {idx}/{len(config.model_configs)}: {model_name} (Size: {img_size})")
        print(f"{'='*70}")
        
        try:
            # Load model
            model, _ = load_model_weights(model_config, fold=config.fold_to_analyze)
            
            # Get target layer
            target_layer = get_target_layer(model, model_name)
            print(f"   Target layer identified: {type(target_layer[0]).__name__}")
            
            # Create transform
            transform = A.Compose([
                A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
                A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
            # Process samples
            print(f"\n   Processing {len(selected_df)} samples...")
            for sample_idx, (_, row) in enumerate(selected_df.iterrows(), 1):
                img_path = row['data']
                true_label = row['label']
                
                print(f"\n   [{sample_idx}/{len(selected_df)}] {os.path.basename(img_path)}")
                print(f"   True class: {config.class_names[true_label]}")
                
                # Load image
                image = cv2.imread(img_path)
                if image is None:
                    print(f"   ✗ Failed to load image")
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Prepare for visualization
                image_resized = cv2.resize(image, (img_size, img_size))
                image_np = image_resized.astype(np.float32) / 255.0
                
                # Prepare for model
                augmented = transform(image=image)
                img_tensor = augmented['image']
                
                # Create visualization
                create_comprehensive_xai_visualization(
                    model, model_name, img_tensor, image_np,
                    target_layer, true_label, img_path, img_size
                )
            
            print(f"\n✅ Completed {model_name}")
            
        except FileNotFoundError as e:
            print(f"\n❌ Model not found: {e}")
        except Exception as e:
            print(f"\n❌ Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("✅ XAI ANALYSIS COMPLETE")
    print(f"📁 Visualizations saved to: {config.output_dir}")
    print(f"{'='*70}")

def generate_summary_report():
    """Generate summary report"""
    print("\n📄 Generating Summary Report...")
    
    xai_files = [f for f in os.listdir(config.output_dir) if f.endswith('.png')]
    
    report = f"""
╔═══════════════════════════════════════════════════════════════╗
║           EXPLAINABLE AI ANALYSIS SUMMARY                     ║
╚═══════════════════════════════════════════════════════════════╝

📊 Analysis Statistics:
   • Total Visualizations: {len(xai_files)}
   • Models Analyzed: {len(config.model_configs)}
   • Samples per Class: {config.num_samples_per_class}
   • Output Directory: {config.output_dir}

🔬 XAI Methods Applied:
   ✓ Grad-CAM: Highlights regions important for classification
   ✓ Grad-CAM++: Improved localization over Grad-CAM
   ✓ LIME: Shows superpixels supporting/opposing predictions

📖 Interpretation Guide:
   • Grad-CAM: Warmer colors (red/yellow) = higher importance
   • LIME Green: Supports the prediction
   • LIME Red: Opposes the prediction

🎯 Clinical Relevance:
   1. Verify model focuses on anatomically relevant regions
   2. Check for attention to joint space, bone margins
   3. Identify potential spurious correlations
   4. Compare explanations across severity levels

📋 Next Steps:
   1. Review visualizations for clinical validity
   2. Consult with radiologists on attention patterns
   3. Identify and address any biases or artifacts
   4. Document findings for model validation

╚═══════════════════════════════════════════════════════════════╝
"""
    
    print(report)
    
    # Save report
    report_path = os.path.join(config.output_dir, 'xai_summary_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"✅ Report saved to: {report_path}")

if __name__ == "__main__":
    """
    Main entry point - just run the script!
    
    Usage:
        python xai_analysis.py
    
    Or in Jupyter/Colab:
        run_xai_analysis()
        generate_summary_report()
    """
    
    print("\n🚀 Starting XAI Analysis...")
    print(f"📁 Models directory: {config.model_base_path}")
    print(f"📊 Test CSV: {config.test_csv}")
    print(f"💾 Output directory: {config.output_dir}\n")
    
    # Run the analysis
    run_xai_analysis()
    
    # Generate summary report
    generate_summary_report()
    
    print("\n" + "="*70)
    print("🎉 ALL DONE!")
    print(f"📂 Check your visualizations in: {config.output_dir}")
    print("="*70)
  