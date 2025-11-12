"""
Grad-CAM and Grad-CAM++ implementations for CNN explainability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks"
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: CNN model
            target_layer: Target convolutional layer for visualization
        """
        self.model = model
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.handlers = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.handlers.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self.handlers.append(
            self.target_layer.register_full_backward_hook(backward_hook)
        )
    
    def remove_hooks(self):
        """Remove hooks"""
        for handler in self.handlers:
            handler.remove()
    
    def generate_cam(
        self,
        input_image: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_image: Input tensor (1, C, H, W)
            target_class: Target class index (None for predicted class)
        
        Returns:
            CAM heatmap (H, W)
        """
        
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        class_loss = output[0, target_class]
        class_loss.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0].cpu().numpy()  # (C, H, W)
        activations = self.activations[0].cpu().numpy()  # (C, H, W)
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)  # (H, W)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def generate_cam_batch(
        self,
        input_images: torch.Tensor,
        target_classes: Optional[List[int]] = None
    ) -> List[np.ndarray]:
        """Generate Grad-CAM for batch of images"""
        cams = []
        batch_size = input_images.size(0)
        
        for i in range(batch_size):
            target_class = target_classes[i] if target_classes else None
            cam = self.generate_cam(
                input_images[i:i+1],
                target_class
            )
            cams.append(cam)
        
        return cams
    
    def __del__(self):
        """Cleanup hooks"""
        self.remove_hooks()


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ implementation
    Reference: Chattopadhay et al., "Grad-CAM++: Generalized Gradient-based Visual Explanations"
    """
    
    def generate_cam(
        self,
        input_image: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Generate Grad-CAM++ heatmap"""
        
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        class_loss = output[0, target_class]
        class_loss.backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0].cpu().numpy()  # (C, H, W)
        activations = self.activations[0].cpu().numpy()  # (C, H, W)
        
        # Compute alpha weights
        numerator = gradients ** 2
        denominator = 2 * (gradients ** 2) + \
                      np.sum(activations * (gradients ** 3), axis=(1, 2), keepdims=True)
        
        # Avoid division by zero
        denominator = np.where(denominator != 0, denominator, 1e-10)
        
        alpha = numerator / denominator
        
        # ReLU on gradients
        relu_grad = np.maximum(gradients, 0)
        
        # Compute weights
        weights = np.sum(alpha * relu_grad, axis=(1, 2))
        
        # Weighted combination
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam


class LayerCAM:
    """
    Layer-CAM: More precise localization than Grad-CAM
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handlers = []
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.handlers.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self.handlers.append(
            self.target_layer.register_full_backward_hook(backward_hook)
        )
    
    def generate_cam(
        self,
        input_image: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Generate Layer-CAM heatmap"""
        
        self.model.eval()
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()
        
        # Element-wise multiplication (key difference from Grad-CAM)
        cam = np.sum(np.maximum(gradients, 0) * activations, axis=0)
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def remove_hooks(self):
        for handler in self.handlers:
            handler.remove()
    
    def __del__(self):
        self.remove_hooks()


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay heatmap on image
    
    Args:
        image: Original image (H, W, 3) in range [0, 255]
        heatmap: CAM heatmap (H, W) in range [0, 1]
        alpha: Blending factor
        colormap: OpenCV colormap
    
    Returns:
        Overlayed image (H, W, 3)
    """
    
    # Resize heatmap to match image size
    if heatmap.shape != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert heatmap to RGB
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = np.uint8(image * 255)
    
    # Blend
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlayed


def visualize_gradcam(
    model: nn.Module,
    image: torch.Tensor,
    original_image: np.ndarray,
    target_layer: nn.Module,
    class_names: List[str],
    target_class: Optional[int] = None,
    method: str = 'gradcam',
    save_path: Optional[str] = None
):
    """
    Visualize Grad-CAM results
    
    Args:
        model: CNN model
        image: Preprocessed image tensor (1, C, H, W)
        original_image: Original image for visualization (H, W, 3)
        target_layer: Target layer
        class_names: List of class names
        target_class: Target class (None for predicted)
        method: 'gradcam', 'gradcam++', or 'layercam'
        save_path: Path to save figure
    """
    
    # Initialize CAM
    if method == 'gradcam':
        cam_extractor = GradCAM(model, target_layer)
    elif method == 'gradcam++':
        cam_extractor = GradCAMPlusPlus(model, target_layer)
    elif method == 'layercam':
        cam_extractor = LayerCAM(model, target_layer)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Generate CAM
    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)[0]
        pred_class = output.argmax(dim=1).item()
    
    if target_class is None:
        target_class = pred_class
    
    cam = cam_extractor.generate_cam(image, target_class)
    
    # Overlay
    overlayed = overlay_heatmap(original_image, cam, alpha=0.5)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap
    im = axes[1].imshow(cam, cmap='jet')
    axes[1].set_title(f'{method.upper()} Heatmap\nTarget: {class_names[target_class]}',
                      fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(overlayed)
    axes[2].set_title('Overlay', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # Add prediction info
    pred_text = f"Predicted: {class_names[pred_class]} ({probs[pred_class]:.2%})"
    fig.text(0.5, 0.02, pred_text, ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Cleanup
    cam_extractor.remove_hooks()
    
    return cam, overlayed


def compare_cam_methods(
    model: nn.Module,
    image: torch.Tensor,
    original_image: np.ndarray,
    target_layer: nn.Module,
    class_names: List[str],
    target_class: Optional[int] = None,
    save_path: Optional[str] = None
):
    """
    Compare different CAM methods side-by-side
    
    Args:
        model: CNN model
        image: Preprocessed image tensor
        original_image: Original image
        target_layer: Target layer
        class_names: List of class names
        target_class: Target class
        save_path: Path to save figure
    """
    
    methods = ['gradcam', 'gradcam++', 'layercam']
    
    # Get prediction
    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)[0]
        pred_class = output.argmax(dim=1).item()
    
    if target_class is None:
        target_class = pred_class
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for idx, method in enumerate(methods):
        # Initialize CAM
        if method == 'gradcam':
            cam_extractor = GradCAM(model, target_layer)
        elif method == 'gradcam++':
            cam_extractor = GradCAMPlusPlus(model, target_layer)
        else:
            cam_extractor = LayerCAM(model, target_layer)
        
        # Generate CAM
        cam = cam_extractor.generate_cam(image, target_class)
        overlayed = overlay_heatmap(original_image, cam, alpha=0.5)
        
        # Plot heatmap
        axes[0, idx].imshow(cam, cmap='jet')
        axes[0, idx].set_title(f'{method.upper()}', fontsize=14, fontweight='bold')
        axes[0, idx].axis('off')
        
        # Plot overlay
        axes[1, idx].imshow(overlayed)
        axes[1, idx].set_title(f'{method.upper()} Overlay', fontsize=14, fontweight='bold')
        axes[1, idx].axis('off')
        
        # Cleanup
        cam_extractor.remove_hooks()
    
    # Add prediction info
    pred_text = f"Predicted: {class_names[pred_class]} ({probs[pred_class]:.2%})\n"
    pred_text += f"Target: {class_names[target_class]}"
    fig.suptitle(pred_text, fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def get_target_layer(model: nn.Module, model_type: str) -> nn.Module:
    """
    Get appropriate target layer for Grad-CAM based on model architecture
    
    Args:
        model: CNN model
        model_type: 'efficientnet', 'resnet', 'vit', 'convnext', etc.
    
    Returns:
        Target layer module
    """
    
    if 'efficientnet' in model_type.lower():
        # Last convolutional layer in EfficientNet
        return model.backbone.conv_head
    
    elif 'resnet' in model_type.lower():
        # Last layer of last residual block
        return model.backbone.layer4[-1]
    
    elif 'densenet' in model_type.lower():
        return model.backbone.features[-1]
    
    elif 'convnext' in model_type.lower():
        return model.backbone.stages[-1]
    
    elif 'vit' in model_type.lower():
        # For Vision Transformer, use last attention block
        return model.backbone.blocks[-1].norm1
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def analyze_gradcam_for_prompt(cam_heatmap: np.ndarray) -> str:
    """
    Analyze Grad-CAM heatmap and generate textual description for LLM prompts
    
    Args:
        cam_heatmap: Grad-CAM heatmap (H, W) with values in [0, 1]
    
    Returns:
        Textual description of findings
    """
    
    # Find high-attention regions
    threshold = 0.7
    high_attention = cam_heatmap > threshold
    
    # Analyze spatial distribution
    h, w = cam_heatmap.shape
    
    # Divide into quadrants
    top_half = cam_heatmap[:h//2, :]
    bottom_half = cam_heatmap[h//2:, :]
    left_half = cam_heatmap[:, :w//2]
    right_half = cam_heatmap[:, w//2:]
    
    # Compute attention scores
    top_score = np.mean(top_half)
    bottom_score = np.mean(bottom_half)
    left_score = np.mean(left_half)
    right_score = np.mean(right_half)
    
    # Generate description
    findings = "The model's attention (Grad-CAM) highlights the following regions:\n"
    
    # Identify primary regions
    regions = []
    if top_score > 0.5:
        regions.append("superior aspect of the joint")
    if bottom_score > 0.5:
        regions.append("inferior aspect of the joint")
    if left_score > 0.5:
        regions.append("medial compartment")
    if right_score > 0.5:
        regions.append("lateral compartment")
    
    if regions:
        findings += "  - High attention in: " + ", ".join(regions) + "\n"
    
    # Identify specific features
    max_val = np.max(cam_heatmap)
    max_coords = np.unravel_index(np.argmax(cam_heatmap), cam_heatmap.shape)
    
    findings += f"  - Peak attention at coordinates ({max_coords[0]}, {max_coords[1]}) "
    findings += f"with intensity {max_val:.2f}\n"
    
    # Attention distribution
    high_attention_pct = 100 * np.sum(high_attention) / high_attention.size
    findings += f"  - {high_attention_pct:.1f}% of the image shows high model attention (>0.7)\n"
    
    # Clinical interpretation hints
    if top_score > bottom_score:
        findings += "  - Attention concentrated in superior region, suggesting possible superior osteophytes\n"
    
    if left_score > right_score * 1.5:
        findings += "  - Predominantly medial compartment involvement\n"
    elif right_score > left_score * 1.5:
        findings += "  - Predominantly lateral compartment involvement\n"
    else:
        findings += "  - Relatively balanced medial and lateral compartment attention\n"
    
    return findings

if __name__ == "__main__":
    # Test Grad-CAM
    import torchvision.models as models
    from PIL import Image
    import torchvision.transforms as transforms
    
    # Load a pretrained model
    model = models.resnet50(pretrained=True)
    model.eval()
    
    # Target layer
    target_layer = model.layer4[-1]
    
    # Load and preprocess image
    img_path = "test_image.jpg"  # Replace with actual path
    image = Image.open(img_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    original_image = np.array(image.resize((224, 224)))
    
    # ImageNet class names (simplified)
    class_names = [f"Class {i}" for i in range(1000)]
    
    # Visualize
    visualize_gradcam(
        model=model,
        image=input_tensor,
        original_image=original_image,
        target_layer=target_layer,
        class_names=class_names,
        method='gradcam'
    )
