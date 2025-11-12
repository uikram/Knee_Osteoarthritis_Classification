"""
LIME (Local Interpretable Model-agnostic Explanations) for image classification
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Callable, Tuple, Optional
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from PIL import Image


class LIMEExplainer:
    """
    LIME explainer for image classification models
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        preprocess_fn: Callable,
        device: torch.device = torch.device('cuda')
    ):
        """
        Args:
            model: PyTorch model
            preprocess_fn: Function to preprocess images
            device: Device to run model on
        """
        self.model = model
        self.preprocess_fn = preprocess_fn
        self.device = device
        self.model.eval()
        
        # Initialize LIME explainer
        self.explainer = lime_image.LimeImageExplainer()
    
    def predict_fn(self, images: np.ndarray) -> np.ndarray:
        """
        Prediction function for LIME
        
        Args:
            images: Batch of images (B, H, W, 3)
        
        Returns:
            Probabilities (B, num_classes)
        """
        batch_tensors = []
        
        for img in images:
            # Convert to PIL Image
            pil_img = Image.fromarray(img.astype('uint8'))
            
            # Preprocess
            tensor = self.preprocess_fn(pil_img)
            batch_tensors.append(tensor)
        
        # Stack into batch
        batch = torch.stack(batch_tensors).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(batch)
            probs = F.softmax(outputs, dim=1)
        
        return probs.cpu().numpy()
    
    def explain_instance(
        self,
        image: np.ndarray,
        top_labels: int = 5,
        num_samples: int = 1000,
        num_features: int = 5,
        hide_color: int = 0
    ) -> Tuple:
        """
        Generate LIME explanation for an image
        
        Args:
            image: Input image (H, W, 3)
            top_labels: Number of top labels to explain
            num_samples: Number of perturbed samples
            num_features: Number of superpixels to highlight
            hide_color: Color for hiding superpixels
        
        Returns:
            explanation, image with boundaries
        """
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            image,
            self.predict_fn,
            top_labels=top_labels,
            hide_color=hide_color,
            num_samples=num_samples,
            batch_size=32
        )
        
        return explanation
    
    def visualize_explanation(
        self,
        image: np.ndarray,
        explanation,
        label: int,
        positive_only: bool = True,
        num_features: int = 5,
        hide_rest: bool = False,
        save_path: Optional[str] = None
    ):
        """
        Visualize LIME explanation
        
        Args:
            image: Original image
            explanation: LIME explanation object
            label: Target label
            positive_only: Show only positive features
            num_features: Number of features to show
            hide_rest: Hide non-important regions
            save_path: Path to save figure
        """
        
        # Get image and mask
        temp, mask = explanation.get_image_and_mask(
            label,
            positive_only=positive_only,
            num_features=num_features,
            hide_rest=hide_rest
        )
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Explanation with boundaries
        axes[1].imshow(mark_boundaries(temp / 255.0, mask))
        axes[1].set_title(f'LIME Explanation\nLabel: {label}', 
                         fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Heatmap
        axes[2].imshow(mask, cmap='RdYlGn', vmin=0, vmax=1)
        axes[2].set_title('Importance Mask', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def compare_predictions(
        self,
        image: np.ndarray,
        explanation,
        class_names: list,
        top_k: int = 5,
        save_path: Optional[str] = None
    ):
        """
        Compare feature importance across top predictions
        
        Args:
            image: Original image
            explanation: LIME explanation
            class_names: List of class names
            top_k: Number of top predictions to show
            save_path: Path to save figure
        """
        
        # Get top labels
        probs = explanation.top_labels[:top_k]
        
        fig, axes = plt.subplots(2, top_k, figsize=(4 * top_k, 8))
        
        for idx, label in enumerate(probs):
            # Get explanation for this label
            temp, mask = explanation.get_image_and_mask(
                label,
                positive_only=True,
                num_features=5,
                hide_rest=False
            )
            
            # Top row: boundaries
            axes[0, idx].imshow(mark_boundaries(temp / 255.0, mask))
            axes[0, idx].set_title(f'{class_names[label]}',
                                   fontsize=12, fontweight='bold')
            axes[0, idx].axis('off')
            
            # Bottom row: mask
            axes[1, idx].imshow(mask, cmap='RdYlGn')
            axes[1, idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def batch_lime_explanation(
    model: torch.nn.Module,
    images: list,
    preprocess_fn: Callable,
    class_names: list,
    device: torch.device = torch.device('cuda'),
    num_samples: int = 500,
    save_dir: Optional[str] = None
):
    """
    Generate LIME explanations for a batch of images
    
    Args:
        model: PyTorch model
        images: List of image arrays
        preprocess_fn: Preprocessing function
        class_names: List of class names
        device: Device
        num_samples: Number of LIME samples
        save_dir: Directory to save figures
    """
    
    explainer = LIMEExplainer(model, preprocess_fn, device)
    
    for idx, image in enumerate(images):
        print(f"Processing image {idx + 1}/{len(images)}")
        
        # Get prediction
        pil_img = Image.fromarray(image.astype('uint8'))
        tensor = preprocess_fn(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(tensor)
            probs = F.softmax(output, dim=1)[0]
            pred_class = output.argmax(dim=1).item()
        
        # Generate explanation
        explanation = explainer.explain_instance(
            image,
            top_labels=3,
            num_samples=num_samples
        )
        
        # Visualize
        save_path = f"{save_dir}/lime_explanation_{idx}.png" if save_dir else None
        explainer.visualize_explanation(
            image,
            explanation,
            pred_class,
            save_path=save_path
        )


if __name__ == "__main__":
    # Test LIME explainer
    import torchvision.models as models
    import torchvision.transforms as transforms
    
    # Load model
    model = models.resnet50(pretrained=True)
    model.eval()
    
    # Preprocessing function
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load image
    img_path = "test_image.jpg"
    image = np.array(Image.open(img_path).resize((224, 224)))
    
    # Create explainer
    explainer = LIMEExplainer(model, preprocess)
    
    # Generate explanation
    explanation = explainer.explain_instance(image, num_samples=1000)
    
    # Visualize
    class_names = [f"Class {i}" for i in range(1000)]
    explainer.visualize_explanation(image, explanation, label=0)
