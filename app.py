# --- START OF GPU/CONFIG SETUP ---
# This MUST be at the very top before torch is imported
import os
import config  # Imports your project's config file
os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_ID
print(f"[INFO] Setting app to use GPU: {config.GPU_ID}")
# --- END OF GPU/CONFIG SETUP ---

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import cv2
import streamlit as st

# Import our project's model builder
from model import build_model

# --- 1. Global Settings (from config) ---
DEVICE = config.DEVICE
MODEL_NAME = config.MODEL_NAME
TRAINED_MODEL_PATH = config.MODEL_SAVE_PATH
NUM_CLASSES = config.NUM_CLASSES
IMG_SIZE = config.IMG_SIZE[0] # Get 224

# --- User-Facing Class Names and Advice ---
CLASS_NAMES = [
    "0: Healthy Knee",
    "1: Mild Arthritis",
    "2: Moderate Arthritis",
    "3: Severe Arthritis",
    "4: End-Stage Arthritis"
]

DOCTOR_ADVICE_MAP = {
    "0: Healthy Knee": """
**Explanation:** This is great news! Your X-ray shows a healthy knee, with no significant signs of arthritis. This means the cartilage and joint space appear normal.

**Suggestions:**
* Maintain a healthy weight to keep pressure off your joints.
* Stay active with low-impact exercises like swimming, cycling, or walking.
* Focus on exercises that strengthen the muscles around your knee (like quadriceps and hamstrings).
* Always warm up before and stretch after exercise.

**Disclaimer:** Please remember, this is a general analysis and not medical advice. It's always best to discuss the full report with your doctor.
""",
    "1: Mild Arthritis": """
**Explanation:** The analysis suggests you have mild arthritis. This typically means there might be some minor joint space narrowing or small bone spurs, but the overall joint structure is still in good condition.

**Suggestions:**
* Focus on low-impact exercises like swimming or cycling to keep the joint mobile without straining it.
* Maintain a healthy weight to reduce stress on your knee.
* Consider physical therapy to learn exercises that strengthen the muscles supporting your knee.
* Over-the-counter anti-inflammatories (like ibuprofen) may help with occasional pain, but please ask your doctor first.

**Disclaimer:** Please remember, this is a general analysis and not medical advice. It's always best to discuss the full report with your doctor.
""",
    "2: Moderate Arthritis": """
**Explanation:** The X-ray indicates moderate arthritis. This means there is definite joint space narrowing, and bone spurs are likely visible. You may be experiencing symptoms like pain, stiffness, and some swelling, especially after activity.

**Suggestions:**
* Work with a physical therapist to create a safe exercise plan to maintain joint mobility and strength.
* Maintaining a healthy weight is very important to reduce the load on your knee.
* You might discuss pain management options with your doctor, which could include stronger medications or joint injections.
* Consider using assistive devices, like a cane or a knee brace, on days when you have more pain.

**Disclaimer:** Please remember, this is a general analysis and not medical advice. It's always best to discuss the full report with your doctor.
""",
    "3: Severe Arthritis": """
**Explanation:** This analysis shows severe arthritis. At this stage, the joint space is significantly reduced, meaning the cartilage has worn down considerably. You are likely experiencing frequent pain, significant stiffness, and a loss of mobility.

**Suggestions:**
* Pain management is a key priority. Please speak with your doctor about all available options, including medications and injections.
* A physical therapist can help you with gentle range-of-motion exercises and suggest assistive devices (like a walker) to improve safety.
* Your doctor may want to discuss surgical options, such as a partial or total knee replacement, as this is often the most effective treatment for this stage.

**Disclaimer:** Please remember, this is a general analysis and not medical advice. It's always best to discuss the full report with your doctor.
""",
    "4: End-Stage Arthritis": """
**Explanation:** This X-ray shows end-stage arthritis. This is the most advanced stage, where the joint space is largely gone, and the bones may be rubbing against each other. This typically causes constant, severe pain and major difficulty with mobility.

**Suggestions:**
* Your primary focus should be a consultation with an orthopedic surgeon to discuss your options.
* Surgical intervention, such as a total knee replacement, is the most common and effective treatment to relieve pain and restore function at this stage.
* In the meantime, your doctor will focus on a robust pain management plan.
* Using assistive devices, like a walker or wheelchair, is highly recommended to maintain safety.

**Disclaimer:** Please remember, this is a general analysis and not medical advice. It's always best to discuss the full report with your doctor.
"""
}


# --- 2. Grad-CAM Class ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradient = None
        self.activation = None
        self.hook_layers()

    def hook_layers(self):
        def save_activation(module, input, output):
            self.activation = output.detach()
        def save_gradient(module, input, output):
            self.gradient = output[0].detach()
        self.target_layer.register_forward_hook(save_activation)
        self.target_layer.register_backward_hook(save_gradient)

    def generate_cam(self, input_image, class_idx=None):
        output = self.model(input_image)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        # Use retain_graph=True for ResNet compatibility
        output[:, class_idx].backward(retain_graph=True) 
        
        gradients = self.gradient
        activation = self.activation
        
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * activation, dim=1)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy(), class_idx, output

# --- 3. Model Loading Functions ---
@st.cache_resource
def load_vision_model(model_name, model_path, num_classes):
    print(f"Loading vision model ({model_name})...")
    
    # Use our project's build_model function
    # Set pretrained=False because we are loading our own weights
    model = build_model(
        model_name=model_name, 
        pretrained=False, 
        num_classes=num_classes
    )
    
    if os.path.exists(model_path):
        try:
            # Load the trained weights
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f"Successfully loaded trained weights from {model_path}")
        except Exception as e:
            st.error(f"Error loading trained weights: {e}. Exiting.")
            sys.exit(1)
    else:
        st.error(f"Error: Trained model '{model_path}' not found. Please run your training script first.")
        sys.exit(1)
        
    model = model.to(DEVICE)
    model.eval()
    print("Vision model loaded.")
    return model

# --- 4. Main Application ---
def main():
    st.set_page_config(layout="wide")
    st.title(f"Knee Arthritis Analyzer ({MODEL_NAME})")
    print(f"Using device: {DEVICE}")

    # --- Load Models ---
    try:
        vision_model = load_vision_model(MODEL_NAME, TRAINED_MODEL_PATH, NUM_CLASSES)
        
        # --- DYNAMIC Grad-CAM Target Layer ---
        # This is now flexible and works for both models
        if MODEL_NAME == 'resnet18':
            target_layer = vision_model.layer4[-1].conv2
            print("Set Grad-CAM target layer for ResNet-18")
        elif MODEL_NAME == 'resnet101':
            target_layer = vision_model.layer4[-1].conv3
            print("Set Grad-CAM target layer for ResNet-101")
        else:
            # Fallback for other models (like resnet34, 50)
            target_layer = vision_model.layer4[-1].conv2 
            print(f"Warning: Defaulting Grad-CAM layer for {MODEL_NAME}")
            
        grad_cam = GradCAM(model=vision_model, target_layer=target_layer)
        
        # Image pre-processing transform (same as val_transforms)
        preprocess_tfm = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    except SystemExit:
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during model loading: {e}")
        st.stop()

    # --- File Uploader ---
    uploaded_file = st.file_uploader(
        "Choose a Knee X-Ray image...", 
        type=["png", "jpg", "jpeg", "bmp"]
    )

    if uploaded_file is not None:
        try:
            image_pil = Image.open(uploaded_file).convert('RGB')
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image_pil, caption="Original Image", use_container_width=True)

            with st.spinner("Analyzing, please wait..."):
                # --- 1. Preprocess Image for Model ---
                input_tensor = preprocess_tfm(image_pil).unsqueeze(0).to(DEVICE)
                
                # --- 2. Run XAI (Grad-CAM) & Get Grading ---
                cam_np, prediction_idx, output = grad_cam.generate_cam(input_tensor, class_idx=None)
                
                # --- 3. Process Grading Results ---
                probabilities = F.softmax(output, dim=1)
                confidence, _ = torch.max(probabilities, 1)
                prediction_name = CLASS_NAMES[prediction_idx]
                conf_score = confidence.item()

                # --- 4. Create Grad-CAM Overlay ---
                image_rgb_np = np.array(image_pil.resize((IMG_SIZE, IMG_SIZE))) # Resize for overlay
                cam_resized = cv2.resize(cam_np, (IMG_SIZE, IMG_SIZE))
                cam_normalized = np.uint8(255 * cam_resized)
                heatmap = cv2.applyColorMap(cam_normalized, cv2.COLORMAP_JET)
                heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                superimposed_img = cv2.addWeighted(image_rgb_np, 0.6, heatmap_rgb, 0.4, 0)
                
                # --- 5. Get Doctor Response (INSTANT) ---
                response_text = DOCTOR_ADVICE_MAP.get(prediction_name, "Error: Could not find advice for this diagnosis.")

            # --- Display Results ---
            with col2:
                st.image(superimposed_img, caption="XAI (Grad-CAM) Overlay", use_container_width=True)
            
            st.header(f"Diagnosis: {prediction_name} (Confidence: {conf_score:.2%})")
            
            st.subheader("Doctor's Advice:")
            st.markdown(response_text) 

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            print(f"Error during analysis: {e}")

# --- 5. Main Execution ---
if __name__ == "__main__":
    main()