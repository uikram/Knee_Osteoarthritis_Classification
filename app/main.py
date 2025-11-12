"""
Flask web application for Knee OA Classification
Production-ready REST API with interactive web interface
"""

import os
import sys
import io
import base64
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.custom_architectures import build_model
from src.explainability.gradcam import GradCAM, overlay_heatmap, get_target_layer
from src.llm.report_generator import MedicalReportGenerator, analyze_gradcam_for_prompt
from src.data.dataset import get_transforms

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Create necessary directories
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['RESULTS_FOLDER']).mkdir(exist_ok=True)

# Global variables for model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
gradcam = None
report_generator = None
transform = None

# Class names
CLASS_NAMES = [
    'Grade 0 (Healthy)',
    'Grade 1 (Doubtful)',
    'Grade 2 (Minimal)',
    'Grade 3 (Moderate)',
    'Grade 4 (Severe)'
]


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    """Load pretrained model"""
    global model, gradcam, report_generator, transform
    
    print("Loading model...")
    
    # Model configuration
    config = {
        'model_type': 'efficientnet',
        'model_name': 'efficientnet_b4',
        'num_classes': 5,
        'pretrained': False,
        'use_ordinal': False,
        'use_attention': True
    }
    
    # Build model
    model = build_model(config)
    
    # Load checkpoint
    checkpoint_path = 'models/saved_models/best_model.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Using randomly initialized weights.")
    
    model.to(device)
    model.eval()
    
    # Initialize Grad-CAM
    target_layer = get_target_layer(model, 'efficientnet')
    gradcam = GradCAM(model, target_layer)
    
    # Initialize report generator
    try:
        report_generator = MedicalReportGenerator()
        print("LLM report generator initialized")
    except Exception as e:
        print(f"Warning: Could not initialize LLM generator: {e}")
        report_generator = None
    
    # Get transforms
    transform = get_transforms(image_size=224, augment=False)
    
    print("Model loaded successfully!")


def preprocess_image(image_file):
    """Preprocess uploaded image"""
    # Read image
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Convert to numpy
    image_np = np.array(image)
    
    # Apply transforms
    augmented = transform(image=image_np)
    image_tensor = augmented['image'].unsqueeze(0)
    
    return image_tensor, image_np


def predict_and_explain(image_tensor, original_image):
    """
    Make prediction and generate explanations
    
    Args:
        image_tensor: Preprocessed image tensor
        original_image: Original image for visualization
    
    Returns:
        Dictionary with prediction results
    """
    
    # Move to device
    image_tensor = image_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)[0]
        pred_class = output.argmax(dim=1).item()
        confidence = probs[pred_class].item()
    
    # Get top-k probabilities
    top_k_probs = {
        i: probs[i].item() for i in range(len(CLASS_NAMES))
    }
    
    # Generate Grad-CAM
    cam = gradcam.generate_cam(image_tensor, pred_class)
    
    # Resize original image to match CAM
    if original_image.shape[:2] != (224, 224):
        original_resized = cv2.resize(original_image, (224, 224))
    else:
        original_resized = original_image
    
    # Overlay heatmap
    overlayed = overlay_heatmap(original_resized, cam, alpha=0.5)
    
    # Analyze Grad-CAM for report
    gradcam_findings = analyze_gradcam_for_prompt(cam)
    
    # Generate LLM report (if available)
    report = None
    if report_generator is not None:
        try:
            report = report_generator.generate_report(
                prediction=pred_class,
                confidence=confidence,
                top_k_probs=top_k_probs,
                gradcam_findings=gradcam_findings
            )
        except Exception as e:
            print(f"Error generating report: {e}")
    
    # Convert images to base64 for JSON response
    def img_to_base64(img):
        pil_img = Image.fromarray(img.astype('uint8'))
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    
    results = {
        'prediction': {
            'class': pred_class,
            'class_name': CLASS_NAMES[pred_class],
            'confidence': float(confidence)
        },
        'probabilities': [
            {
                'class': i,
                'class_name': CLASS_NAMES[i],
                'probability': float(prob)
            }
            for i, prob in enumerate(probs.cpu().numpy())
        ],
        'gradcam': {
            'heatmap': img_to_base64(np.uint8(cam * 255)),
            'overlay': img_to_base64(overlayed),
            'findings': gradcam_findings
        },
        'report': report,
        'timestamp': datetime.now().isoformat()
    }
    
    return results


# ============= API ROUTES =============

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    
    Expects:
        - file: Image file
        - patient_info (optional): JSON with patient metadata
    
    Returns:
        JSON with prediction results and explanations
    """
    
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Preprocess image
        image_tensor, original_image = preprocess_image(file)
        
        # Get prediction and explanations
        results = predict_and_explain(image_tensor, original_image)
        
        # Add patient info if provided
        if 'patient_info' in request.form:
            import json
            patient_info = json.loads(request.form['patient_info'])
            results['patient_info'] = patient_info
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint
    
    Expects:
        Multiple files with key 'files[]'
    
    Returns:
        JSON with list of prediction results
    """
    
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files[]')
    
    if len(files) == 0:
        return jsonify({'error': 'Empty file list'}), 400
    
    results_list = []
    
    for idx, file in enumerate(files):
        if not allowed_file(file.filename):
            results_list.append({
                'filename': file.filename,
                'error': 'Invalid file type'
            })
            continue
        
        try:
            # Preprocess
            image_tensor, original_image = preprocess_image(file)
            
            # Predict
            results = predict_and_explain(image_tensor, original_image)
            results['filename'] = file.filename
            results['index'] = idx
            
            results_list.append(results)
        
        except Exception as e:
            results_list.append({
                'filename': file.filename,
                'error': str(e)
            })
    
    return jsonify({
        'results': results_list,
        'total': len(files),
        'successful': len([r for r in results_list if 'error' not in r])
    })


@app.route('/api/models', methods=['GET'])
def list_models():
    """List available models"""
    models_dir = Path('models/saved_models')
    
    if not models_dir.exists():
        return jsonify({'models': []})
    
    models_list = [
        {
            'name': p.stem,
            'path': str(p),
            'size_mb': p.stat().st_size / (1024 * 1024)
        }
        for p in models_dir.glob('*.pth')
    ]
    
    return jsonify({'models': models_list})


# ============= MAIN =============

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
