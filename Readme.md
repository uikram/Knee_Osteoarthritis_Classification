# 🦴 Knee Osteoarthritis Severity Grading with Explainable AI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Deep Learning Framework for Automated Knee Osteoarthritis Severity Classification with Integrated Explainable AI and LLM-Powered Medical Reporting**

## 🎯 Project Overview

This project develops a state-of-the-art deep learning system for automated grading of knee osteoarthritis (OA) severity from X-ray images using the Kellgren-Lawrence (KL) grading scale (0-4). Unlike traditional "black-box" models, our system integrates multiple Explainable AI (XAI) techniques and Large Language Model (LLM) integration to generate human-readable diagnostic reports, making it suitable for clinical adoption.

### 🌟 Key Features

- **8+ Deep Learning Architectures**: ResNet, EfficientNet, Vision Transformer, ConvNeXt, Swin, DenseNet
- **Advanced XAI Integration**: Grad-CAM, Grad-CAM++, LRP, LIME with comparative analysis
- **LLM-Powered Reporting**: Automated medical report generation using GPT-4 API
- **Production-Ready Web App**: Interactive Flask/Streamlit interface for clinical deployment
- **Comprehensive Experimentation**: 15+ model variants with extensive ablation studies
- **Clinical Validation**: Multi-metric evaluation optimized for false negative minimization

## 📊 Dataset

**Knee Osteoarthritis Severity Grading Dataset**  
- **Source**: Mendeley Data (Pingjun Chen, 2018)  
- **DOI**: 10.17632/56rmx5bjcr.1  
- **Samples**: 8,164 knee X-ray images  
- **Classes**: 5 (KL Grades 0-4)  
- **Distribution**: 
  - Grade 0 (Healthy): 3,218 images
  - Grade 1 (Doubtful): 1,476 images
  - Grade 2 (Minimal): 2,142 images
  - Grade 3 (Moderate): 1,079 images
  - Grade 4 (Severe): 249 images

## 🚀 Quick Start

### Installation

