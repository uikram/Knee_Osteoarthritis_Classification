"""
LLM-powered medical report generation for OA classification
Uses GPT-4 API to generate human-readable diagnostic reports
"""

import os
import json
from typing import Dict, List, Optional, Tuple
import numpy as np
import openai
from pathlib import Path
from datetime import datetime


class MedicalReportGenerator:
    """
    Generate medical reports using LLM based on model predictions and XAI outputs
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.3
    ):
        """
        Args:
            api_key: OpenAI API key (defaults to environment variable)
            model: GPT model to use
            temperature: Sampling temperature (lower = more deterministic)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        openai.api_key = self.api_key
        self.model = model
        self.temperature = temperature
        
        # KL grading descriptions
        self.kl_descriptions = {
            0: "No radiographic features of osteoarthritis",
            1: "Doubtful narrowing of joint space and possible osteophytic lipping",
            2: "Definite osteophytes and possible narrowing of joint space",
            3: "Moderate multiple osteophytes, definite narrowing of joint space, some sclerosis and possible deformity of bone contour",
            4: "Large osteophytes, marked narrowing of joint space, severe sclerosis and definite deformity of bone contour"
        }
        
        # Clinical recommendations
        self.recommendations = {
            0: [
                "No immediate treatment required",
                "Encourage regular physical activity and weight management",
                "Routine follow-up in 2-3 years or if symptoms develop"
            ],
            1: [
                "Monitor for symptom progression",
                "Recommend lifestyle modifications (exercise, weight management)",
                "Follow-up in 12-18 months",
                "Consider prophylactic measures if risk factors present"
            ],
            2: [
                "Initiate conservative management",
                "Recommend physical therapy and strengthening exercises",
                "Consider NSAIDs for symptom management if appropriate",
                "Follow-up in 6-12 months",
                "Patient education on disease progression"
            ],
            3: [
                "Comprehensive pain management strategy",
                "Referral to orthopedic specialist recommended",
                "Consider intra-articular injections (corticosteroids/hyaluronic acid)",
                "Evaluate for assistive devices (knee brace, walking aids)",
                "Follow-up in 3-6 months",
                "Discuss surgical consultation if conservative treatment fails"
            ],
            4: [
                "Urgent orthopedic referral required",
                "Evaluate candidacy for total knee arthroplasty",
                "Aggressive pain management and functional support",
                "Short-term follow-up (1-3 months) until surgical decision",
                "Patient counseling on surgical options and outcomes"
            ]
        }
    
    def create_prompt(
        self,
        prediction: int,
        confidence: float,
        top_k_probs: Dict[int, float],
        gradcam_findings: str,
        patient_info: Optional[Dict] = None
    ) -> str:
        """
        Create comprehensive prompt for LLM
        
        Args:
            prediction: Predicted KL grade (0-4)
            confidence: Prediction confidence
            top_k_probs: Top-k class probabilities
            gradcam_findings: Description of Grad-CAM findings
            patient_info: Optional patient metadata
        
        Returns:
            Formatted prompt string
        """
        
        # Build patient info section
        patient_section = ""
        if patient_info:
            patient_section = f"""
Patient Information:
- Patient ID: {patient_info.get('id', 'N/A')}
- Age: {patient_info.get('age', 'N/A')}
- Gender: {patient_info.get('gender', 'N/A')}
- BMI: {patient_info.get('bmi', 'N/A')}
- Knee Side: {patient_info.get('knee_side', 'N/A')}
"""
        
        # Build probability distribution
        prob_text = "\n".join([
            f"  - Grade {grade}: {prob:.1%}"
            for grade, prob in sorted(top_k_probs.items())
        ])
        
        prompt = f"""You are an expert radiologist specialized in musculoskeletal imaging. Generate a comprehensive, professional medical report for knee osteoarthritis assessment based on the following AI-assisted analysis:

{patient_section}

AI Classification Results:
- Predicted Kellgren-Lawrence Grade: {prediction}
- Prediction Confidence: {confidence:.1%}
- Probability Distribution:
{prob_text}

KL Grade {prediction} Definition:
{self.kl_descriptions[prediction]}

Explainable AI Findings:
{gradcam_findings}

Please generate a structured medical report with the following sections:

1. CLINICAL IMPRESSION
   - Summarize the primary finding (KL Grade) with confidence level
   - Note any diagnostic uncertainties or alternative possibilities

2. DETAILED FINDINGS
   - Describe key radiographic features identified by the AI model
   - Reference the attention regions highlighted by the explainability analysis
   - Include specific observations about:
     * Joint space narrowing
     * Osteophyte formation
     * Subchondral sclerosis
     * Bone deformities

3. COMPARISON TO STANDARD CRITERIA
   - How do the findings align with Kellgren-Lawrence grading criteria?
   - Are there any atypical features?

4. CLINICAL CORRELATION
   - What symptoms might the patient be experiencing at this grade?
   - Risk factors and disease progression considerations

5. RECOMMENDATIONS
   - Treatment recommendations appropriate for this KL grade
   - Follow-up timeline
   - Specialist referrals if needed
   - Additional imaging or tests to consider

6. LIMITATIONS
   - Note that this is an AI-assisted preliminary assessment
   - Emphasize the need for radiologist review and clinical correlation
   - Mention any factors that might affect diagnostic accuracy

Use professional medical terminology while maintaining clarity. Be specific about the confidence level and acknowledge uncertainty where appropriate. The report should be suitable for inclusion in a patient's medical record."""

        return prompt
    
    def generate_report(
        self,
        prediction: int,
        confidence: float,
        top_k_probs: Dict[int, float],
        gradcam_findings: str,
        patient_info: Optional[Dict] = None,
        max_tokens: int = 1500
    ) -> str:
        """
        Generate medical report using GPT-4
        
        Args:
            prediction: Predicted KL grade
            confidence: Prediction confidence
            top_k_probs: Top-k probabilities
            gradcam_findings: Grad-CAM analysis
            patient_info: Patient metadata
            max_tokens: Maximum response length
        
        Returns:
            Generated medical report
        """
        
        prompt = self.create_prompt(
            prediction, confidence, top_k_probs,
            gradcam_findings, patient_info
        )
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert radiologist generating professional medical reports."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=max_tokens,
                top_p=0.9,
                frequency_penalty=0.3,
                presence_penalty=0.3
            )
            
            report = response.choices[0].message.content.strip()
            return report
        
        except Exception as e:
            print(f"Error generating report: {e}")
            return self._generate_fallback_report(
                prediction, confidence, top_k_probs,
                gradcam_findings, patient_info
            )
    
    def _generate_fallback_report(
        self,
        prediction: int,
        confidence: float,
        top_k_probs: Dict[int, float],
        gradcam_findings: str,
        patient_info: Optional[Dict] = None
    ) -> str:
        """Generate rule-based report if LLM fails"""
        
        report = f"""
KNEE OSTEOARTHRITIS ASSESSMENT REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Report Type: AI-Assisted Preliminary Analysis

{'='*60}

CLINICAL IMPRESSION:
The AI model has classified this knee X-ray as Kellgren-Lawrence Grade {prediction} 
with {confidence:.1%} confidence. {self.kl_descriptions[prediction]}

PROBABILITY DISTRIBUTION:
"""
        for grade, prob in sorted(top_k_probs.items()):
            report += f"  Grade {grade}: {prob:.1%}\n"
        
        report += f"""
EXPLAINABLE AI FINDINGS:
{gradcam_findings}

RECOMMENDATIONS:
"""
        for rec in self.recommendations[prediction]:
            report += f"  • {rec}\n"
        
        report += """
LIMITATIONS:
This is an AI-assisted preliminary assessment. Final diagnosis must be confirmed 
by a qualified radiologist with clinical correlation. The AI model has been 
trained on standard posteroanterior knee X-rays and may not generalize to 
atypical presentations or imaging protocols.

{'='*60}
"""
        return report
    
    def save_report(
        self,
        report: str,
        save_path: str,
        metadata: Optional[Dict] = None
    ):
        """
        Save report to file with optional metadata
        
        Args:
            report: Generated report text
            save_path: Path to save report
            metadata: Additional metadata to save
        """
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as text
        with open(save_path, 'w') as f:
            f.write(report)
        
        # Save metadata as JSON
        if metadata:
            metadata_path = save_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"Report saved to: {save_path}")
    
    def batch_generate_reports(
        self,
        predictions: List[Dict],
        save_dir: str
    ) -> List[str]:
        """
        Generate reports for multiple predictions
        
        Args:
            predictions: List of prediction dictionaries
            save_dir: Directory to save reports
        
        Returns:
            List of generated reports
        """
        
        reports = []
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, pred_dict in enumerate(predictions):
            print(f"Generating report {idx + 1}/{len(predictions)}...")
            
            report = self.generate_report(
                prediction=pred_dict['prediction'],
                confidence=pred_dict['confidence'],
                top_k_probs=pred_dict['probabilities'],
                gradcam_findings=pred_dict['gradcam_findings'],
                patient_info=pred_dict.get('patient_info')
            )
            
            # Save report
            save_path = save_dir / f"report_{idx:04d}.txt"
            self.save_report(
                report,
                str(save_path),
                metadata=pred_dict
            )
            
            reports.append(report)
        
        return reports


def analyze_gradcam_for_prompt(cam_heatmap: np.ndarray) -> str:
    """
    Analyze Grad-CAM heatmap and generate textual description
    
    Args:
        cam_heatmap: Grad-CAM heatmap (H, W)
    
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
    # Test report generation
    
    # Example prediction data
    prediction_data = {
        'prediction': 3,
        'confidence': 0.82,
        'probabilities': {
            0: 0.02,
            1: 0.05,
            2: 0.11,
            3: 0.82,
            4: 0.00
        },
        'gradcam_findings': """The model's attention (Grad-CAM) highlights:
  - High attention in medial compartment
  - Peak attention at joint space region
  - 45.3% of image shows high model attention
  - Predominantly medial compartment involvement""",
        'patient_info': {
            'id': 'P12345',
            'age': 62,
            'gender': 'Female',
            'bmi': 28.5,
            'knee_side': 'Right'
        }
    }
    
    # Initialize generator
    # Note: Set your OpenAI API key as environment variable
    try:
        generator = MedicalReportGenerator()
        
        # Generate report
        report = generator.generate_report(
            prediction=prediction_data['prediction'],
            confidence=prediction_data['confidence'],
            top_k_probs=prediction_data['probabilities'],
            gradcam_findings=prediction_data['gradcam_findings'],
            patient_info=prediction_data['patient_info']
        )
        
        print(report)
        
        # Save report
        generator.save_report(
            report,
            'reports/sample_report.txt',
            metadata=prediction_data
        )
    
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set OPENAI_API_KEY environment variable")
