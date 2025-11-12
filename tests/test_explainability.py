import unittest
import torch
from src.models.custom_architectures import build_model
from src.explainability.gradcam import GradCAM, get_target_layer

class TestExplainability(unittest.TestCase):
    def test_gradcam(self):
        config = {'model_type': 'efficientnet', 'model_name': 'efficientnet_b4', 'num_classes': 5, 'pretrained': False}
        model = build_model(config)
        dummy = torch.randn(1, 3, 224, 224)
        target_layer = get_target_layer(model, 'efficientnet')
        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate_cam(dummy, 0)
        self.assertEqual(cam.shape, (7, 7))  # Adjust based on actual model output
if __name__ == '__main__':
    unittest.main()
