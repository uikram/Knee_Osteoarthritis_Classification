import unittest
import torch
from src.models.custom_architectures import build_model

class TestModels(unittest.TestCase):
    def test_forward_pass(self):
        config = {'model_type': 'efficientnet', 'model_name': 'efficientnet_b4', 'num_classes': 5, 'pretrained': False}
        model = build_model(config)
        dummy = torch.randn(2, 3, 224, 224)
        output = model(dummy)
        self.assertEqual(output.shape, (2, 5))
if __name__ == '__main__':
    unittest.main()
