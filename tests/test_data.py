import unittest
from src.data.dataset import KneeOADataset, get_transforms

class TestData(unittest.TestCase):
    def test_dataloader(self):
        dataset = KneeOADataset(root_dir='data/processed', split='train', transform=get_transforms(224))
        self.assertTrue(len(dataset) > 0)
if __name__ == '__main__':
    unittest.main()
