"""
Custom data augmentation utilities for knee OA X-ray dataset
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_advanced_augmentations(image_size=224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.12, rotate_limit=20, p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.15, p=0.5),
        A.GaussNoise(var_limit=(10.0, 60.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.15),
        A.CoarseDropout(max_holes=7, max_height=16, max_width=16, p=0.25),
        A.CLAHE(clip_limit=2.0, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
