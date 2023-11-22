import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transform():
    transform = [
        A.PadIfNeeded(min_height=224, min_width=224, border_mode=0),
        A.Resize(height=224, width=224),
        A.Normalize(),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Perspective(scale=(0.05, 0.1), p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        ToTensorV2(),
    ]
    return A.Compose(transform)


def get_val_transform():
    transform = [
        A.PadIfNeeded(min_height=224, min_width=224, border_mode=0),
        A.Resize(height=224, width=224),
        A.Normalize(),
        ToTensorV2(),
    ]
    return A.Compose(transform)
