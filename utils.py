
from torchvision import transforms

def train_transform(config):
    return transforms.Compose([
        transforms.Resize([config.img_size, config.img_size]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-135, 135)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485], std=[0.5])
    ])