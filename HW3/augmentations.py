from torchvision import transforms


def basic_augmentations():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(30, padding=4),
        transforms.ToTensor(),
    ])


def advanced_augmentations():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(30, scale=(0.8, 1.0)),
        transforms.ToTensor(),
    ])


def grayscale_augmentations():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])


augmentations = {
    'basic': basic_augmentations,
    'advanced': advanced_augmentations,
    'grayscale': grayscale_augmentations
}
