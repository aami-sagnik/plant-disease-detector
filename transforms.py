from torchvision import transforms

train_data_transform = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.5,
                           contrast=0.5,
                           saturation=0.2,
                           hue=0.1),
    transforms.ToTensor()
])

eval_data_transform = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor()
])