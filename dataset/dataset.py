import os
import shutil

from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torch

def prepare_tiny_imagenet_val():
    data_dir='data/tiny-imagenet-200'
    val_dir = os.path.join(data_dir, 'val')
    
    with open(os.path.join(val_dir, 'val_annotations.txt')) as f:
        for line in f:
            fn, cls, *_ = line.split('\t') # fn: filename, cls: class ID
            os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
            
            shutil.copyfile(os.path.join(val_dir, 'images', fn), os.path.join(val_dir, cls, fn))
    
    if os.path.exists(os.path.join(val_dir, 'images')):
        shutil.rmtree(os.path.join(val_dir, 'images'))

def get_transforms():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_dataloaders(data_dir, batch_size=32, num_workers=2, subset_percentage=None):
    transform = get_transforms()
    
    train_dataset = ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform)
    
    if subset_percentage:
        num_train_samples = int(len(train_dataset) * subset_percentage)
        num_val_samples = int(len(val_dataset) * subset_percentage)
        
        train_dataset, _ = torch.utils.data.random_split(
            train_dataset, [num_train_samples, len(train_dataset) - num_train_samples]
        )
        val_dataset, _ = torch.utils.data.random_split(
            val_dataset, [num_val_samples, len(val_dataset) - num_val_samples]
        )
        
        print(f"Using {num_train_samples} samples for training")
        print(f"Using {num_val_samples} samples for validation")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader