import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

train_images_path = "/home/nikolay/ML/It_Jim/eyeglasses_dataset/train/images"
train_masks_path = "/home/nikolay/ML/It_Jim/eyeglasses_dataset/train/masks"

validation_images_path = "/home/nikolay/ML/It_Jim/eyeglasses_dataset/val/images"
validation_masks_path = "/home/nikolay/ML/It_Jim/eyeglasses_dataset/cal/masks"

height = 512
width = 512

class EyeGlassesDataset(Dataset):
    def __init__(self, images_path, masks_path, transform=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.image_names = sorted(os.listdir(self.images_path))
        self.mask_names = sorted(os.listdir(self.masks_path))
        
        image_prefixes = [os.path.splitext(image_name)[0].split('_')[0] for image_name in self.image_names]
        mask_prefixes = [os.path.splitext(mask_name)[0].split('_')[0] for mask_name in self.mask_names]
        
        for i, (image_prefix, mask_prefix) in enumerate(zip(image_prefixes, mask_prefixes)):
            print(f"Comparing image prefix: {image_prefix} with mask prefix: {mask_prefix}, general_number_of_pairs: {i}")
            if image_prefix != mask_prefix:
                raise ValueError("Image and mask prefixes do not match")
                    
        if len(self.image_names) != len(self.mask_names):
            raise ValueError("Number of images and masks are not equal")
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = os.path.join(self.images_path, image_name)
        
        if image_name.endswith(".jpg") or image_name.endswith(".png"):
            image = Image.open(image_path).convert('RGB')
        
        mask_name = self.mask_names[index]
        mask_path = os.path.join(self.masks_path, mask_name)
        if mask_name.endswith(".jpg") or mask_name.endswith(".png"):
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask, dtype=np.float32)
            mask[mask == 255.0] = 1.0
            
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(Image.fromarray(mask.astype(np.uint8)))
            mask = (mask > 0).float()
        
        return image, mask
    
transformss = transforms.Compose([
    transforms.Resize((height, width)),
    transforms.ToTensor()

])

if __name__ == "__main__":
    dataset = EyeGlassesDataset(train_images_path, train_masks_path, transform=transformss)
    print(f'Dataset length: {len(dataset)}')

    image, mask = dataset[0]
    print(f'image shape: {image.shape}')
    print(f'mask shape: {mask.shape}')
    print(torch.cuda.is_available())
    
    print(image)
    print(mask.min(), mask.max())

    plt.imshow(image.permute(1, 2, 0))
    plt.title('image')
    plt.show()
    
    plt.imshow(mask.squeeze(), cmap='gray')
    plt.title('mask')
    plt.show()

