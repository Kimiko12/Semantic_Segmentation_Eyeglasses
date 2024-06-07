import torch
import albumentations as A 
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm 
import torch.nn as nn 
import torch.optim as optim
from unet_model import Unet
import torchvision.transforms as transforms
from dataset import EyeGlassesDataset
import matplotlib.pyplot as plt 
import os
from PIL import Image
import numpy as np

from utils import (
    binary_accuracy,
    dice_coefficient,
    save_model,
    check_accuracy,
    check_accuracy_validation,
    visualize_segmentation
)

Learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
batch_size = 10
num_epochs = 15
num_workers = 2
height = 512
width = 512
Pin_memory = True
Load_model = False
train_images_path = "/home/nikolay/ML/It_Jim/Semantic_Segmentation_task_1/eyeglasses_dataset/train/images"
train_masks_path = "/home/nikolay/ML/It_Jim/Semantic_Segmentation_task_1/eyeglasses_dataset/train/masks"
validation_images_path = "/home/nikolay/ML/It_Jim/Semantic_Segmentation_task_1/eyeglasses_dataset/val/images"
validation_masks_path = "/home/nikolay/ML/It_Jim/Semantic_Segmentation_task_1/eyeglasses_dataset/val/masks"
predictions_path = "/home/nikolay/ML/It_Jim/Semantic_Segmentation_task_1/predictions"

transforms = transforms.Compose([
    transforms.Resize((height, width)),
    transforms.ToTensor()
])


def train(loader, model, optimizer, loss_function, metrics):
    epoch_loss = 0.0
    epoch_metrics = [0.0] * len(metrics)
    loop = tqdm(loader)
    
    for batch_index, (data, target) in enumerate(loop):
        data = data.to(device)
        target = target.float().to(device)
        
        predictions = model(data)
        loss = loss_function(predictions, target)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        for i, metric_function in enumerate(metrics):
            metric = metric_function(target, predictions)
            epoch_metrics[i] += metric.item()
        
        loop.set_postfix(loss=epoch_loss / (batch_index + 1),
                         metrics=[m / (batch_index + 1) for m in epoch_metrics])
        
    return epoch_loss / len(loader), [m / len(loader) for m in epoch_metrics]
    
def train_model(train_dataset, model, optimizer, loss_function, metrics, device, num_epochs, batch_size, num_workers, pin_memory):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    model.load_state_dict(torch.load('/home/nikolay/ML/It_Jim/Semantic_Segmentation_task_1/saved_models/best_model_0.0048.pth'))
    for epoch in range(num_epochs):
        model.train() 
        epoch_loss = 0.0
        metric_values = {metric.__name__: 0.0 for metric in metrics}

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            for metric in metrics:
                metric_value = metric(outputs, masks)
                metric_values[metric.__name__] += metric_value.item()
        
        epoch_loss /= len(train_loader)
        metric_values = {key: value / len(train_loader) for key, value in metric_values.items()}
        if epoch % 5 == 0:
            save_model(model, epoch_loss, save_dir="saved_models")
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Metrics: {metric_values}")

def test_model(validation_dataset, model, device, batch_size, num_workers, pin_memory):
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
    
    model.load_state_dict(torch.load('/home/nikolay/ML/It_Jim/Semantic_Segmentation_task_1/saved_models/best_model_0.0048.pth'))
    model.eval() 
    
    for idx, (image, mask) in enumerate(validation_loader):
        
        image = image.to(device)
        mask_preds = check_accuracy_validation(image, model)
        mask_trues = mask
            
        visualize_segmentation(image.cpu().numpy().transpose(0, 2, 3, 1), mask_preds, mask_trues, save_dir = predictions_path)
     

if __name__ == "__main__":
    train_dataset = EyeGlassesDataset(train_images_path, train_masks_path, transform=transforms)
    validation_dataset = EyeGlassesDataset(validation_images_path, validation_masks_path, transform=transforms)

    model = Unet(input_channels=3, output_channels=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=Learning_rate)
    loss_function = nn.BCEWithLogitsLoss()
    metrics = [dice_coefficient, binary_accuracy]
    
    # train_model(train_dataset, model, optimizer, loss_function, metrics, device, num_epochs, batch_size, num_workers, Pin_memory)


    test_model(validation_dataset, model, device, batch_size, num_workers, Pin_memory)
    
