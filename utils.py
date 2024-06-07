import torch
import torch.nn as nn
import os
import numpy as np
import uuid
import matplotlib.pyplot as plt

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            
            x = x.to(device)
            y = y.to(device)
            
            predictions = model(x)
            predictions = (predictions > 0.5).float()
            num_correct += (predictions == y).sum()
            num_pixels += torch.numel(predictions)
            
            dice_score = dice_coefficient(torch.tensor(y.cpu()), torch.tensor(predictions.cpu().detach().numpy()))
    
    print(f'Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}')
    print(f'Dice score: {dice_score/len(loader)}')
    model.train()
    

def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = y_true.flatten()  
    y_pred_f = y_pred.flatten()
    intersection = torch.sum(y_true_f * y_pred_f)
    union = torch.sum(y_true_f) + torch.sum(y_pred_f) - intersection
    return (2. * intersection + smooth) / (union + smooth)

def bce_dice_loss(y_true, y_pred, smooth=1):
    bce = nn.BCELoss()
    bce_loss = bce(y_pred, y_true) 
    dice_loss = (1 - dice_coefficient(y_true, y_pred, smooth))
    return 0.2 * bce_loss + 0.8 * dice_loss

def binary_accuracy(y_true, y_pred):
    y_pred = (y_pred > 0.5).float()
    return (y_pred == y_true).float().mean()
    
def save_model(model, val_loss, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, f"best_model_{val_loss:.4f}.pth")
    torch.save(model.state_dict(), model_path)
    
    print(f"Model saved to {model_path}")


def check_accuracy_validation(images, model):
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        preds = model(images)
        preds = (preds > 0.5).float()
        all_predictions.append(preds.cpu().detach().numpy())
    
    model.train()
    return np.concatenate(all_predictions, axis=0)

def visualize_segmentation(images, mask_preds, mask_trues, save_dir='segmentation_results'):
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(len(images)):
        image = images[i]
        mask_pred = mask_preds[i].squeeze()
        mask_true = mask_trues[i].squeeze()
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(mask_pred, cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(mask_true, cmap='gray')
        plt.title('True Mask')
        plt.axis('off')
        
        unique_id = uuid.uuid4() 
        save_path = os.path.join(save_dir, f'segmentation_{i}_{unique_id}.png')
        plt.savefig(save_path)
        
        plt.close()  

