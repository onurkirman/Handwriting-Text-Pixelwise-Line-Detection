
import torch
import torchvision
from timeit import default_timer as timer
import sys

class Validation: 
    def __init__(self, data_loader, device, criterion):
        self.data_loader = data_loader
        self.device = device
        self.criterion = criterion
    
    # Validation Method for the model
    def validate(self, model):
        val_loss = 0
        correct_pixel = 0
        total_pixel = 0

        for images, masks, _ in self.data_loader:
            images = images.to(self.device)
            masks = masks.type(torch.LongTensor)
            masks = masks.reshape(masks.shape[0], masks.shape[2], masks.shape[3])
            masks = masks.to(self.device)

            outputs = model(images)
            val_loss += self.criterion(outputs, masks).item()

            _, predicted = torch.max(outputs.data, 1)
            correct_pixel += (predicted == masks).sum().item()

            b, h, w = masks.shape
            batch_total_pixel = b*h*w

            total_pixel += batch_total_pixel

        acc = correct_pixel/total_pixel
        return val_loss, acc, len(self.data_loader)

