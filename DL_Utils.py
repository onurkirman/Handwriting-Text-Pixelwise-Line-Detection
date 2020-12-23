''' 
    Onur Kirman S009958 Computer Science Undergrad at Ozyegin University
'''

import glob
import os
import random
import re
import sys
import time
from timeit import default_timer as timer

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from torchvision import transforms

from models.CNN_network import Network
from models.Unet_model import UnetModel
from models.Unet_model_clipped import UnetModelClipped


# Plots the given batch in 3 rows; Raw, Mask, Bitwise_Anded
def plt_images(images, masks, batch_size):
    fig, axs = plt.subplots(3, batch_size, figsize=(images[0].shape))

    for i in range(len(images)):
        axs[0][i].imshow(images[i], cmap='gray')
        axs[1][i].imshow(masks[i], cmap='gray')
        axs[2][i].imshow(images[i] & masks[i], cmap='gray')
    fig.suptitle("Top Row: raw images, Middle Row: masks, Bottom Row: bitwise_and masks")
    plt.show()


# Returns the images and masks in the original format
def undo_preprocess(images, predicts):
    x = []
    y = []

    images = images.cpu().numpy()
    predicts = predicts.cpu().numpy()

    for index in range(images.shape[0]):
        image = images[index]
        # Needed to convert c,h,w -> h,w,c
        image = np.transpose(image, (1, 2, 0))
        # make every pixel 0-1 range than mul. 255 to scale the value
        image = np.squeeze(image) * 255
        x.append(image.astype(np.uint8))

        predict = predicts[index]
        # Needed to convert c,h,w -> h,w,c
        mask_array = np.transpose(predict, (1, 2, 0))
        # Every pixel has two class grad, so we pick the highest
        mask_array = np.argmax(mask_array, axis=2) * 255
        mask_array = mask_array.astype(np.uint8)
        y.append(mask_array)
    return np.array(x), np.array(y)


# Saves the given batch in directory
def save_output_batch(images, outputs):
    path = os.path.join(os.getcwd(), 'output_batch\\')
    os.makedirs(path, exist_ok=True)
    print(f'You can find samples in \'{path}\'')

    for index in range(len(images)):
        image = images[index]
        save_image = Image.fromarray(image)
        save_image.save(path + str(index) + '_input.png')

        mask = outputs[index]
        save_mask = Image.fromarray(mask)
        save_mask.save(path + str(index) + '_output.png')


# Saves the given batch in directory. By default it is set to be '/output'
def save_predictions(images, predictions, filenames, path='output'):
    path = os.path.join(os.getcwd(), path)
    
    form_path = os.path.join(path, 'form')
    pred_path = os.path.join(path, 'mask')
    
    os.makedirs(form_path, exist_ok=True)
    os.makedirs(pred_path, exist_ok=True)

    for idx, (image, prediction) in enumerate(zip(images, predictions)):
        save_image = Image.fromarray(image)
        save_image.save(os.path.join(form_path , str(filenames[idx])))

        save_prediction = Image.fromarray(prediction)
        save_prediction.save(os.path.join(pred_path , str(filenames[idx])))
    print(f'You can find predictions in \'{path}\'')


# Loads the data from the given path
def load_data(dataset_path):
    forms = []
    masks = []
    filenames = []

    # sample path -> './dataset/train/form/*.png'
    form_names = glob.glob('./' + dataset_path + '/form' + '/*.png')
    # Sorts them as 0,1,2..
    form_names.sort(key=lambda f: int(re.sub('\D', '', f)))

    # sample path -> './dataset/train/mask/*.png'
    mask_names = glob.glob('./' + dataset_path + '/mask' + '/*.png')
    # Sorts them as 0,1,2..
    mask_names.sort(key=lambda f: int(re.sub('\D', '', f)))

    for i, (form_name, mask_name) in enumerate(zip(form_names, mask_names)):
        form = np.asarray(Image.open(form_name))
        mask = np.asarray(Image.open(mask_name))

        forms.append(form)
        masks.append(mask)
        filenames.append(form_name.split('\\')[-1])

    return np.array(forms), np.array(masks), np.array(filenames)


# DATASET CLASS
class FormDS(Dataset):
    def __init__(self, path, number_of_classes: int, augmentation=False):
        images, masks, filenames = load_data(path)
        self.images = images
        self.masks = masks
        self.filenames = filenames
        self.number_of_classes = number_of_classes
        self.length = len(images)
        self.augmentation = augmentation

    # Converts the image, a PIL image, into a PyTorch Tensor
    def transform(self, image, mask):
        # needed to apply transforms
        image = transforms.ToPILImage()(image)
        mask = transforms.ToPILImage()(mask)

        if self.augmentation:
            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random vertical flipping
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            
            # # Random rotation on clockwise or anticlockwise
            # if random.random() > 0.5:
            #     rotate_direction = [-1, 1] # -1 -> anticlockwise 
            #     angle = 90 * random.choice(rotate_direction)
            #     image = TF.rotate(image, angle)
            #     mask = TF.rotate(mask, angle)

            # # Resize -> Surprisingly decreasing our accuracy!
            # resize_image = transforms.Resize(size=(312, 312))
            # resize_mask = transforms.Resize(size=(312, 312), interpolation=Image.NEAREST)
            # image = resize_image(image)
            # mask = resize_mask(mask) # -> needs to be exactly 0-1
            # # Random Crop
            # i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(256, 256))
            # image = TF.crop(image, i, j, h, w)
            # mask = TF.crop(mask, i, j, h, w)


        # # use this in case of observation need     -> will be deleted!
        # im = np.array(image, dtype='float32')
        # ma = np.array(mask, dtype=int)
        # im = im * 255
        # ma = ma * 255
        # im = Image.fromarray(np.uint8(im))
        # ma = Image.fromarray(np.uint8(ma))
        # im.show()
        # ma.show()


        # Swaps color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # Transform to tensor
        img = TF.to_tensor(np.array(image))
        msk = TF.to_tensor(np.array(mask))
        return img, msk

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        image = self.images[idx]
        image = image.astype(np.float32)
        image = image / 255  # make pixel values between 0-1

        mask = self.masks[idx]
        mask = mask.astype(np.float32)
        mask = mask / 255   # make pixel values 0-1

        image, mask = self.transform(image, mask)

        return image, mask, filename

    def __len__(self):
        return self.length


# Builds model as it is specified
def build_model(name, device, number_of_classes, dropout_rate=0):
    """
    Builds a model from one of the provided models.
    Currently provided models are specified as:
    Unet_model, Unet_model_clipped and CNN_network
    number_of_classes: Number of class that network needs to understand 
                       which also used at the output layer
    dropout_rate: The rate for dropouts in the given model. It is 0 by default
    """
    if name is 'unet':
        model = UnetModel(number_of_classes, dropout_rate).to(device)
    elif name is 'unet_small':
        model = UnetModelClipped(number_of_classes, dropout_rate).to(device)
    elif name is 'network':
        model = Network(number_of_classes, dropout_rate).to(device)
    else:
        sys.exit('You need to choose one network model that is being provided. Stopping...')
    return model


# Loads data into torch
def torch_loader(path, number_of_classes, batch_size, augmentation=True):
    dataset = FormDS(path, number_of_classes, augmentation=augmentation)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f'{path.split("/")[-1].capitalize()} DS Size: {len(dataset)} ({len(data_loader)} batches)')
    return data_loader


# Validation Class
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


# Train Class
class Train:
    def __init__(self, data_loader, device, criterion, optimizer, validation=None, scheduler=None):
        self.data_loader = data_loader 
        self.device = device 
        self.criterion = criterion 
        self.optimizer = optimizer 
        self.validation = validation 
        self.scheduler = scheduler
    
    # Training of the Model
    def start(self, model, epochs, batch_extender, validation_on, scheduler_on, loss_print_per_epoch):
        if validation_on and self.validation is None:
            sys.exit('Validation function is not provided. It cannot validate!')
        
        if scheduler_on and self.scheduler is None:
            sys.exit('No scheduler provided. Create instance with a scheduler!')

        total_steps = len(self.data_loader)
        print(f"{epochs} Epochs & {total_steps} Total Steps per Epoch")

        start_time = timer()
        print(f'Training Started in {start_time} sec.')
        model.train()
        try:
            batch_step = 0
            for epoch in range(epochs):
                for idx, (images, masks, _) in enumerate(self.data_loader, 1):
                    images = images.to(self.device)  # Sends to GPU
                    masks = masks.type(torch.LongTensor)
                    masks = masks.reshape(masks.shape[0], masks.shape[2], masks.shape[3])
                    masks = masks.to(self.device)    # Sends to GPU

                    # Forward pass
                    predicts = model(images)
                    loss = self.criterion(predicts, masks)

                    # This doubles our batch size
                    if batch_extender:
                        if batch_step == 0:
                            self.optimizer.zero_grad()
                            loss.backward()
                            batch_step = 1
                        elif batch_step == 1:
                            loss.backward()
                            self.optimizer.step()
                            batch_step = 0
                    else:
                        # Backward and optimize
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    if idx % int(total_steps/loss_print_per_epoch) == 0:
                        if validation_on:
                            acc = 0
                            # Validation Part
                            model.eval()
                            with torch.no_grad():
                                validation_loss, validation_accuracy, length_validation = self.validation.validate(model)
                            model.train()
                        valstr = f'\tValid. Loss: {(validation_loss/length_validation):.4f}\tValid. Acc.: {validation_accuracy * 100:.3f}%' if validation_on else ''
                        print(f'Epoch: {epoch + 1}/{epochs}\tSt: {idx}/{total_steps}\tLast.Loss: {loss.item():4f}{valstr}')
                if scheduler_on:
                    self.scheduler.step(acc) # -> ReduceLROnPlateau

                    # scheduler.step() # -> StepLR
                    # print(f'LR: {scheduler.get_last_lr()}')
        
        except Exception as e:
            print(e)

        print('Execution time:', '{:5.2f}'.format(timer() - start_time), 'seconds')
        return model


# Test Class
class Test:
    def __init__(self, data_loader, batch_size, device):
        super().__init__()
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.device = device
        
    # Testing of the Model
    def start(self, model, is_saving_output, sample_view, path='output'):
        all_forms = []
        all_predictions = []
        all_filenames = []
        view_count = 0

        # Test the model
        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():  # used for dropout layers
            correct_pixel = 0
            total_pixel = 0
            for images, masks, filenames in self.data_loader:
                images = images.to(self.device)
                masks = masks.type(torch.LongTensor)
                # delete color channel to compare directly with prediction
                masks = masks.reshape(masks.shape[0], masks.shape[2], masks.shape[3])
                masks = masks.to(self.device)

                predicts = model(images)
                _, predicted = torch.max(predicts.data, 1)
                correct_pixel += (predicted == masks).sum().item()

                b, h, w = masks.shape
                batch_total_pixel = b * h * w
                total_pixel += batch_total_pixel
                
                # if pre-set addes images to list
                if is_saving_output:
                    af, ap = undo_preprocess(images, predicts)
                    all_forms.extend(af)
                    all_predictions.extend(ap)
                    all_filenames.extend(filenames)

                # To observe random batch prediction uncomment!
                if sample_view and view_count < 10 and random.random() > 0.5:
                    view_count += 1
                    images, masks = undo_preprocess(images, predicts)
                    plt_images(images, masks)

            print(f"{correct_pixel} / {total_pixel}")
            print(f"Test Accuracy on the model with {len(self.data_loader) * self.batch_size} images: {100 * correct_pixel / total_pixel:.4f}%")
        
        # Saves the output
        if is_saving_output:
            save_predictions(np.array(all_forms), np.array(all_predictions), np.array(all_filenames), path)

