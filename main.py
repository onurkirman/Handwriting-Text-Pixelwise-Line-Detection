''' 
    Onur Kirman S009958 Computer Science Undergrad at Ozyegin University

                                *** Notes before starting ***
    In the folder named data, we have; line_info.txt and 1539 form images from the IAM Handwriting Database
        - The line_info.txt is the reformatted version (header part deleted version) of lines.txt file
    
    Hyperparameters that can be adjusted, and the Data containing paths are listed at the top of the code

    Directory Hierarchy:
    src
        data
            - forms
            - line_info.txt
        dataset
            - train
            - test
            - validation
        output_batch -> (created at the end of main.py to save the output batch)
        utils
            - image_preprocess.py
            - lines_segmentation.py
        main.py
        CNN_network.py  -> Simple CNN Model
        Unet_model.py   -> Unet alike CNN Model

    Steps Followed:
    Load Data -> Make Dataset -> Load Dataset -> Built Model -> Train Model -> Validate -> Save Model -> Load Model -> Test Model -> Output Batch 
    
    Note that: Validation done in the Training Part if wanted

    **** Also -> Train: Overlapsing, Train2: Condition Specific Not Overlaping, Train3: Generic Not Overlaping ****

    Note for Loss Function ->  "For a binary classification you could use nn.CrossEntropyLoss() with a logit output of shape [batch_size, 2] 
                                or nn.BCELoss() with a nn.Sigmoid() in the last layer."
'''

import os
import re
import cv2
import time
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from PIL import Image
# from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

import torchvision
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchsummary import summary

from Unet_model import UnetModel
from CNN_network import Network

print("Program Started!")

print(f'Cuda Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}')
print(f'Cuda Available: {torch.cuda.is_available()}\n')


# Hyperparameters
epochs = 8                # 4 predicts well, might be 2. 8 doesn't affect much ~0.5%
batch_size = 4            # 4 is OK, might be 8 (exceed mem.)
batch_extender = True     # Extends the batch so that training process done once in twice -> gives better result
learning_rate = 1e-3      # 1e-3 is OK., 5e-4 also OK. (0.01 -> 0.001 -> 0.0005) LR Scheduler!
dropout_rate = 0.0        # 0.2 is nice with big train data
loss_print_per_epoch = 1  # desired # loss data print per epoch
number_of_classes = 2     # OK.
validation_on = False
scheduler_on = False
sample_view = True

# CUDA for PyTorch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# Image Paths
data_dir = 'dataset'
train_path = data_dir + '/train3'
test_path = data_dir + '/test3'
validation_path = data_dir + '/validation3'

# Trained Model Path
trained_model_path = 'utils\\model_check3.pt'


# Hyperparameter Print
print(f'Batch Size: {batch_size*2 if batch_extender else batch_size} {"(Artificial Batch)" if batch_extender else ""}')
print(f'Learning Rate: {learning_rate}')
print(f'Dropout  Rate: {dropout_rate}\n')


# Plots the given batch in 3 rows; Raw, Mask, Anded
def plt_images(images, masks):
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


numberofForm = 700
# Loads the data from the given path
def load_data(dataset_path):
    forms = []
    masks = []

    # sample path -> './dataset/train/form/*.png'
    form_names = glob.glob('./' + dataset_path + '/form' + '/*.png')
    # Sorts them as 0,1,2..
    form_names.sort(key=lambda f: int(re.sub('\D', '', f)))

    # sample path -> './dataset/train/mask/*.png'
    mask_names = glob.glob('./' + dataset_path + '/mask' + '/*.png')
    # Sorts them as 0,1,2..
    mask_names.sort(key=lambda f: int(re.sub('\D', '', f)))

    for i, (form_name, mask_name) in enumerate(zip(form_names, mask_names)):

        # Added to observe speed of training
        if dataset_path == train_path and i == numberofForm:
            break

        form = np.asarray(Image.open(form_name))
        mask = np.asarray(Image.open(mask_name))

        forms.append(form)
        masks.append(mask)

    return np.array(forms), np.array(masks)


# DATASET CLASS
class FormDS(Dataset):
    def __init__(self, path, number_of_classes: int, augmentation=False):
        images, masks = load_data(path)
        self.images = images
        self.masks = masks
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
            
            # Random rotation on clockwise or anticlockwise
            if random.random() > 0.5:
                rotate_direction = [-1, 1] # -1 -> anticlockwise 
                angle = 90 * random.choice(rotate_direction)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

            # # Resize -> Surprisingly decreasing our accuracy!
            # resize = transforms.Resize(
            #     size=(312, 312), interpolation=Image.NEAREST)
            # image = resize(image)
            # mask = resize(mask)
            # # Random Crop
            # i, j, h, w = transforms.RandomCrop.get_params(
            #     image, output_size=(256, 256))
            # image = TF.crop(image, i, j, h, w)
            # mask = TF.crop(mask, i, j, h, w)

        # # use this in case of observation need
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
        image = self.images[idx]
        image = image.astype(np.float32)
        image = image / 255  # make pixel values between 0-1

        mask = self.masks[idx]
        mask = mask.astype(np.float32)
        mask = mask / 255   # make pixel values 0-1

        # make each pixel to have either 0 or 1
        mask[mask > .7] = 1
        mask[mask <= .7] = 0

        image, mask = self.transform(image, mask)

        return image, mask

    def __len__(self):
        return self.length


# Train Dataset Loaded to Torch Here
train_dataset = FormDS(train_path, number_of_classes, augmentation=True)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print(f'Train DS Size: {len(train_dataset)} ({len(train_data_loader)} batches)')

# Test Dataset Loaded to Torch Here
test_dataset = FormDS(test_path, number_of_classes, augmentation=True)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
print(f'Test  DS Size: {len(test_dataset)} ({len(test_data_loader)} batches)')

# Validation Dataset Loaded to Torch Here
validation_dataset = FormDS(validation_path, number_of_classes, augmentation=True)
validation_data_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
print(f'Valid DS Size: {len(validation_dataset)} ({len(validation_data_loader)} batches)\n')


### Peak a look at the dataset (forms, masks and their combination) ###
# plt_images(train_dataset.images[:batch_size], train_dataset.masks[:batch_size])


# Built Model
model = UnetModel(number_of_classes, dropout_rate).to(device)


# Loss & Optimizer of the Model
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)


### Learning Rate Scheduler ###
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=1, verbose=True)
# scheduler = StepLR(optimizer, step_size=2, gamma=0.2)


# Validation Method for the model
def validation(validation_data_loader, device, criterion, model):
    val_loss = 0
    correct_pixel = 0
    total_pixel = 0

    for images, masks in validation_data_loader:
        images = images.to(device)
        masks = masks.type(torch.LongTensor)
        masks = masks.reshape(masks.shape[0], masks.shape[2], masks.shape[3])
        masks = masks.to(device)

        outputs = model(images)
        val_loss += criterion(outputs, masks).item()

        _, predicted = torch.max(outputs.data, 1)
        correct_pixel += (predicted == masks).sum().item()

        b, h, w = masks.shape
        batch_total_pixel = b*h*w

        total_pixel += batch_total_pixel

    acc = correct_pixel/total_pixel
    return val_loss, acc


# Training of the Model
total_steps = len(train_data_loader)
print(f"{epochs} Epochs & {total_steps} Total Steps per Epoch")

start_time = timer()
print(f'Training Started in {start_time} sec.')
model.train()

batch_step = 0
for epoch in range(epochs):
    for i, (images, masks) in enumerate(train_data_loader, 1):
        images = images.to(device)  # Sends to GPU
        masks = masks.type(torch.LongTensor)
        masks = masks.reshape(masks.shape[0], masks.shape[2], masks.shape[3])
        masks = masks.to(device)    # Sends to GPU

        # Forward pass
        predicts = model(images)
        loss = criterion(predicts, masks)

        # This doubles our batch size
        if batch_extender:
            if batch_step == 0:
                optimizer.zero_grad()
                loss.backward()
                batch_step = 1
            elif batch_step == 1:
                loss.backward()
                optimizer.step()
                batch_step = 0
        else:
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i % int(total_steps/loss_print_per_epoch) == 0:
            if validation_on:
                acc = 0
                # Validation Part
                model.eval()
                with torch.no_grad():
                    validation_loss, validation_accuracy = validation(validation_data_loader, device, criterion, model)
                model.train()

            print(f'Epoch: {epoch + 1}/{epochs}\tStep: {i}/{total_steps}\tLoss: {loss.item():4f}{f"    Valid. Loss: {(validation_loss/len(validation_data_loader)):.4f}    Valid. Acc.: {validation_accuracy * 100:.3f}%" if validation_on else ""} ')
    if scheduler_on:
        scheduler.step(acc)

        # scheduler.step()
        # print(f'LR: {scheduler.get_last_lr()}')

print('Execution time:', '{:5.2f}'.format(timer() - start_time), 'seconds')


# Save the model
torch.save(model.state_dict(), trained_model_path)


# Restore the model from "model_check.pt"
model = UnetModel(number_of_classes, dropout_rate).to(device)

# Load to CPU. Later it can be moved to GPU as needed
model.load_state_dict(torch.load(trained_model_path, map_location=torch.device('cpu')))

count = 0
# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():  # used for dropout layers
    correct_pixel = 0
    total_pixel = 0
    for images, masks in test_data_loader:
        images = images.to(device)
        masks = masks.type(torch.LongTensor)
        # delete color channel to compare direclty with prediction
        masks = masks.reshape(masks.shape[0], masks.shape[2], masks.shape[3])
        masks = masks.to(device)

        predicts = model(images)
        _, predicted = torch.max(predicts.data, 1)
        correct_pixel += (predicted == masks).sum().item()

        b, h, w = masks.shape
        batch_total_pixel = b * h * w
        total_pixel += batch_total_pixel

        # To see random batch prediction uncomment!
        if sample_view and count < 10 and random.random() > 0.5:
            count += 1
            images, masks = undo_preprocess(images, predicts)
            plt_images(images, masks)

    print(f"{correct_pixel} / {total_pixel}")
    print(f"Test Accuracy on the model with {len(test_data_loader) * batch_size} images: {100 * correct_pixel / total_pixel:.4f}%")

# Gets the images and their predicted masks in normalized
images, masks = undo_preprocess(images, predicts)

# Showing last batch as sample
plt_images(images, masks)

# Saves the last batch as sample as input and output images
save_output_batch(images, masks)

print("Program Finished!")
