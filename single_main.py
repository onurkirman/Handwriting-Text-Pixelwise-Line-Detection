''' 
    Onur Kirman S009958 Computer Science Undergrad at Ozyegin University

                                *** Notes Before Starting ***
    
    In the folder named data, we have; line_info.txt and 1539 form images from the IAM Handwriting Database
        - The line_info.txt is the reformatted version (header part deleted version) of lines.txt file
    
    Hyperparameters that can be adjusted, and the Data containing paths are listed at the top of the code

    Directory Hierarchy:
    src
        data            -> Raw Data
            - forms                     -> Raw images provided by IAM Handwriting DB
            - line_info.txt             -> Reformatted version (header part deleted version) of lines.txt file
        dataset         -> folder that has the preprocessed images seperated as foldered below
            - train         -> images for training
                - form                  -> preprocessed form images
                - mask                  -> mask images that is a creation of preprocess using given line information
            - test          -> images for testing
                - form
                - mask
            - validation    -> images for validation
                - form
                - mask
        models
            - CNN_network.py            -> Simple CNN Model
            - Unet_model.py             -> Full Unet Model
            - Unet_model_clipped.py     -> Sliced Unet Model
        output
            - rect                      -> Rectangle-Fitted tested form images
            - box_fitted                -> Bounding Box Created Over the Predictions
            - form                      -> Form images in the dataset/test folder with random transformations
            - mask                      -> Predictions/Outputs of the network
        output_batch -> (created, if requested, at the end of main.py to save the output batch) ->
        utils
            - image_preprocess.py       -> Module that preprocesses the raw images and saves them to given directory.
            - DL_Utils.py               -> Module has the required classes and boiler functions needed.
        weight
            - model_check.pt            -> checkpoint of the model used.
        main.py        
        

    Steps Followed:
    Load Data -> Make Dataset -> Load Dataset -> Built Model -> Train Model -> Validate -> Save Model -> Load Model -> Test Model -> Output/View 
    
    Note that: Validation done in the Training Part if wanted

    Note for Loss Function ->  "For a binary classification you could use nn.CrossEntropyLoss() with a logit output of shape [batch_size, 2] 
                                or nn.BCELoss() with a nn.Sigmoid() in the last layer."
'''


import glob
import os
import random
import re
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

print("Program Started!")

print(f'Cuda Available: {torch.cuda.is_available()}')
print(f'{"Cuda Device Name: " + torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "No Cuda Device Found"}')


# Hyperparameters
epochs = 8                # 4 predicts well, might be 2. 8 is the best
batch_size = 4            # 4 is OK, might be 8 (exceed mem.)
batch_extender = True     # Extends the batch so that training process done once in twice -> gives better result
learning_rate = 1e-2      # 1e-3 is OK., 5e-4 also OK. (0.01 -> 0.001 -> 0.0005) LR Scheduler!
dropout_rate = 0.0        # 0.2 is nice with big train data
loss_print_per_epoch = 1  # desired # loss data print per epoch
number_of_classes = 2     # OK.
validation_on = True
scheduler_on = True
sample_view = False
is_saving_output = False

# CUDA for PyTorch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# Image Paths
data_dir = 'dataset'
train_path = data_dir + '/train'
test_path = data_dir + '/test'
validation_path = data_dir + '/validation'

# Trained Model Path
trained_model_path = 'weight\\model_check.pt'
os.makedirs(os.path.join(os.getcwd(), trained_model_path.split("\\")[0]), exist_ok=True)

# Hyperparameter Print
print(f'Batch Size: {batch_size*2 if batch_extender else batch_size} {"(Artificial Batch)" if batch_extender else ""}')
print(f'Learning Rate: {learning_rate}')
print(f'Dropout  Rate: {dropout_rate}\n')


# Plots the given batch in 3 rows; Raw, Mask, Bitwise_Anded
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


# Saves the given batch in directory
def save_predictions(images, predictions, filenames):
    path = os.path.join(os.getcwd(), 'output\\')
    
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


train_data_size = 984
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

        # Added to observe speed of training    -> will be deleted!
        if dataset_path == train_path and i == train_data_size:
            break

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
model = UnetModelClipped(number_of_classes, dropout_rate).to(device)


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

    for images, masks, _ in validation_data_loader:
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
losses = []
for epoch in range(epochs):
    running_loss = 0.0
    for idx, (images, masks, _) in enumerate(train_data_loader, 1):
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

        if idx % int(total_steps/loss_print_per_epoch) == 0:
            if validation_on:
                acc = 0
                # Validation Part
                model.eval()
                with torch.no_grad():
                    validation_loss, validation_accuracy = validation(validation_data_loader, device, criterion, model)
                model.train()
            valstr = f'\tValid. Loss: {(validation_loss/len(validation_data_loader)):.4f}\tValid. Acc.: {validation_accuracy * 100:.3f}%' if validation_on else ''
            print(f'Epoch: {epoch + 1}/{epochs}\tSt: {idx}/{total_steps}\tLast.Loss: {loss.item():4f}{valstr}')
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_data_loader)
    losses.append(epoch_loss)
    if scheduler_on:
        scheduler.step(acc) # -> ReduceLROnPlateau

        # scheduler.step() # -> StepLR
        # print(f'LR: {scheduler.get_last_lr()}')
print('Execution time:', '{:5.2f}'.format(timer() - start_time), 'seconds')
# plt.plot(losses)
# plt.show()
print(f"Overall Average Loss: {sum(losses)/len(losses)}")


# Save the model
torch.save(model.state_dict(), trained_model_path)


# Restore the model from "model_check.pt"
model = UnetModelClipped(number_of_classes, dropout_rate).to(device)

# Load to CPU. Later it can be moved to GPU as needed
model.load_state_dict(torch.load(trained_model_path, map_location=torch.device('cpu')))

all_forms = []
all_predictions = []
all_filenames = []
view_count = 0
# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():  # used for dropout layers
    correct_pixel = 0
    total_pixel = 0
    for images, masks, filenames in test_data_loader:
        images = images.to(device)
        masks = masks.type(torch.LongTensor)
        # delete color channel to compare directly with prediction
        masks = masks.reshape(masks.shape[0], masks.shape[2], masks.shape[3])
        masks = masks.to(device)

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
    print(f"Test Accuracy on the model with {len(test_data_loader) * batch_size} images: {100 * correct_pixel / total_pixel:.4f}%")


# # Saves the output
# if is_saving_output:
#     save_predictions(np.array(all_forms), np.array(all_predictions), np.array(all_filenames))

# # Gets the images and their predicted masks in normalized
# images, masks = undo_preprocess(images, predicts)

# # Showing last batch as sample
# plt_images(images, masks)

# # Saves the last batch as sample as input and output images
# save_output_batch(images, masks)

print("Program Finished!")
