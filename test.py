''' 
    Onur Kirman S009958 Computer Science Undergrad at Ozyegin University

                                *** Notes Before Starting ***
    
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
        models
            - CNN_network.py            -> Simple CNN Model
            - Unet_model.py             -> Full Unet Model
            - Unet_model_clipped.py     -> Sliced Unet Model
        output
            - rect              -> Rectangle-Fitted tested form images
            - box_fitted        -> Bounding Box Created Over the Predictions
            - form              -> form images tested saved again for easy use and comparason
            - mask              -> Predictions/Outputs of the network
        output_batch -> (created, if requested, at the end of main.py to save the output batch) ->
        utils
            - image_preprocess.py
            ** Add a new script for data allocation with network train & test. Remove the part from the main
            ** Keep only the requered script runs in main.py -> TODO LATER !! 
        weight
            - model_check.pt    -> checkpoint of the model used.
        main.py
        
        

    Steps Followed:
    Load Data -> Make Dataset -> Load Dataset -> Built Model -> Train Model -> Validate -> Save Model -> Load Model -> Test Model -> Output/View 
    
    Note that: Validation done in the Training Part if wanted

    **** Also -> "Train: Overlapsing, Train2: Condition Specific Not Overlaping, Train3: Generic Not Overlaping" ****

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

from DL_Utils import (FormDS, Test, Train, Validation, load_data, plt_images,
                      save_output_batch, save_predictions, undo_preprocess,
                      torch_loader)
from models.CNN_network import Network
from models.Unet_model import UnetModel
from models.Unet_model_clipped import UnetModelClipped

print("Test Started!")

print(f'Cuda Available: {torch.cuda.is_available()}')
print(f'{"Cuda Device Name: " + torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "No Cuda Device Found"}')


# Hyperparameters
batch_size = 4            # 4 is OK, might be 8 (exceed mem.)
dropout_rate = 0.0        # 0.2 is nice with big train data
number_of_classes = 2     # OK.
sample_view = False
is_saving_output = True

# CUDA for PyTorch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# Image Paths
data_dir = 'dataset_combined'
test_path = data_dir + '/test'

# Trained Model Path
trained_model_path = 'weight\\model_check_combined.pt'

# Test Dataset Loaded to Torch Here
test_data_loader = torch_loader(test_path, number_of_classes, batch_size, augmentation=True)

# Restore the model from "model_check.pt"
model = UnetModel(number_of_classes, dropout_rate).to(device)

# Load to CPU. Later it can be moved to GPU as needed
model.load_state_dict(torch.load(trained_model_path, map_location=torch.device('cpu')))

# Testing Process
test = Test(test_data_loader, batch_size, device)
test.start(model, is_saving_output, sample_view)

print("Program Finished!")
