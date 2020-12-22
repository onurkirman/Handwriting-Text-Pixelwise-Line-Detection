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
                      build_model, torch_loader)
from models.CNN_network import Network
from models.Unet_model import UnetModel
from models.Unet_model_clipped import UnetModelClipped


def process(epochs,
            batch_size,
            batch_extender,
            learning_rate,
            dropout_rate,
            loss_print_per_epoch,
            number_of_classes,
            validation_on,
            scheduler_on,
            sample_view,
            is_saving_output,
            device,
            train_path,
            test_path,
            validation_path,
            trained_model_path):
    print('Process Started.')

    # Hyperparameter Print
    print(f'Batch Size: {batch_size*2 if batch_extender else batch_size} {"(Artificial Batch)" if batch_extender else ""}')
    print(f'Learning Rate: {learning_rate}')
    print(f'Dropout  Rate: {dropout_rate}\n')

    # Train Dataset Loaded to Torch Here
    train_data_loader = torch_loader(train_path, number_of_classes, batch_size, augmentation=True)

    # Validation Dataset Loaded to Torch Here
    validation_data_loader = torch_loader(validation_path, number_of_classes, batch_size, augmentation=True)

    ### Peak a look at the dataset (forms, masks and their combination) ###
    # plt_images(train_dataset.images[:batch_size], train_dataset.masks[:batch_size])


    # Built Model
    model = build_model('unet', device, number_of_classes, dropout_rate)


    # Loss & Optimizer of the Model
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)


    ### Learning Rate Scheduler ###
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=1, verbose=True)
    # scheduler = StepLR(optimizer, step_size=2, gamma=0.2)

    # VALIDATION
    validation = Validation(validation_data_loader, device, criterion)


    ### TRAIN ###
    train = Train(train_data_loader, device, criterion, optimizer, validation, scheduler)
    model = train.start(model,epochs,batch_extender,validation_on, scheduler_on, loss_print_per_epoch)

    # Save the model
    torch.save(model.state_dict(), trained_model_path)


    ### TEST ###
    # Test Dataset Loaded to Torch Here
    test_data_loader = torch_loader(test_path, number_of_classes, batch_size, augmentation=True)
    
    # Rebuild the model, before the restore
    model = build_model('unet', device, number_of_classes, dropout_rate)

    # Restore the model from "model_check.pt"
    # Load to CPU. Later it can be moved to GPU as needed
    model.load_state_dict(torch.load(trained_model_path, map_location=torch.device('cpu')))

    test = Test(test_data_loader, batch_size, device)
    test.start(model, is_saving_output, sample_view)

if __name__ == "__main__":
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
    data_dir = 'dataset_combined'
    train_path = data_dir + '/train'
    test_path = data_dir + '/test'
    validation_path = data_dir + '/validation'

    # Trained Model Path
    trained_model_path = 'weight\\model_check_combined.pt'
    os.makedirs(os.path.join(os.getcwd(), trained_model_path.split("\\")[0]), exist_ok=True)

    process(epochs,
            batch_size,
            batch_extender,
            learning_rate,
            dropout_rate,
            loss_print_per_epoch,
            number_of_classes,
            validation_on,
            scheduler_on,
            sample_view,
            is_saving_output,
            device,
            train_path,
            test_path,
            validation_path,
            trained_model_path)


    print('Program Finished!')
