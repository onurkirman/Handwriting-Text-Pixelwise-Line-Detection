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
                - mask                  -> mask images that is a creation of preprocess using given line informations
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
            - form                      -> Form images in the dataset/test folder, saved again for easy use & comparison
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

import os
import time

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary

from utils.image_preprocess import preprocess_logic
from DL_Utils import (FormDS, Test, Train, Validation, build_model, load_data,
                      plt_images, save_output_batch, save_predictions,
                      torch_loader, undo_preprocess)
import boundingbox

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
            trained_model_path,
            output_dir):

    print('Process Started.')
    ## Hyperparameters ##
    print(f'Batch Size: {batch_size*2 if batch_extender else batch_size} {"(Artificial Batch)" if batch_extender else ""}')
    print(f'Learning Rate: {learning_rate}')
    print(f'Dropout  Rate: {dropout_rate}\n')

    ## Train Dataset Loaded to Torch Here ##
    train_data_loader = torch_loader(train_path, number_of_classes, batch_size, augmentation=True)

    ## Validation Dataset Loaded to Torch Here ##
    validation_data_loader = torch_loader(validation_path, number_of_classes, batch_size, augmentation=True)

    ## Peak a look at the dataset (forms, masks and their combination) ##
    if sample_view:
        plt_images(train_data_loader.dataset.images[:batch_size], train_data_loader.dataset.masks[:batch_size], batch_size)


    ## Built Model ##
    model = build_model('unet', device, number_of_classes, dropout_rate)


    ## Loss Function ##
    criterion = nn.CrossEntropyLoss()

    ## Optimizer of the Model ##
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)    # Works better than SGD
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)


    ## Learning Rate Scheduler ##
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=1, verbose=True)  # Works better
    # scheduler = StepLR(optimizer, step_size=2, gamma=0.2)


    ### VALIDATION ##
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
    test.start(model, is_saving_output, sample_view, output_dir)

if __name__ == "__main__":
    
    # ### Preprocess Part ###
    # # folder name for raw form images
    # raw_data_folder = 'data/forms/'

    # # Hyperparameters
    # final_image_size = (256, 256)
    # split_percentage = 0.2 # used for data split into two sub-parts

    # # Dataset directory
    # dataset_folder_name = 'dataset_combined'

    # # Logic Part of Pre-Process
    # preprocess_logic(raw_data_folder,
    #                 final_image_size,
    #                 split_percentage,
    #                 dataset_folder_name
    #                 )
    
    print(f'Cuda Available: {torch.cuda.is_available()}')
    print(f'{"Cuda Device Name: " + torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "No Cuda Device Found"}')
    
    # Hyperparameters for Training & Testing
    epochs = 8                # 4 predicts well, might be 2. 8 is the best
    batch_size = 4            # 4 is OK, might be 8 (exceed mem.)
    batch_extender = True     # Extends the batch so that training process done once in twice -> gives better result
    learning_rate = 1e-2      # 1e-3 is OK., 5e-4 also OK. 1e-2 is the best. (0.01 -> 0.001 -> 0.0005) LR Scheduler!
    dropout_rate = 0.0        # 0.2 is nice with big train data
    loss_print_per_epoch = 1  # desired number of loss data print per epoch
    number_of_classes = 2     # OK.
    validation_on = True
    scheduler_on = True
    sample_view = False
    is_saving_output = True

    # CUDA for PyTorch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Image Paths
    data_dir = 'dataset_combined_fixed'
    train_path = data_dir + '/train'
    test_path = data_dir + '/test'
    validation_path = data_dir + '/validation'

    # Trained Model Path
    trained_model_path = 'weight\\model_check_combined.pt'
    os.makedirs(os.path.join(os.getcwd(), trained_model_path.split("\\")[0]), exist_ok=True)

    # Directory for predictions
    output_dir = 'output_combined_fixed'

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
            trained_model_path,
            output_dir)

    # if is_saving_output:
    #     boundingbox.post_process(output_dir)

    print('Program Finished!')
