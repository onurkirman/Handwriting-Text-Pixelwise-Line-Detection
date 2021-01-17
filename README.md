# Handwriting-Text-Pixelwise-Line-Detection
    Onur Kirman S009958 Computer Science Undergrad at Ozyegin University
    
    Handwriting-Text-Pixelwise-Line-Detection is my senior project that detects handwritten text lines
    using 3 different Conv. NN. (Simple_CNN, U-net and U-net_Clipped [half size U-net]).
    
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
    Load Data->Make Dataset->Load Dataset->Built Model->Train Model->Validate->Save/Load Model->Test Model->Output/View 