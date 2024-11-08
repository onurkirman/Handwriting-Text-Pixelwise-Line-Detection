''' 
    Onur Kirman S009958 Computer Science Undergrad at Ozyegin University     
'''

import torch
import utils.boundingbox as boundingbox
from utils.DL_Utils import (FormDS, Test, Train, Validation, build_model, load_data,
                      plt_images, save_output_batch, save_predictions,
                      torch_loader, undo_preprocess, plot_test_predictions)

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
data_dir = 'dataset'
test_path = data_dir + '/test'

# Trained Model Path
trained_model_path = 'weight\\model_check.pt'

# Directory for predictions
output_dir = 'output_test'


# Test Dataset Loaded to Torch Here
test_data_loader = torch_loader(test_path, number_of_classes, batch_size)

# Restore the model from "model_check.pt"
model = model = build_model('unet', device, number_of_classes, dropout_rate)

# Load to CPU. Later it can be moved to GPU as needed
model.load_state_dict(torch.load(trained_model_path, map_location=torch.device('cpu')))

# Testing Process
test = Test(test_data_loader, batch_size, device)
test.start(model, is_saving_output, sample_view, output_dir)

# Post-Process Part
if is_saving_output:
    boundingbox.post_process(output_dir)

# Presentation Demo Plot
plot_test_predictions(output_dir, batch_size)

print("Program Finished!")
