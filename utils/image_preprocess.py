''' 
    Onur Kirman S009958 Computer Science Undergrad at Ozyegin University

                                *** Notes Before Starting ***
    
    In the folder named data, we have; line_info.txt and form folder which 
    contains images with total number of 1539
    
'''

import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


#   Parses the line information txt and later puts them into a dict in the
#   format of [filename, line coordinations] to the dictionary
def parse_line_info():
    d = {}
    with open('data/line_info.txt', 'r') as f:
        for line in f:
            ss = line.split(' ')
            form = ss[0].split('-')
            form = form[0]+'-'+form[1]
            coord = [ss[4], ss[5], ss[6], ss[7]]
            if form in d:
                d[form].append(coord)
            else:
                d[form] = [coord]
    return d


#   Upscaling the given image with corresponding scale percentage 
#   It is used for experiment to see what pixel ratio it still holds its readable data
def upscale_image(img, percent):
    width = int(img.shape[1] * percent / 100)
    height = int(img.shape[0] * percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


#   Downscaling the given image into given width and height
def downscale_image(img, final_image_size, interpolation=cv2.INTER_AREA):
    resized = cv2.resize(img, final_image_size, interpolation=interpolation)
    return resized


# Creates folders in the given path
def folder_creator(paths):
    try:
        for path in paths:
            os.makedirs(os.path.join(path, 'form'), exist_ok=True)
            os.makedirs(os.path.join(path, 'mask'), exist_ok=True)
        
    except OSError as error:
        print(error)


# Narrows every line box by 0.6 (0.2 from top & 0.4 from bottom) to overcome line box overlaps
def generic_line_preprocess(lines):
    processed_lines = []
    lines = np.array(lines, dtype=int)

    for line in lines:
        x, y, width, height = line
        width_constant = round(height * 0.1)
        y = y + width_constant * 3
        height = height - width_constant * 6
        processed_lines.append([x, int(y), width, int(height)])
    return np.array(processed_lines)


# Preprocess the images in the given file and returns them as np array
def covert_images(folder_name, d, final_image_size):
    print("Converting Images...")
    dl = list(d) # gets a list of file names in the raw_data folder

    images = []
    masks = []

    for filename in tqdm(dl):
        img = cv2.imread(folder_name + filename + '.png', 0)
        img = np.array(img)

        height, width = img.shape
        mask = np.zeros((height, width), dtype='uint8')

        lines = d[filename]
        lines = generic_line_preprocess(lines)

        for line in lines:
            x, y, width, height = line
            mask[y:y+height, x:x+width] = 255
        
        # removes the unwanted writer name part at the end
        img[y+height:, 100:-100] = img.mean() + 5 if img.mean() < 251 else 255

        # cropping image from top and bottom parts by 500 pixel to make square downscale readable
        img = img[500:-500, :]
        img = downscale_image(img, final_image_size)
        images.append(img)

        # cropping mask to match the image
        mask = mask[500:-500, :]
        mask = downscale_image(mask, final_image_size, interpolation=cv2.INTER_NEAREST)
        masks.append(mask)

        # winname = 'image'
        # cv2.namedWindow(winname)        
        # cv2.moveWindow(winname, 400, 100)  
        # cv2.imshow(winname, img & mask)
        
        # winname = 'imageN'
        # cv2.imshow(winname, mask)
        # cv2.moveWindow(winname, 410 + final_image_size[0], 100)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()
    
    return np.array(images), np.array(masks), tuple(dl)


# Saves given forms with filenames in the given dataset path
def save_files(forms, masks, filenames, path):
    for idx in range(forms.shape[0]):
        cv2.imwrite(os.path.join(path + '\\form', str(filenames[idx]) + '.png'), forms[idx])
        cv2.imwrite(os.path.join(path + '\\mask', str(filenames[idx]) + '.png'), masks[idx])


def preprocess_logic(raw_data_folder,
                    final_image_size,
                    split_percentage,
                    dataset_folder_name
                    ):
    print("Preprocess Started!")
    # Paths for folders
    dataset_path = os.path.join(os.getcwd(), dataset_folder_name)
    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')
    validation_path = os.path.join(dataset_path, 'validation')
    

    # we have file name and lines in the corres. image
    parsed_data = parse_line_info()

    # returns all image and mask as np array
    train_images, train_masks, filenames = covert_images(raw_data_folder, parsed_data, final_image_size)
    

    print("Dataset Split Session Started...")
    # train_test_split -> we can use the stratification parameters
    # First divide the data into test/train datasets
    train_images, test_images, train_masks, test_masks, train_filenames, test_filenames = train_test_split(train_images, train_masks, filenames, test_size = split_percentage)
    
    # Later divide the remaining train dataset into train & validation datasets
    train_images, validation_images, train_masks, validation_masks, train_filenames, validation_filenames = train_test_split(train_images, train_masks, train_filenames, test_size = split_percentage)
    print(train_images.shape[0], test_images.shape[0], validation_images.shape[0]) # length of the datasets 


    # Creates the folders
    folder_creator([train_path, test_path, validation_path])


    # Save all train, test & validation files
    print("Saving Datasets")
    data_images = [train_images, test_images, validation_images]
    data_masks = [train_masks, test_masks, validation_masks]
    data_filenames = [train_filenames, test_filenames, validation_filenames]
    data_paths = [train_path, test_path, validation_path]

    for imgs, masks, flnames, path in zip(data_images,
                                 data_masks,
                                 data_filenames,
                                 data_paths):
        save_files(imgs, masks, flnames, path)
    print("Preprocess Finished!")


if __name__ == '__main__':
    # folder name for raw form images
    raw_data_folder = 'data/forms/'

    # Hyperparameters
    final_image_size = (256, 256)
    split_percentage = 0.2 # used for data split into two sub-parts

    # Dataset directory
    dataset_folder_name = 'dataset_combined'

    # Logic Part of Pre-Process
    preprocess_logic(raw_data_folder,
                    final_image_size,
                    split_percentage,
                    dataset_folder_name
                    )
