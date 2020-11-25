''' 
                    *** Notes before starting ***
    In the folder named data, we have; line_info.txt and form folder which 
    contains images with total number of 1539
    
'''

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


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
def downscale_image(img, width, height):
    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return resized


# Creates folders in the given path
def folder_creator(paths):
    try:
        for path in paths:
            os.makedirs(os.path.join(path, 'form'), exist_ok=True)
            os.makedirs(os.path.join(path, 'mask'), exist_ok=True)
        
    except OSError as error:
        print(error)


# Normalizes the given image by first subtracting lowest intensity from all pixels
# and second multiplying each pixel with 255/(highest-lowest)
def normalization(image):
    # Lowest & Highest Intensity Extraction
    lowest = np.amin(image, axis=(0, 1))
    highest = np.amax(image, axis=(0, 1))

    for i in range(len(image)):
        for j in range(len(image[i])):
            image[i][j] = image[i][j] - lowest
            image[i][j] = image[i][j] * 255 / (highest - lowest)
    return image


def denoise_image(image):
    image = cv2.fastNlMeansDenoising(image, None, 10, 7, 5)  # check the values for threshold
    # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                         cv2.ADAPTIVE_THRESH_MEAN_C, 13, 2)
    # image = cv2.bitwise_not(image)
    return image

# Checks the line boxes, if overlaps fixes it by reducing height of the upper line
def lines_preprocess(lines):
    corrected_lines = []
    margin_factor = 50  # hyperparameter to decide how wide you want the lines are. (45 & 50 gives the best)

    lines = np.array(lines, dtype=int)
    for i in range(len(lines)-1):
        
        upper_line = lines[i]
        lower_line = lines[i+1]

        ux, uy, uwidth, uheight = upper_line
        lx, ly, lwidth, lheight = lower_line

        # if not overlaps it will return negative
        distance = (uy + uheight) - ly
        
        # if there exist a margin within the lines with less than needed make it bigger
        if(distance <= 0 and distance > -margin_factor):
            shift = -margin_factor - distance
            lines[i][3] = lines[i][3] + shift  # dont forget to fix on the returned array
        elif(distance > 0): # if overlaps
            shift = margin_factor + distance
            lines[i][3] = lines[i][3] - shift
        
    return lines


# Preprocess the images in the given file and returns them as np array
def covert_images(folder_name, d, numIM):
    print("Converting Images...")
    dl = list(d) # gets a list of file names in the raw_data folder

    images = []
    masks = []

    for count, filename in enumerate(dl, 0):
        img = cv2.imread(folder_name + filename + '.png', 0)
        img = np.array(img)

        height, width = img.shape
        mask = np.zeros((height, width), dtype='uint8')


        lines = d[filename]
        # Checks the lines beforehand, and if overlaps makes it narrower for the upper line
        lines = lines_preprocess(lines)

        for line in lines:
            x, y, width, height = line
            mask[y:y+height, x:x+width] = 255
        
        # removes the unwanted writer name part at the end
        img[y+height:, 100:-100] = img.mean() + 5 if img.mean() < 251 else 255

        # cropping image from top and bottom parts by 500 pixel to make square downscale readable
        img = img[500:-500, :]
        img = downscale_image(img, 256, 256)
        
        # winname = 'image'
        # cv2.namedWindow(winname)        
        # cv2.moveWindow(winname, 400,100)  
        # cv2.imshow(winname, img)

        img = denoise_image(img)
        img = normalization(img)

        # winname = 'imageN'
        # cv2.imshow(winname, img)
        # cv2.moveWindow(winname, 656,100)
        # cv2.waitKey(5000)
        # cv2.destroyAllWindows()

        images.append(img)

        # cropping mask to match the image
        mask = mask[500:-500, :]
        mask = downscale_image(mask, 256, 256)
        masks.append(mask)
        count += 1

        if count == numIM:
            break
        
    return np.array(images), np.array(masks)


# Saves given forms and masks in the given dataset path
def save_files(forms, masks, path):
    for i in range(forms.shape[0]):
        cv2.imwrite(os.path.join(path + '\\form', str(i) + '.png'), forms[i])
        cv2.imwrite(os.path.join(path + '\\mask', str(i) + '.png'), masks[i])


if __name__ == '__main__':
    print("Program Started!")

    # folder name for raw form images
    raw_data_folder = 'data/forms/' # will be -> 'raw_data/forms/'

    # Hyperparameters
    numIM = 1539 # number of images -> might be deleted later!
    split_percentage = 0.2 # used for data split into two sub-parts

    # Gets current working directory
    path = os.getcwd()
    
    # Paths for folders
    dataset_path = '\\dataset'
    train_path = os.path.join(path + dataset_path, 'train2')
    test_path = os.path.join(path + dataset_path, 'test2')
    validation_path = os.path.join(path + dataset_path, 'validation2')
    

    # we have file name and lines in the corres. image
    parsed_data = parse_line_info()

    # returns all image and mask as np array
    train_images, train_masks = covert_images(raw_data_folder, parsed_data, numIM)
    
    print("Dataset Split Session Started...")
    # First divide the data into test/train datasets
    train_images, test_images, train_masks, test_masks = train_test_split(train_images, train_masks, test_size = split_percentage)
    
    # Later divide the remaining train dataset into train & validation datasets
    train_images, validation_images, train_masks, validation_masks = train_test_split(train_images, train_masks, test_size = split_percentage)


    # Creates the folders
    folder_creator([train_path, test_path, validation_path])

    print(train_images.shape[0], test_images.shape[0], validation_images.shape[0]) # length of the datasets 

    # print("Saving Datasets")
    # # Save all train, test & validation files
    # save_files(train_images, train_masks, train_path)
    # save_files(test_images, test_masks, test_path)
    # save_files(validation_images, validation_masks, validation_path)

    print("Program Finished!")
