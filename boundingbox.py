''' 
    Onur Kirman S009958 Computer Science Undergrad at Ozyegin University

                                *** Notes Before Starting ***
    
    This script detects lines in the predicted masks and encapsulates them with a rectangle. 
    
    Particular database is a post-processed database which originally called 'IAM Handwriting Database 3.0'
        - Overlaping and other issue like outputs are rooted to database. 
    
'''
import os
import re
import cv2
import glob
import random
import numpy as np
from tqdm import tqdm


# Finds the white ares using their pixel values 
# Later, puts every area into bounding-box rectangle that fits those pixels
def bounding_box(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(image_hsv)

    # First creating a mask that has the white area pixels
    mask = cv2.inRange(r, 255, 255)

    # Secondly extracting the corresponding part from the original image
    blob = cv2.bitwise_and(image, image, mask=mask)

    # Later finding contours to particular image using the following hyperparameters
    # We do not care about the hierarchy so discard that part
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Looping over the contours and bound them with a rectangle
    # Later drawing the box masked region to our mask by taking x & y of top left and bottom right coordinates
    for i, contour in enumerate(contours):
        bbox = cv2.boundingRect(contour)
        # Create a mask for this contour
        contour_mask = np.zeros_like(mask)
        # cv2.drawContours(contour_mask, contours, i, 255, -1)

        # Extract the pixels belonging to this contour
        result = cv2.bitwise_and(blob, blob, mask=contour_mask)
        # And draw a bounding box
        top_left, bottom_right = (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3])
        cv2.rectangle(mask, top_left, bottom_right, (255, 255, 255), -1)
    return mask


# Works same as bounding-box but does mix the input with bb-rectangle
def bb_rectangle(input, prediction):

    # input = cv2.cvtColor(input, cv2.COLOR_GRAY2RGB)

    image_hsv = cv2.cvtColor(prediction, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(image_hsv)

    # First creating a mask that has the white area pixels
    mask = cv2.inRange(r, 255, 255)

    # Secondly extracting the corresponding part from the original image
    blob = cv2.bitwise_and(prediction, prediction, mask=mask)

    # Later finding contours to particular image using the following hyperparameters
    # We do not care about the hierarchy so discard that part
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # colors for visualization
    colors = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]] 
    
    for c in range(len(colors)):
        for i in range(len(colors[c])):
            colors[c][i] = int(colors[c][i] * 255)

    # # colors = round(colors)
    # colors = list(map(round, colors))

    # colors = np.array(colors * 100, dtype=int)

    # Looping over the contours and bound them with a rectangle
    # Later drawing the box masked region to our mask by taking x & y of top left and bottom right coordinates
    for contour in contours:
        bbox = cv2.boundingRect(contour)

        # Create a mask for this contour
        contour_mask = np.zeros_like(mask)

        # Extract the pixels belonging to this contour
        result = cv2.bitwise_and(blob, blob, mask=contour_mask)
        # And draw a bounding box
        top_left, bottom_right = (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3])

        color = tuple((int(random.random()*200),int(random.random()*200),int(random.random()*180)))
        # color = (255,30,0)
        cv2.rectangle(input, top_left, bottom_right, color, 2)

    return input

if __name__ == "__main__":
    script_name = __file__.split('/')
    print(f"\'{script_name[-1].capitalize()}\' Started!")

    # folder name for raw form images
    raw_data_folder = 'output/'

    path = os.path.join(os.getcwd(), raw_data_folder)
    saving_path = os.path.join(path, 'box_fitted')
    os.makedirs(saving_path, exist_ok=True)
    
    saving_path_rect = os.path.join(path, 'rect')
    os.makedirs(saving_path_rect, exist_ok=True)


    predictions = glob.glob('./' + raw_data_folder + 'prediction' + '/*_output.png')
    predictions.sort(key=lambda f: int(re.sub('\D', '', f)))

    inputs = glob.glob('./' + raw_data_folder + 'prediction' + '/*_input.png')
    inputs.sort(key=lambda f: int(re.sub('\D', '', f)))

    for i, (prediction, input) in enumerate(tqdm(zip(predictions, inputs))):
        mask = cv2.imread(prediction)
        box_fitted = bounding_box(mask)

        # # Used for peaking the outputs, ->  will be deleted!
        # winname = 'mask'
        # cv2.imshow(winname, mask)
        # cv2.moveWindow(winname, 400, 100)

        # winname = 'box_fitted'
        # cv2.imshow(winname, box_fitted)
        # cv2.moveWindow(winname, 410+256, 100)
        # cv2.waitKey(2000)
        # cv2.destroyAllWindows()

        cv2.imwrite(os.path.join(saving_path, str(i) + '_boxfitted.png'), box_fitted)

        rect = bb_rectangle(cv2.imread(input), mask)
        
        # winname = 'rect_fitted'
        # cv2.imshow(winname, rect)
        # cv2.moveWindow(winname, 410+256, 100)
        # cv2.waitKey(2000)
        # cv2.destroyAllWindows()

        cv2.imwrite(os.path.join(saving_path_rect, str(i) + '_rect.png'), rect)


    print("Program Finished!")



