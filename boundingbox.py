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
def bounding_box(input, prediction, draw_rect=False):
    image_hsv = cv2.cvtColor(prediction, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(image_hsv)

    # First creating a mask that has the white area pixels
    mask = cv2.inRange(r, 255, 255)

    # Secondly extracting the corresponding part from the original image
    blob = cv2.bitwise_and(prediction, prediction, mask=mask)

    # Later finding contours to particular image using the following hyperparameters
    # We do not care about the hierarchy so discard that part
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Looping over the contours and bound them with a rectangle
    # Later drawing the box masked region to our mask by taking x & y of top left and bottom right coordinates
    for contour in contours:
        bbox = cv2.boundingRect(contour)
        # Create a mask for this contour
        contour_mask = np.zeros_like(mask)
        # cv2.drawContours(contour_mask, contours, i, 255, -1)

        # Extract the pixels belonging to this contour
        result = cv2.bitwise_and(blob, blob, mask=contour_mask)
        # And draw a bounding box
        top_left, bottom_right = (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3])
        cv2.rectangle(mask, top_left, bottom_right, (255, 255, 255), -1)
        
        if draw_rect:
            color = tuple((int(random.random()*200),int(random.random()*200),int(random.random()*180)))
            cv2.rectangle(input, top_left, bottom_right, color, 2)
    return input, mask

def post_process(prediction_folder):
    path = os.path.join(os.getcwd(), prediction_folder)
    saving_path = os.path.join(path, 'box_fitted')
    os.makedirs(saving_path, exist_ok=True)
    
    saving_path_rect = os.path.join(path, 'rect')
    os.makedirs(saving_path_rect, exist_ok=True)


    predictions = glob.glob('./' + prediction_folder + '/mask' + '/*.png')
    predictions.sort(key=lambda f: int(re.sub('\D', '', f)))

    inputs = glob.glob('./' + prediction_folder + '/form' + '/*.png')
    inputs.sort(key=lambda f: int(re.sub('\D', '', f)))

    # Enables/Disables boinding-boxing the input image with colored rectangles
    draw_rect = True

    for i, (prediction, input) in enumerate(tqdm(zip(predictions, inputs))):
        form = cv2.imread(input)
        mask = cv2.imread(prediction)
        rect, box_fitted = bounding_box(form, mask, draw_rect=draw_rect)

        # # Used for peaking the outputs, ->  will be deleted!
        # winname = 'mask'
        # cv2.imshow(winname, mask)
        # cv2.moveWindow(winname, 400, 100)

        # winname = 'box_fitted'
        # cv2.imshow(winname, box_fitted)
        # cv2.moveWindow(winname, 410+256, 100)
        # cv2.waitKey(2000)
        # cv2.destroyAllWindows()

        cv2.imwrite(os.path.join(saving_path, str(input.split('\\')[-1][:-4]) + '_boxfitted.png'), box_fitted)

        # rect = bb_rectangle(input, mask)
        
        # winname = 'rect_fitted'
        # cv2.imshow(winname, rect)
        # cv2.moveWindow(winname, 410+256, 100)
        # cv2.waitKey(2000)
        # cv2.destroyAllWindows()

        cv2.imwrite(os.path.join(saving_path_rect, str(input.split('\\')[-1][:-4]) + '_rect.png'), rect)

if __name__ == "__main__":
    script_name = __file__.split('/')
    print(f"\'{script_name[-1].capitalize()}\' Started!")

    # folder name for raw form images
    raw_data_folder = 'output_combined_fixed'

    # post process
    post_process(raw_data_folder)

    print("Program Finished!")



