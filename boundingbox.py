# Script for finding line bounding boxes

import os
import re
import cv2
import glob
import numpy as np
from tqdm import tqdm

def bounding_box(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(image_hsv)

    # First we create a mask selecting all the pixels of this hue
    mask = cv2.inRange(r, 255, 255)
    # And use it to extract the corresponding part of the original colour image
    blob = cv2.bitwise_and(image, image, mask=mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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


if __name__ == "__main__":
    aa = __file__.split('\\')
    print(f"\'{aa[-1]}\' Started!")

    # folder name for raw form images
    raw_data_folder = 'predictions/'

    path = os.path.join(os.getcwd(), 'box_fitted\\')
    os.makedirs(path, exist_ok=True)


    preds = glob.glob('./' + raw_data_folder + '/*_output.png')
    preds.sort(key=lambda f: int(re.sub('\D', '', f)))

    count = 0
    for pred in tqdm(preds):
        mask = cv2.imread(pred)

        box_fitted = bounding_box(mask)

        # winname = 'mask'
        # cv2.imshow(winname, mask)
        # cv2.moveWindow(winname, 400, 100)

        # winname = 'box_fitted'
        # cv2.imshow(winname, box_fitted)
        # cv2.moveWindow(winname, 410+256, 100)
        # cv2.waitKey(2000)
        # cv2.destroyAllWindows()

        # cv2.imwrite(os.path.join(path, str(count) + '.png'), box_fitted)
        count += 1


    print("Program Finished!")



