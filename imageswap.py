import cv2
import numpy as np
import os

current_directory = os.path.dirname(os.path.realpath(__file__))

# Iterate over all files in the current directory
for filename in os.listdir(current_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load the image
        image_path = os.path.join(current_directory, filename)

        # Load the target photo and the image containing the target photo
        target_photo = cv2.imread('target.jpg', cv2.IMREAD_COLOR)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # Convert the images to grayscale
        target_gray = cv2.cvtColor(target_photo, cv2.COLOR_BGR2GRAY)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Perform template matching to find the coordinates of the target photo within the image
        result = cv2.matchTemplate(image_gray, target_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Get the coordinates of the target photo within the image
        top_left = max_loc
        bottom_right = (top_left[0] + target_photo.shape[1], top_left[1] + target_photo.shape[0])

        # Swap the target photo with another image
        replacement_image = cv2.imread('goal.jpg', cv2.IMREAD_COLOR)
        image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = replacement_image

        # Save the modified image
        cv2.imwrite(image_path, image)