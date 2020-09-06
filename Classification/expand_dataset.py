# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
import glob
import os	
import re
import cv2
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output directory to store augmentation examples")
ap.add_argument("-t", "--total", type=int, default=100,
	help="# of training samples to generate")
args = vars(ap.parse_args())

# load the input image, convert it to a NumPy array, and then
# reshape it to have an extra dimension
print("[INFO] loading example image...")
#image = load_img(args["image"])
for filename in glob.glob('/Users/kelvi/Desktop/DATA/severityDivided/Train_Test/70_30_split/CRST_ZerosVsMild/train/Zeros/*.jpg'):
    image = cv2.imread(filename)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    imageName = os.path.basename(filename)
    imageName = imageName.replace(".jpg","")
    
    # construct the image generator for data augmentation then
    # initialize the total number of images generated thus far
    aug = ImageDataGenerator(
        #rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode="constant",
        cval=255.0)
    total = 0

    # construct the actual Python generator
    print("[INFO] generating images...")
    imageGen = aug.flow(image, batch_size=1, save_to_dir=args["output"],
        save_prefix="image_"+imageName, save_format="jpg")
    
    # loop over examples from our image data augmentation generator
    for image in imageGen:
        # increment our counter
        total += 1
    
        # if we have reached the specified number of examples, break
        # from the loop
        if total == args["total"]:
            break

