# Matthew Venezia
# Masters research project

# Imports
from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2
################################FUNCTIONS####################################################
def convert_2_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def blur_image(img):
    return cv2.GaussianBlur(img, (3,3), cv2.BORDER_DEFAULT)

def find_hotspots(img, img2):
    # Set a threshold value to find the hot spots (brightest part of image) and highlight them
    thresh_val = 200
    for i in range(25):
        thresh = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY)[1]
        cv2.imshow('THRESHOLD',thresh)
        cv2.waitKey(0)
        # perform a series of erosions and dilations to remove
        # any small blobs of noise from the thresholded image
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=4)
        # show the output image
        print(thresh_val)
        # draw circles and show
        draw_circles(thresh, img2)
        # dec thresh val
        thresh_val = thresh_val - 5

def draw_circles(thresh_in, img_in):
    # Label connected regions of image that are the same color (white)
    labels = measure.label(thresh_in, background=0, connectivity=2)
    # This will store the large hot spots
    mask = np.zeros(thresh_in.shape, dtype="uint8")
    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the number of pixels 
        # mask for current label
        labelMask = np.zeros(thresh_in.shape, dtype="uint8")
        # put the current label (in the loop) into its own mask labelMask
        labelMask[labels == label] = 255
        # number of pixels in the blob
        numPixels = cv2.countNonZero(labelMask)
        # if the number of pixels in the component is within range add it to mask
        if numPixels < 50 and numPixels > 2:
            mask = cv2.add(mask, labelMask)
    
    # find the contours in the mask
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if not cnts:
        print("this is empty")
    else:
        # sort them
        cnts = contours.sort_contours(cnts)[0]

        # loop over the contours
        for (i, c) in enumerate(cnts):
            # draw the bright spot on the image
            (x, y, w, h) = cv2.boundingRect(c)
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            # draw on the OG image
            cv2.circle(img_in, (int(cX), int(cY)), int(radius),(0, 0, 255), 3)

        # show the output image
        cv2.imshow("Image", img_in)
        cv2.waitKey(0)


# MODE 1
def thermal_mode(img):
    # crop out the scale and temp values
    crop_img = img[0:640, 0:387].copy()
    cv2.imshow('Crop image',crop_img)
    cv2.waitKey(0) 
    # convert to grayscale
    gray = convert_2_grayscale(crop_img)
    cv2.imshow('THERMAL GRAYSCALE',gray)
    cv2.waitKey(0)
    # blur the image
    blur = blur_image(gray)
    cv2.imshow('THERMAL BLUR GRAY', blur)
    cv2.waitKey(0)
    # find hot spots
    find_hotspots(blur, crop_img)
    exit()

# MODE 2
def optical_mode(img):
    # convert to grayscale
    gray = convert_2_grayscale(img)
    cv2.imshow('OPTICAL GRAYSCALE',gray)
    cv2.waitKey(0)
    # blur the image
    blur = blur_image(gray)
    cv2.imshow('OPTICAL BLUR GRAY', blur)
    cv2.waitKey(0)
    # find hot spots
    find_hotspots(blur, img)
    exit()
#############################################################################################
# TODO: take both images at once, make it multithreaded??

# Argument handling
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mode", required=True, type=int, choices=range(1, 3), help="select mode, THERMAL=1, OPTICAL=2")
ap.add_argument("-i", "--image", required=True,	help="path to the image file")
# ap.add_argument("-i2", "--image", required=True,   help="path to the regular image file")
args = vars(ap.parse_args())

# Load the image
image = cv2.imread(args["image"])

# Mode selection
mode = args["mode"]
if mode == 1:
    print("Thermal mode selected")
    thermal_mode(image)
elif mode == 2:
    print("Optical mode selected")
    optical_mode(image)
else:
    print("error")
    exit()

# Kill windows
cv2.destroyAllWindows()