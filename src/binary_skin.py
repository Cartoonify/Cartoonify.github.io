import numpy as np
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
from skimage.feature import canny
from typing import Tuple
from imageio import imread
from scipy import ndimage
import math as m
from matplotlib.patches import Circle

import matplotlib.pyplot as plt

# detect skin in RGB image using HSV thresholding
# params: n x m RGB image
# return: n x m array with face pixels as 1 and else as 0
# hue and saturation values modified from https://stackoverflow.com/a/8757076
def binary_skin(img):
    hsvImg = rgb2hsv(img)
    newImg = np.zeros((hsvImg.shape[0], hsvImg.shape[1])).astype(int)
    for row in range(hsvImg.shape[0]):
        for col in range(hsvImg.shape[1]):
            pixel = hsvImg[row][col]
            if pixel[0] < 50/255 and pixel[0] > 0 and pixel[1] > 0.18 and pixel[1] < 0.68:
                newImg[row][col] = 1

    kernel = np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]])
    newImg = ndimage.binary_dilation(newImg, structure=kernel, iterations=6).astype(int)
    newImg = ndimage.binary_erosion(newImg, structure=kernel, iterations=6).astype(int)

    plt.set_cmap('gray')
    plt.imshow(newImg)
    plt.show()
    return

# test
imgName = "../res/images/test_binary_skin_2.jpg"
img = imread(imgName)
r = int(min(img.shape[0], img.shape[1]))
binary_skin(img)