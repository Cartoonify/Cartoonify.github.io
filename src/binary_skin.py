import numpy as np
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
from skimage.feature import canny
from typing import Tuple
from imageio import imread
from scipy import ndimage
import math as m
from matplotlib.patches import Circle

import matplotlib.pyplot as plt
from skimage.morphology import label
from tree import applyToArrayReturnArray
# import sys
# np.set_printoptions(threshold=sys.maxsize)

# detect skin in RGB image using HSV thresholding
# params: img - n x m rgb image with skin
# return: newImg - n x m array with face pixels as 1 and else as 0 eroded and dilated to form connected components
# hue and saturation values modified from https://stackoverflow.com/a/8757076
def binary_skin_erosion_dilation(img):
    newImg = applyToArrayReturnArray(img, True).reshape((img.shape[0], img.shape[1]))
    kernel = np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]])
    newImg = ndimage.binary_erosion(newImg, structure=kernel, iterations=1).astype(int)
    newImg = ndimage.binary_dilation(newImg, structure=kernel, iterations=1).astype(int)
    return newImg

# detect skin in RGB image using HSV thresholding
# params: img - n x m RGB image
# return: newImg - n x m array with face pixels as 1 and else as 0
# hue and saturation values modified from https://stackoverflow.com/a/8757076
# def binary_skin(img):
#     hsvImg = rgb2hsv(img)
#     newImg = np.zeros((hsvImg.shape[0], hsvImg.shape[1])).astype(int)
#     for row in range(hsvImg.shape[0]):
#         for col in range(hsvImg.shape[1]):
#             pixel = hsvImg[row][col]
#             if pixel[0] < 50/255 and pixel[0] > 0 and pixel[1] > 0.18 and pixel[1] < 0.68:
#                 newImg[row][col] = 1

#     kernel = np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]])
#     newImg = ndimage.binary_dilation(newImg, structure=kernel, iterations=6).astype(int)
#     newImg = ndimage.binary_erosion(newImg, structure=kernel, iterations=6).astype(int)
#     return newImg

# computes connected components in image
# params: binaryImg - a binary image to perform CC on
# return: labeledImg - image with each pixel labeled as the component number it belongs to
#         numComponents - number of connected components, subtracting 1 as 0 values are simply background
def connected_components(binaryImg):
    labeledImg = label(binaryImg, connectivity=2)
    numComponents = np.unique(labeledImg)
    return labeledImg, numComponents.shape[0] - 1

# def getAvgSkinColor(img):
#     hsvImg = rgb2hsv(img)
#     skinColor = np.array([[0, 0, 0]])
#     numPixels = 0
#     for row in range(hsvImg.shape[0]):
#         for col in range(hsvImg.shape[1]):
#             pixel = hsvImg[row][col]
#             if pixel[0] < 50/255 and pixel[0] > 0 and pixel[1] > 0.18 and pixel[1] < 0.68 and pixel[2] > 0.2:
#                 skinColor = np.vstack((skinColor, pixel))
#                 numPixels += 1
#     print(np.median(skinColor ,axis = 0))
#     return (skinColor / numPixels)

# test ------------------------------------------------
# imgName = "../face.png"
# img = imread(imgName)

# result = binary_skin_erosion_dilation(img)
# plt.set_cmap('gray')
# plt.imshow(result)
# plt.show()

# test connected components:
# result, numComponents = connected_components(result)
# plt.set_cmap('viridis')
# plt.imshow(result)
# plt.show()