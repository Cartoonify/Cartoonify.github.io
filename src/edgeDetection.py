from skimage.color import rgb2gray
import numpy as np
from skimage import feature

def edgeDetection(img, quantize_img):
    img_gray = rgb2gray(quantize_img.astype(np.uint8))
    edges = feature.canny(img_gray, sigma=2, low_threshold=0.1, high_threshold=0.3)
    overlaid_img = np.copy(quantize_img)
    overlaid_img[edges] = 0
    return edges, overlaid_img