from skimage.color import rgb2gray
import numpy as np
from skimage import feature

def edgeDetection(img, sigma=2):
    img_gray = rgb2gray(img.astype(np.uint8))
    edges = feature.canny(img_gray, sigma=sigma, low_threshold=0.1, high_threshold=0.3)
    return edges