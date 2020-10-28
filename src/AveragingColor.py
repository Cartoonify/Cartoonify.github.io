import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2hsv, hsv2rgb
from imageio import imread, imsave
import matplotlib.pyplot as plt

"""
This function of the process takes in an input image and averages the colors 
in each region of the image using kmeans. The resulting output in a "quanitized" image
in which the pixels have a more uniform distribution across the entire image.

A consideration to be made for this function is the k-value being inputted as
an argument which controls how many clusters are being formed in the kmeans clustering
process. Each test case image will require testing to observe its most effective k-value
as while lower k-values result in more thorough averaging (and a more "cartoon"-ified effect), they 
also reduce the amount of colors in the image as a whole. The k-values in a sense dictate
how many colors will be visible in the image as each color correlates to a cluster center.
Therefore, a specific image will require a specifically chosen k-value to balance this consideration
of preserving coloring and gaining averaged results. 

"""

def average_colors(img: np.ndarray, k: int) -> np.ndarray:
      
    quantized_img = np.zeros_like(img)

  
    w,h,_ = img.shape
    image = img.reshape(w*h,3)

    clusters = KMeans(n_clusters=k, random_state=101)
    kmeans = clusters.fit(image)
    labels = kmeans.predict(image)
    colors = np.array(kmeans.cluster_centers_).astype('int8')

    quantized_img = np.copy(image)
    for index in range(len(quantized_img)):
      quantized_img[index] = colors[labels[index]]


    
    quantized_img = quantized_img.reshape(w,h,3)

    return quantized_img


test_img_sky = imread("/content/TestImageSkyline.jpeg")
test_img_dog = imread("/content/TestImageDog.jpeg")

dog = average_colors(test_img_dog, 10)
fig, axs = plt.subplots(1, 2)
axs[0].axis("off")
axs[0].imshow(test_img_dog)
axs[1].axis("off")
axs[1].imshow(dog)


sky = average_colors(test_img_sky, 14)
fig, bxs = plt.subplots(1, 2)
bxs[0].axis("off")
bxs[0].imshow(test_img_sky)
bxs[1].axis("off")
bxs[1].imshow(sky)