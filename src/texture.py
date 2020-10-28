# import cv2
from imageio import imread, imsave
from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import color

class FillColors():
    def __init__(self, img, color_threshold, face_mask):
        self.img_array = img

        smaller_img = np.copy(img).astype(np.float32) / 255
        self.img_vals = color.rgb2lab(smaller_img)

        self.color_threshold = color_threshold

        self.face_mask = face_mask

        self.curr_col = None
        self.curr_row = None
        self.curr_color = None
        
        self.filled_locations = np.zeros(img.shape, dtype=bool)
        
        filled_flag = True
        print('Filling colors...')
        while filled_flag:
            next_loc = np.argwhere(self.filled_locations == False)[0]
            
            self.curr_row = next_loc[0]
            self.curr_col = next_loc[1]
            self.curr_color = self.img_vals[self.curr_row][self.curr_col]

            self.fill_colors()
            if False in self.filled_locations:
                continue
            else:
                filled_flag = False
        print('Done filling colors!')

    def similar_color(self, row, col):
        color_to_check = self.img_vals[row][col]
        deltaE = color.deltaE_ciede94(color_to_check, self.curr_color)

        return deltaE <= self.color_threshold#(1 if self.face_mask[row][col] > 0 else self.color_threshold)

    def fill_colors(self):
        img_rows, img_cols, _ = self.img_vals.shape
        
        similar_colors = np.zeros((img_rows, img_cols), dtype=np.bool)

        pixels_to_check = set()
        pixels_to_check.add((self.curr_row, self.curr_col))

        checked = np.zeros_like(similar_colors)

        while pixels_to_check:
            (row, col) = pixels_to_check.pop()

            if self.filled_locations[row][col][0]:
                continue

            if checked[row][col] or similar_colors[row][col]:
                continue

            checked[row, col] = True

            if not self.similar_color(row, col):
                continue

            similar_colors[row][col] = True
            if col > 0 and self.filled_locations[row][col - 1][0] == False:
                pixels_to_check.add((row, col - 1))

            if col < img_cols - 1 and self.filled_locations[row][col + 1][0] == False:
                pixels_to_check.add((row, col + 1))

            if row > 0 and self.filled_locations[row - 1][col][0] == False:
                pixels_to_check.add((row - 1, col))

            if row < img_rows - 1 and self.filled_locations[row + 1][col][0] == False:
                pixels_to_check.add((row + 1, col))

        indices = np.where(similar_colors)

        avg_color = np.average(self.img_array[indices], axis=0)
        self.img_array[indices] = avg_color

        self.filled_locations[indices] = True

    def get_img(self):
        return self.img_array

def median_filter(img: np.ndarray):
    print('Applying median filter...')
    filtered = ndimage.median_filter(img, 1)
    print('Done applying median filter!')
    return filtered
    

# def main():
#     img_array = imread('hill.jpg')

#     fig = plt.figure()
#     axes = plt.axes()

#     temp = FillColors(img_array, 15)
#     img_array = temp.get_img()

#     filtered = median_filter(img_array)

#     axes.imshow(filtered)
#     plt.show()

# if __name__ == "__main__":
#     main()
