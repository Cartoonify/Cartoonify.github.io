# Original Source of code and test data: https://github.com/mrgloom/skin-detection-example
# Modifications include refactoring code to remove of cv2 dependency, helper functions
# Modified by CS4476 group:
# # Joseph Lee
# # Kelsey Henson
# # Roger Nhan
# # Sabrina Chua
# # Lyndon Puzon

#Skin Segmentation Data Set from https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation
#face.png image from http://graphics.cs.msu.ru/ru/node/899

import numpy as np
from imageio import imread, imsave

from sklearn import tree
from sklearn.model_selection import train_test_split
from skimage.color import rgb2hsv

def ReadData():
    #Data in format [B G R Label] from
    data = np.genfromtxt('../data/Skin_NonSkin.txt', dtype=np.int32)

    labels= data[:,3]
    data= data[:,0:3]

    return data, labels

# modified by cs4476 project team (removal of cv2 dependency)
def BGR2HSV(bgr):
    bgr= np.reshape(bgr,(bgr.shape[0],1,3))
    hsv= rgb2hsv(rgb2bgrAndViceVersa(np.uint8(bgr)))
    hsv= np.reshape(hsv,(hsv.shape[0],3))

    return hsv

def TrainTree(data, labels, flUseHSVColorspace):
    if(flUseHSVColorspace):
        data= BGR2HSV(data)

    trainData, testData, trainLabels, testLabels = train_test_split(data, labels, test_size=0.20, random_state=42)

    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(trainData, trainLabels)

    return clf

# cs4476 project team
def rgb2bgrAndViceVersa(rgb):
    bgr = np.copy(rgb)
    for row in range(rgb.shape[0]):
        for col in range(rgb.shape[1]):
            bgr[row][col] = np.flip(rgb[row][col])
    return bgr

# Source: https://stackoverflow.com/a/58748986
def rgba2rgb( rgba, background=(255,255,255) ):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8' )

# modified by 4476 project team (remove cv2 dependency)
def ApplyToImage(path, flUseHSVColorspace):
    data, labels= ReadData()
    clf= TrainTree(data, labels, flUseHSVColorspace)

    # converting to bgr from rgb or rgba
    img= rgb2bgrAndViceVersa(rgba2rgb(imread(path)))

    data= np.reshape(img,(img.shape[0]*img.shape[1],3))

    if(flUseHSVColorspace):
        data= BGR2HSV(data)

    predictedLabels= clf.predict(data)

    imgLabels= np.reshape(predictedLabels,(img.shape[0],img.shape[1],1))
    saveImg = ((-(imgLabels-1)+1)*255).astype('uint8')
    if (flUseHSVColorspace):
        imsave('../results/result_HSV.png', saveImg)# from [1 2] to [0 255]
    else:
        imsave('../results/result_RGB.png', saveImg)

# cs4476 project team
def applyToArrayReturnArray(img, flUseHSVColorspace):
    data, labels= ReadData()    
    clf= TrainTree(data, labels, flUseHSVColorspace)

    # converting to bgr from rgb or rgba
    img= rgb2bgrAndViceVersa(rgba2rgb(img))

    data= np.reshape(img,(img.shape[0]*img.shape[1],3))

    if(flUseHSVColorspace):
        data= BGR2HSV(data)

    predictedLabels= clf.predict(data)

    imgLabels= np.reshape(predictedLabels,(img.shape[0],img.shape[1],1))
    saveImg = ((-(imgLabels-1)+1)*255).astype('uint8')
    return saveImg

#---------------------------------------------
# ApplyToImage("../face.png", True)
# ApplyToImage("../face.png", False)
# 
# imsave("result.png", applyToArrayReturnArray(imread("../../res/images/jogging_lowres.jpg"), True))
