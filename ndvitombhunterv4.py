# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 06:21:38 2021

@author: cosmi

tombhunterv2
"""


import cv2

flags = [i for i in dir(cv2) if i.startswith('COLOR_')]

import matplotlib.pyplot as plt

import numpy as np

#For RGB and HSV Color Space Plots

from matplotlib import colors

import random as rng
rng.seed(12345)    #random number seed

#for ndvi colorbars

from matplotlib import colors
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap


#image = misc.imread('DJI_0860.JPG')

NIR = cv2.imread('./testimages/DJI_0705.JPG')



#You’ll notice that it looks like the blue and red channels have been mixed up. 
#In fact, OpenCV by default reads images in BGR format. 
#You can use the cvtColor(image, flag) and the flag we looked at above to fix this:

NIR = cv2.cvtColor(NIR, cv2.COLOR_BGR2RGB)


#In HSV space, the red to orange color of plants in NIR are much more localized and visually separable. 
#The saturation and value of the oranges do vary, but they are mostly located 
#within a small range along the hue axis. 
#This is the key point that can be leveraged for segmentation.

#convert to HSV

hsv_NIR = cv2.cvtColor(NIR, cv2.COLOR_RGB2HSV)


#threshold vegetation using the red to orange colors of the vegetation in the NIR

# Red color
low_red = np.array([160, 105, 84])
high_red = np.array([179, 255, 255])

deep_red = np.array([150, 55, 65])
light_red = np.array([250, 200, 200])
#create a binary mask and threshold the image using the selected colors

red_mask = cv2.inRange(hsv_NIR, low_red, high_red)


#keep every pixel the same as the original image


#show results

#plt.subplot(1, 2, 1)
#plt.imshow(red_mask, cmap="gray")
#plt.subplot(1, 2, 2)

#use not gate bitwise here to separate the nir values around the tombs only

result2 = cv2.bitwise_not(NIR, NIR, mask=red_mask)

plt.imshow(result2)
plt.show()
#NDVI Processing

#compute, using the green threshold mask on the NDVI, the circles around
# potential cairn and/or tombs that maybe standing, submerged or buried in the soil.





ir = (result2[:,:,0]).astype('float')


# Get one of the IR image bands (all bands should be same)
#blue = image[:, :, 2]

#r = np.asarray(blue, float)

r = (result2[:,:,2]).astype('float')


#compute endvi here instead of standard ndvi

ndvi = np.true_divide(np.subtract(ir, r), np.add(ir, r))



# Display the results
output_name = './testimages/SegmentedInfraBlueNDVI.jpg'

#a nice selection of grayscale colour palettes
cols1 = ['blue', 'green', 'yellow', 'red']
cols2 =  ['gray', 'gray', 'red', 'yellow', 'green']
cols3 = ['gray', 'blue', 'green', 'yellow', 'red']

cols4 = ['black', 'gray', 'blue', 'green', 'yellow', 'red']

def create_colormap(args):
    return LinearSegmentedColormap.from_list(name='custom1', colors=cols3)

#colour bar to match grayscale units
def create_colorbar(fig, image):
        position = fig.add_axes([0.125, 0.19, 0.2, 0.05])
        norm = colors.Normalize(vmin=-1., vmax=1.)
        cbar = plt.colorbar(image,
                            cax=position,
                            orientation='horizontal',
                            norm=norm)
        cbar.ax.tick_params(labelsize=6)
        tick_locator = ticker.MaxNLocator(nbins=3)
        cbar.locator = tick_locator
        cbar.update_ticks()
        cbar.set_label("NDVI", fontsize=10, x=0.5, y=0.5, labelpad=-25)

fig, ax = plt.subplots()

image = ax.imshow(ndvi, cmap=create_colormap(colors))



#create_colorbar(fig, image)
extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig(output_name, dpi=600, transparent=True, bbox_inches=extent, pad_inches=0)

cv2.waitKey(0)

ndviimage = cv2.imread('./testimages/SegmentedInfraBlueNDVI.jpg')

ndviimage = cv2.GaussianBlur(ndviimage, (27, 27), 0)
canny_output = cv2.Canny(ndviimage, 0, 130)



contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


contours_poly = [None]*len(contours)
boundRect = [None]*len(contours)
centers = [None]*len(contours)
radius = [None]*len(contours)

for i, c in enumerate(contours):
    contours_poly[i] = cv2.approxPolyDP(c, 3, True)
    boundRect[i] = cv2.boundingRect(contours_poly[i])
    centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
    
    
drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    
    
for i in range(len(contours)):
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,))
    cv2.drawContours(drawing, contours_poly, i, color)
    cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
                  (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
    cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
    
    
cv2.imshow('Contours', drawing)

cv2.imwrite("result2.png", drawing)
cv2.waitKey(0)