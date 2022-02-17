# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 23:02:58 2020

@author: cosmi
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap

import warnings

warnings.filterwarnings("ignore")

cap = cv2.VideoCapture('./testvideos/loughcrewnir.mp4')




#custom colormap for ndvi greyscale video

cols3 = ['gray', 'blue', 'green', 'yellow', 'red']

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


while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of red NIR vegetation color in HSV
    low_red = np.array([160, 105, 84])
    high_red = np.array([179, 255, 255])

    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, low_red, high_red)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    
    
    #NDVI Processing
    ir = (res[:,:,0]).astype('float')
    r = (res[:,:,2]).astype('float')
    
    ndvi = np.true_divide(np.subtract(ir, r), np.add(ir, r))
    
    cols3 = ['gray', 'blue', 'green', 'yellow', 'red']
    
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

    
    
    image = plt.imshow(ndvi, cmap=create_colormap(colors))
    #plt.axis('off')
    #image = cv2.imshow(ndvi, cmap=create_colormap(colors))
    

    #this step adds considerable processing, be sure to use only 720p files at most a minute long
    #cv2.imshow('ndvi',ndvi)
    
    
    src_height, src_width, src_channels = frame.shape

    max_value = src_height * src_width * 255
        
    
    #locking onto the ndvi areas of interest
    
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(ndvi)

        
    #img = cv2.medianBlur(res, 5)
    ccimg = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    cimg = cv2.cvtColor(ccimg, cv2.COLOR_BGR2GRAY)
    
    #ret, circles = cv2.HoughCircles(cimg, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=20, maxRadius=30)
        

    # Find changes in the masked grayscale image
    ret, contours = cv2.findContours(cimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours is not None:
        print("tomb is found")
        contours = np.uint16(np.around(contours))
        Val = float(mask.sum()) / float(max_value)
        print(Val)
        for i in contours[0, :]:
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
            
            cv2.circle(ndvi, maxLoc, 50, (179, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('flash', frame)
            cv2.imshow('detected circles', cimg)
            cv2.imshow('res', res)
            cv2.imshow('ndvi (greyscale)', ndvi)

        else:
            print("Read Failed")


#wait for q key to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()