# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 09:41:41 2020

@author: craig
"""

from PIL import Image, ImageOps 
import os.path


#import skimage
from skimage.color import rgb2gray
from skimage.filters import sobel,gaussian,hessian,frangi,laplace,median
#, meijering, prewitt, prewitt_h, prewitt_v, sato
#from skimage.segmentation import random_walker
#from skimage.data import binary_blobs, moon, astronaut, hubble_deep_field, ts

import numpy as np
import matplotlib.pyplot as plt
import cv2




#for i in range(1,61,1):    #offset = int(input('Input Integer Offset = '))
for i in range(3,4,1):    #offset = int(input('Input Integer Offset = '))


    TS=str(i)
    #offset_plus = 50


    img=os.path.join('weights/out/', TS + '.png')    
#    img=os.path.join('weights/out_color/', TS + '.png')

#   TS_Im = Image.open(img).convert('L')
    TS_Im = Image.open(img)
#    TS_Im = Image.open(img)
    
   

    a = np.array(TS_Im) 
    data = np.array(TS_Im) 
    #gradient = np.array(TS_Im)
    #gradient = sobel(rgb2gray(data))  # true gradients

    gradient = (1-gaussian(rgb2gray(data)))*256 #averagint
    #gradient = (gaussian(rgb2gray(data))) #averagint
    #gradient = median(rgb2gray(data)) #average this is best for thin sections scaled 0 to 256
    
    #gradient = hessian(rgb2gray(data)) # picks out grains
    #gradient = frangi(rgb2gray(data)) # really weird 
    #gradient = laplace(rgb2gray(data)) # creates higher grains and lower matrix extremely well 
    #gradient = meijering(rgb2gray(data)) #shows paths ofpossible flow????
    #gradient = prewitt(rgb2gray(data)) #gradient like sobel
    #gradient = sobel(rgb2gray(data))  # true gradients
    #gradient = prewitt_h(rgb2gray(data)) #creates higher grains and lower matrix extremely well 
    #gradient = prewitt_v(rgb2gray(data)) #creates higher grains and lower matrix extremely well with vertical bias
    #gradient = sato(rgb2gray(data)) #weidrd like frangi
    


# =============================================================================
# Partition Histogram into grains and porosity
# =============================================================================
    
    label = np.zeros(gradient.shape )

#    label = gradient 


    label[gradient < 40]   = 10   #dark grains 
    label[gradient > 40]   = 100  #epoxy

    label[gradient > 120]  = 200  # big quartz grains
    label[gradient > 130]   = 100  #epoxy
    label[gradient > 170]  = 150  #  dark Quartz   

# =============================================================================
# Plots
# =============================================================================

    
    plt.figure(0)
    plt.imshow(data)  #Predicted Image

    plt.figure(1)    
    histogram, bin_edges = np.histogram(data, bins=256, range=(0.0, 256))   
    plt.title(" Histogram Predicted RGB Image")
    plt.xlabel(" value")
    plt.ylabel("pixels")  
    plt.plot(bin_edges[0:-1], histogram)  # <- or here



    plt.figure(2)
    plt.imshow(gradient)  #Gradient Image
    
    plt.figure(3)    
    histogram, bin_edges = np.histogram(gradient, bins=256, range=(0.0, 256))   
    plt.title(" Histogram Gradient Gray-Level Image")
    plt.xlabel(" value")
    plt.ylabel("pixels")  
    plt.plot(bin_edges[0:-1], histogram)  # <- or here

    
    
    
    plt.figure(6) 
    histogram, bin_edges = np.histogram(gradient, bins=256, range=(0.001, 50))   
    plt.title(" Histogram Gradient 0 to 50")
    plt.xlabel(" value")
    plt.ylabel("pixels")  
    plt.plot(bin_edges[0:-1], histogram)  # <- or her

    plt.figure(7) 
    histogram, bin_edges = np.histogram(gradient, bins=256, range=(50, 100))   
    plt.title(" Histogram Gradient 50 to 100")
    plt.xlabel(" value")
    plt.ylabel("pixels")  
    plt.plot(bin_edges[0:-1], histogram)  # <- or her



    plt.figure(8) 
    histogram, bin_edges = np.histogram(gradient, bins=256, range=(100, 150))   
    plt.title(" Histogram Gradient 100 to 150")
    plt.xlabel(" value")
    plt.ylabel("pixels")  
    plt.plot(bin_edges[0:-1], histogram)  # <- or her

    plt.figure(9) 
    histogram, bin_edges = np.histogram(gradient, bins=256, range=(150, 255))   
    plt.title(" Histogram Gradient 150 to 255")
    plt.xlabel(" value")
    plt.ylabel("pixels")  
    plt.plot(bin_edges[0:-1], histogram)  # <- or her
    
    plt.figure(10)
    plt.imshow(label)  #Labeled Image
 
    plt.figure(11) 
    histogram, bin_edges = np.histogram(label, bins=256, range=(0.001, 255))   
    plt.title(" Histogram Labels")
    plt.xlabel(" value")
    plt.ylabel("pixels")  
    plt.plot(bin_edges[0:-1], histogram)  # <- or here
  
    
    
    

    img_out=os.path.join('weights/out_color', TS + '.png')
    im = Image.fromarray(label)
    im = im.convert("L")
    im.save(img_out)


#img = Image.open('sample_images/1_output.png')







    #img_out=os.path.join('weights/out_color', TS + '.png')
#    plt.imshow(label, cmap="gnuplot")
    #save image in color
    ##plt.imsave(img_out, label, cmap="gnuplot")

    
    
    
    
    
    
    
#
#    plt.figure(6) 
#    plt.imshow()

#    img_out=os.path.join('weights/out_gray', TS + '.png')
##    plt.imshow(label, cmap="gnuplot")
#    #save image in color
#    plt.imsave(img_out, label, cmap="gnuplot")