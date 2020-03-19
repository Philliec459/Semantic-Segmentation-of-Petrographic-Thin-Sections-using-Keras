# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 09:41:41 2020

@author: craig
"""

from PIL import Image
import os.path


#import skimage
from skimage.color import rgb2gray
from skimage.filters import sobel,gaussian,hessian,frangi,laplace,median  
#    meijering, prewitt, prewitt_h, prewitt_v, sato
#from skimage.segmentation import random_walker
#from skimage.data import binary_blobs, moon, astronaut, hubble_deep_field, ts
from skimage.io import imsave
import numpy as np
import matplotlib.pyplot as plt



for i in range(1,13,1):


    TS=str(i)
    #offset_plus = 50
    img=os.path.join('dataset1/images_prepped_test', TS + '.png')

#    TS_Im = Image.open(img).convert('L')
    TS_Im = Image.open(img)
    
   
#    a = np.array(TS_Im) 
    data = np.array(TS_Im) 
    #gradient = np.array(TS_Im)
    #gradient = sobel(rgb2gray(data))  # true gradients
    gradient = gaussian(rgb2gray(data)) #averagint
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
    
    
    '''
    #------------------------------------------------------------------------------
    #
    #        Partition Image from 0 to n_classes
    #
    #------------------------------------------------------------------------------
            
    '''


    label = np.zeros(gradient.shape )
    
    label[gradient < 0.25] = 1 #black grains 
    label[gradient > 0.25] = 2 #darker grains
    label[gradient > 0.4]  = 3 #blue-dye epoxy or visual porosity  
    label[gradient > 0.6]  = 4 #darker grains 
    label[gradient > 0.75] = 5 #bright quartz grains   


    '''
    #------------------------------------------------------------------------------
    #
    #        Verification of Input and Labeled Results
    #
    #------------------------------------------------------------------------------
            
    '''
   
    
    
    plt.figure(0)
    plt.imshow(data)  #Original Image

    plt.figure(1)    
    histogram, bin_edges = np.histogram(data, bins=256, range=(0.001, 250))   
    plt.title(" Histogram Original Image")
    plt.xlabel(" value")
    plt.ylabel("pixels")  
    plt.plot(bin_edges[0:-1], histogram)  # <- or here


    plt.figure(2)
    plt.imshow(gradient)  #Original Image

   
    plt.figure(3)    
    histogram, bin_edges = np.histogram(gradient, bins=256, range=(0.0, 1))   
    plt.title(" Histogram Gradient Filter")
    plt.xlabel(" value")
    plt.ylabel("pixels")  
    plt.plot(bin_edges[0:-1], histogram)  # <- or here

    plt.figure(4)
    plt.imshow(data)  #Original Image


    plt.figure(5)
    plt.imshow(label)  #Original Image
    
    plt.figure(6) 
    histogram, bin_edges = np.histogram(label, bins=256, range=(0.0, 10))   
    plt.title(" Histogram Labels")
    plt.xlabel(" value")
    plt.ylabel("pixels")  
    plt.plot(bin_edges[0:-1], histogram)  # <- or here

    
 
    
### Original way of doing this.
#    img_out=os.path.join('label/', TS + '.png')
#
#    im = Image.fromarray(label)
#    im = im.convert("L")
#   
#    im.save(img_out)
    
#    ##### for images scaled 0 to 1
#    img_out=os.path.join('ClasticThinSectionsLabels', TS + '.png')
#    #############im = Image.fromarray(label)
#    ############im = label.convert("L")
#    imsave(img_out,label)
    
    ##### for images 0 to whatever

    '''
    #------------------------------------------------------------------------------
    #
    #        Save Label Image Results to dataset1/annotations_prepped_test
    #
    #------------------------------------------------------------------------------
            
    '''
 
    img_out=os.path.join('dataset1/annotations_prepped_test', TS + '.png')
    
    im = Image.fromarray(label)
    im = im.convert("L")
       
    im.save(img_out)

