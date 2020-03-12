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


#for i in range(1,39,1):    #offset = int(input('Input Integer Offset = '))
#for i in range(40,60,1):    #offset = int(input('Input Integer Offset = '))
for i in range(1,13,1):    #offset = int(input('Input Integer Offset = '))

    TS=str(i)
    #offset_plus = 50
#    img=os.path.join('dataset1/annotations_prepped_train', TS + '.png') #1-39
#    img=os.path.join('dataset1/annotations_prepped_test', TS + '.png') #40-60
#    img=os.path.join('dataset1/annotations_prepped_test', TS + '.png') #1-12 for new test set
    img=os.path.join('ClasticThinSectionsLabels', TS + '.png')
#    img=os.path.join('ClasticThinSectionsLabels2', TS + '.png')
#    TS_Im = Image.open(img).convert('L')
    TS_Im = Image.open(img)
    
   
#    a = np.array(TS_Im) 
    data = np.array(TS_Im) 
    #gradient = np.array(TS_Im)
    #gradient = sobel(rgb2gray(data))  # true gradients
#    gradient = gaussian(rgb2gray(data)) #averagint
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
    
    


 


   
    
    
    plt.figure(0)
    plt.imshow(data)  #Original Image

    plt.figure(1)    
    histogram, bin_edges = np.histogram(data, bins=256, range=(0.0, 10))   
    plt.title(" Histogram Original Image")
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
#    imsave(img_out,data)
    
#    ##### for images 0 to whatever
#    img_out=os.path.join('ClasticThinSectionsLabels2', TS + '.png')
#    
#    im = Image.fromarray(data)
#    im = im.convert("L")
#       
#    im.save(img_out)

   

