#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:08:34 2020

@author: craig
"""
import cv2
import numpy as np

from keras_segmentation.models.unet import vgg_unet
from keras_segmentation.predict import model_from_checkpoint_path
from PIL import Image
#from keras.models import load_model
import matplotlib.pyplot as plt

model = vgg_unet(
        n_classes=51 ,  
        input_height=416, 
        input_width=608 
)



model=model_from_checkpoint_path("weights/vgg_unet_1")


#model = load_model('vgg_unet_1.h5')

# =============================================================================
# model.train(
#     train_images =  "dataset1/images_prepped_train/",
#     train_annotations = "dataset1/annotations_prepped_train/",
#     val_images="dataset1/images_prepped_test/" ,
#     val_annotations="dataset1/annotations_prepped_test/",
#     verify_dataset=True,
# #    load_weights="weights/vgg_unet_1.4" ,
#     optimizer_name='adadelta' , do_augment=True , augmentation_name="aug_all",    
#     checkpoints_path = "weights/vgg_unet_1" , epochs=10
# )
# =============================================================================

# Display the model's architecture
model.summary()

# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model should be saved to HDF5.
model.save('vgg_unet_1.h5') 



from keras_segmentation.predict import predict


predict( 
	checkpoints_path="weights/vgg_unet_1"  , 
	inp= "dataset1/images_prepped_test/3.png" , 
	out_fname="3_predict.png" 
)




from keras_segmentation.predict import predict_multiple


predict_multiple( 
	checkpoints_path="weights/vgg_unet_1" , 
	inp_dir="dataset1/images_prepped_test/" , 
	out_dir="weights/out/" 
)





'''
#------------------------------------------------------------------------------
#
#        Verification of Input and Predicted Results
#
#------------------------------------------------------------------------------
        
'''
# Show Images
img = Image.open("dataset1/images_prepped_test/3.png")
plt.figure(1)
plt.imshow(img)

plt.figure(2)    
histogram, bin_edges = np.histogram(img, bins=256, range=(0.0, 250))   
plt.title(" Histogram Original TS")
plt.xlabel(" value")
plt.ylabel("pixels")  
plt.plot(bin_edges[0:-1], histogram)  # <- or





anno = Image.open("dataset1/annotations_prepped_test/3.png")
plt.figure(3)
plt.imshow(anno)

plt.figure(4)    
histogram, bin_edges = np.histogram(anno, bins=256, range=(0.0, 10))   
plt.title(" Histogram Annotated Prepped")
plt.xlabel(" value")
plt.ylabel("pixels")  
plt.plot(bin_edges[0:-1], histogram)  # <- or





out = Image.open("weights/out/3.png")
plt.figure(5)
plt.imshow(out)

plt.figure(6)    
histogram, bin_edges = np.histogram(out, bins=256, range=(0.0, 256))   
plt.title(" Histogram Prediction")
plt.xlabel(" value")
plt.ylabel("pixels")  
plt.plot(bin_edges[0:-1], histogram)  # <- or here









# =============================================================================
# '''
# When the image file is read with the OpenCV function imread(), the order of colors is BGR (blue, green, red). On the other hand, in Pillow, the order of colors is assumed to be RGB (red, green, blue).
# 
# Therefore, if you want to use both the Pillow function and the OpenCV function, you need to convert BGR and RGB.
# 
# You can use the OpenCV function cvtColor()
# '''
# im_cv = cv2.imread('weights/out/3.png')
# cv2.imwrite('test.jpg', im_cv)
# 
# #pil_img = Image.fromarray(im_cv)
# #pil_img.save('test_bgr.jpg')
# 
# '''
# Various color spaces such as RGB, BGR, HSV can be mutually converted using OpenCV function cvtColor().
# 
# dst = cv2.cvtColor(src, code)
# '''
# #im_rgb = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
# #Image.fromarray(im_rgb).save('test_bgr_rgb.jpg')
# 
# im_hsv = cv2.cvtColor(im_cv, cv2.COLOR_RGB2HSV)
# Image.fromarray(im_hsv).save('test_bgr_hsv.jpg')
# 
# =============================================================================

#evaluating the model 
#print(model.evaluate_segmentation( inp_images_dir="dataset1/images_prepped_test/"  , annotations_dir="dataset1/annotations_prepped_test/" ) )