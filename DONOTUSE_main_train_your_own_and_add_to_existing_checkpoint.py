#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:08:34 2020

@author: craig
"""

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



model.train(
    checkpoints_path = "weights/vgg_unet_1" ,
    #load_weights = "weights/vgg_unet_1.4"  ,   
    train_images =  "dataset1/images_prepped_train/",
    train_annotations = "dataset1/annotations_prepped_train/",
    val_images="dataset1/images_prepped_test/" ,
    val_annotations="dataset1/annotations_prepped_test/",
    verify_dataset=True,
    optimizer_name='adadelta' , do_augment=True , augmentation_name="aug_all",    
    epochs=1
)


# Display the model's architecture
model.summary()

# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model should be saved to HDF5.
model.save('vgg_unet_1.h5') 

##predict an image from the training data
#out = model.predict_segmentation(
#    checkpoints_path="weights/vgg_unet_1" , 
#    inp="dataset1/images_prepped_test/43.jpg",
#    out_fname="newout.png"
#)





## Show Images
#img = Image.open("dataset1/images_prepped_test/43.jpg")
#plt.figure(1)
#plt.imshow(img)
#
#plt.figure(2)
#plt.imshow(out)



#from keras_segmentation.predict import predict
#
#
#predict( 
#	checkpoints_path="weights/vgg_unet_1"  , 
#	inp= "dataset1/images_prepped_test/43.jpg" , 
#	out_fname="43_predict.png" 
#)




from keras_segmentation.predict import predict_multiple


predict_multiple( 
	checkpoints_path="weights/vgg_unet_1" , 
	inp_dir="dataset1/images_prepped_test/" , 
	out_dir="weights/out/" 
)




#evaluating the model 
#print(model.evaluate_segmentation( inp_images_dir="dataset1/images_prepped_test/"  , annotations_dir="dataset1/annotations_prepped_test/" ) )