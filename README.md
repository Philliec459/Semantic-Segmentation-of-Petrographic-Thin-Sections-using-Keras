# ThinSection-image-segmentation-keras
This repository was inspired from Divam Gupta's GitHub repository on Image Segmentation Keras:

https://github.com/divamgupta/image-segmentation-keras

Our primary goal in this repository is to discriminate 5 different types of grains observed in clastic petrographic thin sections. This is still work in progress. We will also try to estimate the Petrophysical Rock Types and Petrophysical properties from a clastic Thin Section photomicrograph as the project progresses.

This is an example of a typiclal clastic thin section that we are working with in our test data:

![Image](5ts.png)

This is an example of the predicted output from this process. 

![Image](5predict.png)

We are predicting 5 primary segment types predicted from the python program main_load_your_own_model_from_existing_checkpoint.py where the checkpoint weights are stored int the 'weights/' subdirectory. We are not furnishing the original training data due to the proprietary nature of these data. Our test images were taken from the web trying to obtain similar thin sections from clastic rock as was the training data. 

(At this time I am unable to supply the vgg_unet_1 type of weights file due to a size restriction in GitHub. I also have one computer running training epochs too for future uploads of the checkpoint weights). 

One issue that we are having is re-coloring the predicted results to be a bit more intuitive. Please see the example below as our first attempt. 

![Image](5_recolor.png)

For this step we are using the review_predictions_and_create_better_color_pedictions.py program code gathering the predictions from 'weights/out/' subdirectory and writing out our new color presentations to the 'weights/out_color/' subdirectory. 

We are working in Ubuntu and each python program can be driven from an xterm or Spyder. 


