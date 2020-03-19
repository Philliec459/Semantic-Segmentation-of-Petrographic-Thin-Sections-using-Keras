# ThinSection-image-segmentation-Keras
This repository was inspired from Divam Gupta's GitHub repository on Image Segmentation Keras:

https://github.com/divamgupta/image-segmentation-keras

This is a brilliant repository that has served as the foundation for numerous image segmentation applications that are available on the web (GitHub) today.

## Objectives
The primary goal for this repository is to discriminate 5 different image objects observed in typical clastic rock petrographic Thin Sections. This is still work in progress.  Our next objectives will be to classify Petrophysical Rock Types (PRT) and Petrophysical properties based on image segmentation alone, where our estimates will be based totally on Thin Section photomicrograph of the rock. 


## Typical Thin Section
The following image is an example of a typical Sandstone Thin Section similar to what we are working with in our training data:

![Image](5ts.png)


## Data used for Training
Our primary training data was setup similar to what Mr. Gupta had done for his repository that is mentioned above. Our only exception is that we have used Thin Section images for both training and testing using the following data structure:

    dataset1
      images_prepped_train
      images_prepped_test
      annotations_prepped_train
      annotations_prepped_test


For our training data we have used 40 Thin section images for the initial training with 40 matching annotation images that were created using the methodology explained in the following GitHub repository:

https://github.com/Philliec459/Create-Thin-Section-Image-Labels-for-Image-Segmentation-Training


## Annotated Images
We have created our own annotated images. Each annotated image has 5 labeled segments ranging from 1 to 5. This labeling represents the 5 distinguishable features observed in Thin Section. We first create a gray-level image on the Thin section and then partition the gray-level image data into different bins which become our labeled images:

    label = np.zeros(gradient.shape )

    label[gradient < 0.25] = 1 #black grains 
    label[gradient > 0.25] = 2 #darker grains
    label[gradient > 0.4]  = 3 #blue-dye epoxy or visual porosity  
    label[gradient > 0.6]  = 4 #darker grains 
    label[gradient > 0.75] = 5 #bright quartz grains   


## Training Code
For the training portion of the project we used 40 training images and 20 validation images in datset1. Due to confidentiality these data are not being provided in this repository. The following is our training code:

    from keras_segmentation.models.unet import vgg_unet
    from keras_segmentation.predict import model_from_checkpoint_path


    model = vgg_unet(
            n_classes=51 ,  
            input_height=416, 
            input_width=608 
    )


    model=model_from_checkpoint_path("weights/vgg_unet_1")

    model.train(
        train_images =  "dataset1/images_prepped_train/",
        train_annotations = "dataset1/annotations_prepped_train/",
        val_images="dataset1/images_prepped_test/" ,
        val_annotations="dataset1/annotations_prepped_test/",
        verify_dataset=True,
        #load_weights="weights/vgg_unet_1.0" ,
        optimizer_name='adadelta' , do_augment=True , augmentation_name="aug_all",    
        checkpoints_path = "weights/vgg_unet_1" , epochs=5
    )


    #Display the model's architecture
    model.summary()

    #Save the entire model to a HDF5 file.
    #The '.h5' extension indicates that the model should be saved to HDF5.
    model.save('vgg_unet_1.h5') 

    #predict an image from the training data
    out = model.predict_segmentation(
        checkpoints_path="weights/vgg_unet_1" , 
        inp="dataset1/images_prepped_test/43.jpg",
        out_fname="newout.png"
    )

    from keras_segmentation.predict import predict_multiple

    predict_multiple( 
      checkpoints_path="weights/vgg_unet_1" , 
      inp_dir="dataset1/images_prepped_test/" , 
      out_dir="weights/out/" 
    )


## Test Data
For the test image Thin Section data that is provided in this repository, we are providing 12 "dataset1/images_prepped_test/" images with 12 matching "dataset1/annotations_prepped_test/" image files. We ae using "review_images_Create_Labels_out_gray.py" to create the labeled, annotated images from the test imates. We are writing out the annotated images to "dataset1/annotations_prepped_test/". The labeled images are scaled from 1 to 5 representing the n_classes. However, it does not appear that the annotated images are used in the predictive mode. 


## Predicted Results
We are using the same code as shown above for testing except that we have commented out the training portion of the code (model.train) for our image segmentation predictions.  

The predict_multiple predicted Thin Sections for these test data were written to the "weights/out/" subdirectory. The following shows the process flow from the original Thin Section image, to the annotated image and finally to the predicted image shown in the upper right of the image below. 


![Image](Process_Histograms.png)


This above predicted image in the upper right is a RGB image scaled from 0 to 256. The histogram of the predicted image appears to illustrate the image segmentation. The histogram of the predicted images is in sharp contrast to the rather Gaussian distribution observed from the original Thin Section image on the left. The following image shows the frequencey histograms for all RGB values:


![Image](RGB_histogram.png)


We are predicting 5 primary image segments or objects on our test Thin Sections using the python program "main_load_your_own_model_from_existing_checkpoint.py" where the checkpoint weights being used are stored int the "weights/" subdirectory. We are not furnishing the original training data due to the proprietary nature of these data, however; our test images were downloaded from the web and are similar in Clastic sandstone Rock Types to the training data. 

(At this time I am unable to supply the vgg_unet_1 type of weights file due to a size restriction in GitHub.) 


## Way Forward 
The comparison of the of the original vs. the predicted, segmented Thin Section From the image below demonstrates that the image segmentation process is discriminating various features observed in the Thin Section. In the future we will want use this image segmentation to classify the Thin Sections into Petrophysical Rock Types (PRT) based on this process. Each PRT should also have similar Petrophysical properties (Porosity, Permeability) within each Rock Type. Since we have the Petrophysical property data avalable from our training data set, we could easily create regressions methods for these estimations. This would be the more traditional approach, however, we plan on performing this estimation directly from image segmentation.  


![Image](Preducted_comparison.png)

In the above image we have used the "interactive_plot.py" driven from a command line xterm to observe the RGB values of each segment. 

For this repository we have been working on a Ubuntu workstation, and each python program has been driven from an xterm command line. 


