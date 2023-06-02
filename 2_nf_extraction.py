# -*- coding: utf-8 -*-
"""
@author: alice
Script: neural feature extraction using the pre-trained neural network pvmlnet
"""

import numpy as np 
import matplotlib.pyplot as plt 
import os 
import pvml 
# from image_features import *


def extract_neural_features(im, cnn, remove_layer):
    """ im:image, cnn:convolutional neural network """
    ## a list of arrays, each one represents the activation values for 1 of the layers in the network:
    activations = cnn.forward(im[None, :, :, :]) # from 224x224x3 to 1x224x224x3 
    # for act in activations: 
    #     print(act.shape) 
    features = activations[-remove_layer] 
    ## now features is a 4-dimensional array with size=1 over the first three dimensions 
    features = features.reshape(-1) # remove the 3 dimensions getting a vector 1024 
    return features 

def process_directory(path, cnn, remove_layer): 
    """ extract the features of all the images in that particular order """ 
    all_features = [] 
    all_labels = [] 
    klass_label = 0 
    for klass in classes: 
        image_files = os.listdir(path + "/" + klass) 
        # image_files = [c for c in image_files if not c.startswith(".")] # row in case of Apple computer
        for imagename in image_files: 
            image_path = path + "/" + klass + "/" + imagename 
            image = plt.imread(image_path) / 255.0 # from integer to floating value 
            # print(image_path) 
            features = extract_neural_features(image, cnn, remove_layer) 
            all_features.append(features) 
            all_labels.append(klass_label) 
        klass_label += 1 
    X = np.stack(all_features, 0) 
    Y = np.array(all_labels) 
    return X, Y 



###############################################################################
# MAIN FUNCTION
###############################################################################

## COLLECT all the classes

classes = os.listdir('images/train') 
classes.sort() # store in alphabetical order 
# print(classes) 

## LOAD the pre-trained network 

cnn = pvml.CNN.load("pvmlnet.npz") 

## PERFORM feature extraction and SAVE the results

remove_layer = 3 # -3 corresponds to the last hidden layer 
## Test set
X, Y = process_directory("images/test", cnn, remove_layer)
# print("test", X.shape, Y.shape) 
data = np.concatenate([X, Y[:, None]], 1) 
np.savetxt("test_nf3.txt.gz", data) 
## Training set
X, Y = process_directory("images/train", cnn, remove_layer) 
# print("train", X.shape, Y.shape) 
data = np.concatenate([X, Y[:, None]], 1) 
np.savetxt("train_nf3.txt.gz", data) 
