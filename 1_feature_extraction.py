# -*- coding: utf-8 -*-
"""
@author: alice
script: low-level feature extraction 
"""
import numpy as np 
import matplotlib.pyplot as plt 
import os # to scan the directory
from image_features import *


def minmax_normalization(Xtrain, Xtest): 
    """ Min-max scaling """ 
    xmin = Xtrain.min(0) 
    xmax = Xtrain.max(0) 
    Xtrain = (Xtrain - xmin) / (xmax - xmin) 
    Xtest = (Xtest - xmin) / (xmax - xmin) 
    return Xtrain, Xtest  


def process_directory(path, feature_f): 
    """ Extract the features of all the images in that particular order """
    all_features = [] 
    all_labels = [] 
    klass_label = 0
    for klass in classes: 
        image_files = os.listdir(path + "/" + klass) 
        for imagename in image_files: 
            image_path = path + "/" + klass + "/" + imagename 
            image = plt.imread(image_path) / 255.0 
            # print(image_path) 
            # print(image.shape, image.dtype) 
            # plt.imshow(image) 
            # plt.show() 
            # return 
            features = feature_f(image)
            # features = features.reshape(64 * 3) # reshape the image so that is is all stored in a single vector
            features = features.reshape(-1) # equivalent to the previous row
            # print(features.shape) # output: (3,64) because it is 1 for red, 1 for green and 1 for blue
            # plt.bar(np.arange(features.shape[0]), features) 
            all_features.append(features) 
            all_labels.append(klass_label)
        klass_label += 1 
    X = np.stack(all_features, 0) # take the lists and turn them into numpy arrays
    Y = np.array(all_labels) 
    return X, Y 



###############################################################################
# MAIN FUNCTION 
###############################################################################

## COLLECT all the classes

classes = os.listdir('images/train') 
classes.sort() # store in alphabetical order
print(classes) 


## EXTRACT the features   

Xtrain_edh, Ytrain_edh = process_directory("images/train", edge_direction_histogram) 
datatrain_edh = np.concatenate([Xtrain_edh, Ytrain_edh[:, None]], 1) 
Xtest_edh, Ytest_edh = process_directory("images/test", edge_direction_histogram) 
datatest_edh = np.concatenate([Xtest_edh, Ytest_edh[:, None]], 1)
datatrain_edh, datatest_edh = minmax_normalization(Xtrain_edh, Xtest_edh) 
np.savetxt("train_edh.txt.gz", datatrain_edh) 
np.savetxt("test_edh.txt.gz", datatest_edh) 

Xtrain_ch, Ytrain_ch = process_directory("images/train", color_histogram) 
datatrain_ch = np.concatenate([Xtrain_ch, Ytrain_ch[:, None]], 1) 
Xtest_ch, Ytest_ch = process_directory("images/test", color_histogram) 
datatest_ch = np.concatenate([Xtest_ch, Ytest_ch[:, None]], 1)
datatrain_ch, datatest_ch = minmax_normalization(Xtrain_ch, Xtest_ch) 
np.savetxt("train_ch.txt.gz", datatrain_ch) 
np.savetxt("test_ch.txt.gz", datatest_ch) 

Xtrain_cm, Ytrain_cm = process_directory("images/train", cooccurrence_matrix) 
datatrain_cm = np.concatenate([Xtrain_cm, Ytrain_cm[:, None]], 1) 
Xtest_cm, Ytest_cm = process_directory("images/test", cooccurrence_matrix) 
datatest_cm = np.concatenate([Xtest_cm, Ytest_cm[:, None]], 1) 
datatrain_cm, datatest_cm = minmax_normalization(Xtrain_cm, Xtest_cm) 
np.savetxt("train_cm.txt.gz", datatrain_cm)
np.savetxt("test_cm.txt.gz", datatest_cm)  

Xtrain_rcm, Ytrain_rcm = process_directory("images/train", rgb_cooccurrence_matrix) 
# print("train_rcm", X.shape, Y.shape) 
datatrain_rcm = np.concatenate([Xtrain_rcm, Ytrain_rcm[:, None]], 1) 
Xtest_rcm, Ytest_rcm = process_directory("images/test", rgb_cooccurrence_matrix) 
# print("test_rcm", X.shape, Y.shape) 
datatest_rcm = np.concatenate([Xtest_rcm, Ytest_rcm[:, None]], 1)
datatrain_rcm, datatest_rcm = minmax_normalization(Xtrain_rcm, Xtest_rcm) 
np.savetxt("train_rcm.txt.gz", datatrain_rcm) 
np.savetxt("test_rcm.txt.gz", datatest_rcm) 

datatrain_edh_ch = np.concatenate([Xtrain_edh, Xtrain_ch, Ytrain_edh[:, None], Ytrain_ch[:, None]], 1) 
datatest_edh_ch = np.concatenate([Xtest_edh, Xtest_ch, Ytest_edh[:, None], Ytest_ch[:, None]], 1) 
# datatrain_edh_ch, datatest_edh_ch = minmax_normalization(datatrain_edh_ch, datatest_edh_ch) 
np.savetxt("train_edh_ch.txt.gz", datatrain_edh_ch) 
np.savetxt("test_edh_ch.txt.gz", datatest_edh_ch) 

datatest_rcm_ch = np.concatenate([Xtest_rcm, Xtest_ch, Ytest_rcm[:, None], Ytest_ch[:, None]], 1)
datatrain_rcm_ch = np.concatenate([Xtrain_rcm, Xtrain_ch, Ytrain_rcm[:, None], Ytrain_ch[:, None]], 1)
# datatrain_rcm_ch, datatest_rcm_ch = minmax_normalization(datatrain_rcm_ch, datatest_rcm_ch) 
np.savetxt("train_rcm_ch.txt.gz", datatrain_rcm_ch) 
np.savetxt("test_rcm_ch.txt.gz", datatest_rcm_ch)

datatest_edh_ch_cm = np.concatenate([Xtest_edh, Xtest_ch, Xtest_cm, Ytest_edh[:, None], Ytest_ch[:, None], Ytest_cm[:, None]], 1)
datatrain_edh_ch_cm = np.concatenate([Xtrain_edh, Xtrain_ch, Xtrain_cm, Ytrain_edh[:, None], Ytrain_ch[:, None], Ytrain_cm[:, None]], 1)
# datatrain_edh_ch_cm, datatest_edh_ch_cm = minmax_normalization(datatrain_edh_ch_cm, datatest_edh_ch_cm) 
np.savetxt("train_edh_ch_cm.txt.gz", datatrain_edh_ch_cm) 
np.savetxt("test_edh_ch_cm.txt.gz", datatest_edh_ch_cm) 
