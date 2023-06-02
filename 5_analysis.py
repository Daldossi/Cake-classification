# -*- coding: utf-8 -*-
"""
@author: alice
Script: analysis linear MLP
"""

import numpy as np 
import pvml 
import matplotlib.pyplot as plt 
import os 


def make_confusion_matrix(predictions, labels): 
    """ build the confusion matrix """
    cmat = np.zeros((15, 15)) 
    errs = [] 
    for i in range(len(predictions)): # for each pair of prediction in the true label
        cmat[labels[i], predictions[i]] += 1 
        # if predictions[i] != labels[i]: 
        #     errs.append([i, labels[i], int(predictions[i])]) 
    s = cmat.sum(1, keepdims=True) 
    cmat /= s 
    return cmat

def dispaly_confusion_matrix(cmat): 
    """ display a confusion matrix cmat such that: 
        - the columns represent the predictions
        - the rows represent the correct labels """
    print(" " * 10, end="") 
    for j in range(15): 
        print(f"{classes[j][:4]:4} ", end="")  
    print() 
    for i in range(15): 
        print(f"{classes[i]:5}", end="") 
        for j in range(15): 
            val = cmat[i, j] * 100 
            print(f"{val:4.1f} ", end="") 
        print() 

def dispaly_confusion_matrix2(cmat): 
    """ display the image of the confusion matrix cmat such that: 
        - the columns represent the predictions
        - the rows represent the correct labels """
    plt.imshow(cmat, cmap="Blues") 
    for i in range(15): 
        # print(f"{words[i]:5}", end="") 
        for j in range(15): 
            val = cmat[i, j] * 100 
            # print(f"{val:4.1f} ", end="")
            plt.text(j-0.25, i+0.1, int(val)) 
    plt.title("Confusion matrix")
    plt.xlabel("Predictions")
    plt.ylabel("Labels")
    plt.show() 
    print(" " * 10, end="") 
    for j in range(15): 
        print(f"{classes[j]:10} ", end="") 
    print() 
    # plt.savefig("matrix.pdf") 
    


###############################################################################
# MAIN FUNCTION
###############################################################################

# LOAD the data and the classes 

classes = os.listdir("images/test") 
# classes = [c for c in classes if not c.startswith(".")] # row in case of an Apple computer
classes.sort() # sort the classes 
data = np.loadtxt("test_nf3.txt.gz") 
Xtest = data[:, :-1] 
Ytest = data[:, -1].astype(int) 
cnn = pvml.CNN.load("cake-cnn.npz") 


## CONFUSION MATRIX 

predictions = [] 
labels = [] 
paths = [] 
klass_label = 0 
errs = [] 
i = 0 
for klass in classes: 
    image_files = os.listdir("images/test" + "/" + klass) 
    for imagename in image_files: 
        image_path = "images/test" + "/" + klass + "/" + imagename 
        paths.append(image_path) 
        image = plt.imread(image_path) / 255.0 # from integer to floating value
        preds, probs = cnn.inference(image[None, :, :, :]) 
        prob = probs.max() 
        predictions.append(preds) 
        labels.append(klass_label) 
        i += 1 
        if preds != klass_label: 
            errs.append([prob, i, klass_label, int(preds)]) 
    klass_label += 1 

cmat = make_confusion_matrix(predictions, labels) 
dispaly_confusion_matrix(cmat) 
dispaly_confusion_matrix2(cmat) 


## WORST ERRORS 

n = 5 # take the first n misclassification (with the highest probability)
errs.sort(reverse = True)
for i in range(n): 
    error = errs[i]
    path = paths[error[1]] 
    image = plt.imread(path) / 255.0 
    plt.imshow(image) 
    plt.show() 
    print(f"{path} Prediction:{classes[error[3]]} Probability:{error[0] * 100:.1f} Label:{classes[error[2]]}")
    