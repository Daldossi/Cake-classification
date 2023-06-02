# -*- coding: utf-8 -*"-
""""
@author: alice
Script: compute transfer learning
"""

import numpy as np 
import pvml 
import matplotlib.pyplot as plt 
import os 


## LOAD the 2 different networks

cnn = pvml.CNN.load("pvmlnet.npz") 
mlp = pvml.MLP.load("cakes-mlp_nf3.npz") 


## MODIFY the cnn to substitute its last layer with the last one of the mlp 

## Weights: from a matrix to a 4-dim array (1x1x1024x15) 
cnn.weights[-1] = mlp.weights[0][None, None, :, :] 
## Biases 
cnn.biases[-1] = mlp.biases[0] 
cnn.save("cake-cnn.npz") # with this cnn I can read the class of an input-image 


## CLASSIFY an image 

imagepath = "images/test/ice_cream/45200.jpg" 

image = plt.imread(imagepath) / 255 
predictions, probs = cnn.inference(image[None, :, :, :]) 
## cnn.inference(image[a, b, c, d]) # a: changes the image, b:number of rows, c:number of columns, d:color numbers 

classes = os.listdir("images/test") 
# classes = [c for c in classes if not c.startswith(".")] # row in case of an Apple computer
classes.sort() # sort the classes 

indices = (-probs[0]).argsort() # sort the probabilities 
for k in range(5): 
    index = indices[k] 
    print(f"{k+1} {classes[index]:10} {probs[0][index] * 100:.1f}") 

plt.imshow(image) 
plt.show() 
