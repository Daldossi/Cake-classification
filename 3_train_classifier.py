# -*- coding: utf-8 -*-
"""
@author: alice
script: train the classifier
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pvml


def accuracy(network, X, Y):
    """ Calculate the accuracy of X, calling Y the true values """
    predictions, probs = network.inference(X)
    accuracy = (predictions == Y).mean()
    return accuracy



###############################################################################
# MAIN FUNCTION
###############################################################################

## LOAD the data saved in the feature extraction process

data = np.loadtxt("train_edh_ch_cm.txt.gz")
Xtrain = data[:, :-1]
Ytrain = data[:, -1].astype(int)
data = np.loadtxt("test_edh_ch_cm.txt.gz")
Xtest = data[:, :-1]
Ytest = data[:, -1].astype(int)

## CONFIGURATION OF A MLP

nclasses = Ytrain.max() + 1 # nclasses = 15:number of classes 
mlp = pvml.MLP([Xtrain.shape[1], nclasses]) # no hidden layers 
# mlp = pvml.MLP([Xtrain.shape[1], 64, nclasses]) # a single hidden layer of 64 neurons

plt.ion() 
batch_size = 50 
steps = Xtrain.shape[0] // batch_size 
train_accs = [] 
test_accs = [] 
epochs = [] 
for epoch in range(5000): 
    mlp.train(Xtrain, Ytrain, lr=0.0001, batch=batch_size, steps=steps)
    if epoch % 100 == 0:
        train_acc = accuracy(mlp, Xtrain, Ytrain)
        test_acc = accuracy(mlp, Xtest, Ytest)
        print(f"epoca:{epoch}, train:{train_acc * 100:.1f} test:{test_acc * 100:.1f}") 
        train_accs.append(train_acc * 100) 
        test_accs.append(test_acc * 100) 
        epochs.append(epoch) 
        plt.clf() 
        plt.plot(epochs, train_accs) 
        plt.plot(epochs, test_accs) 
        plt.legend(["train", "test"]) 
        plt.title("edh + ch + cm")
        plt.pause(0.01) 
print(f"epoca:{epoch}, train:{train_acc * 100:.1f} test:{test_acc * 100:.1f}") 
        
## Saving the result
mlp.save("cakes-mlp_edh_ch_cm.npz") 
plt.ioff() 
plt.show() 
