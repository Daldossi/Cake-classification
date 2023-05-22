# Cake-classification
Image classification through neural networks.

I have 1800 photos of desserts and I want the computer to tell me which one between 15 kinds (apple pie, cannoli, carrot cake, chocolate cake, chocolate mousse, churros, creme brulee, cup cakes, donuts, ice cream, macarons, panna cotta, red velvet cake, tiramisu, waffles) they represent. I built three classifiers.

1) The first one is a linear multilayer perceptron (MLP) that uses a low-level feature extraction (color histograms, edge direction histogram, gray-level co-occurence matrix, rgb-level co-occurrence matrix).
2) The second one is a linear MLP trained on the features made by the PVMLNet convolutional neural network without the last layer.
3) The third one performs a transfer learning, so the PVMLNet is deprived form its last layer and it is put a 15-vector of classification on its place; then the weights and the biases are re-trained with a very low learning rate.  

Finally an analysis of the performance is done.
