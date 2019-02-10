## 1. Tensors:
tensors are N-Dimensional Arrays. tensors make it very convenient to feed in sets of images into our model -(I,H,W,C).

## 2. DNN vs CNN: 
DNN every neuron layer is dirrectly connected to the next neuron layer, for CNN we have different approach where each unit is connected to a smaller number of nearby units in next layer. having units only connected to nearby units also aids in invariance. CNN also helps with regularization, limiting the search of weights to the size of the convolution.

## 3. Convolutional and Filters

## 4. Padding

## 5. Pooling Layers: 
Pooling layers will subsample the input image, which reduces the memory use and computer load as well as reducing the number of parameters.

## 6. Dropout: 
another common technique deployed with CNN is called "Dropout".  Dropout can be thought of as a form of regularization to help prevent overfitting. during training, units are randomly dropped, along with their connections.


## Different CNN architectures:

1. LeNet-5 by Yann LeCun
2. AlexNet by Alex Krizhevsky et al
3. GoogLenet by Szegedy at Google Research
4. ResNet by Kaiming He et al.
