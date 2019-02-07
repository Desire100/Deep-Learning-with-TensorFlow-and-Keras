Learning Rate: defines the step size during gradient deecent.

Batch Size: batches allow us to use stochastic gradient descent. Smaller and bigger correspond to less representative of data and longer training time respectively.

Second-Order Behavior of the gradient descent allows us to adjust our learning rate based off the rate of descent 
    1. AdaGrad
    2. RMSProp
    3. Adam
    this allows us to start with larger steps and then eventually go to smaller step sizes. Adam allows this change to happend         automatically.
    
Unstable/Vanishing gradients: As you increase the number of layers in a network, the layers towards the input will be affected less by the error calculation occuring at the output as you backwards through the network. Initialization and Normalization will help us mitigate these issues.

Overfitting vs Underfitting: if you are fitting very well to your training data but you get a larger error on your test data your model is overfitting. with potentially hundreds of parameters in a deep learning neural network, the possibility of overfitting is very high! there are a few ways to help mitigate this issue. L1/L2 Regularization: Adds a penalty for larger weights in the model.

Dropout: remove neurons during training randomly.

Expanding Data: Artificially expand data by adding noise. tilting images, adding low white noise to sound data, etc...
