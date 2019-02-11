### 1. GRAPHS:
graphs are sets of connected nodes (vertices). the connections are referred to as edges. in tensorflow each node is an operation with possible inputs that can supply some output.In general, with tensorflow we will construct a graph and then excute it.

There are two main types of tensor objects in a gaph; variables and placeholders.
During the optimization process TensorFlow tunes the parameters of the model.

### 2. Variables：
Variables can hold the values of weights and biases throughout the session. they need to be initialized.
### 3. Placeholders：
Placeholders are initially empty and are used to feed in the actual training examples and that is your actual data that you are training your model on. However they do need a declared expected data type(tf.float32) with an optional shape argument.
