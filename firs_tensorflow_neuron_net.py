# First neuron network

import numpy as np
import tensorflow as tf

#Set random seeds
np.random.seed(101)
tf.set_random_seed(101)

#Data set up
"""
Setting Up some Random Data for Demonstration Purposes
"""
rand_a = np.random.uniform(0,100,(5,5))
rand_b = np.random.uniform(0,100,(5,1))

# Placeholders

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

# Operations 
add_op = a+b # tf.add(a,b)
mult_op = a*b #tf.multiply(a,b)

# Running Sessions to create Graphs with Feed Dictionaries
with tf.Session() as sess:
    add_result = sess.run(add_op,feed_dict={a:rand_a,b:rand_b})
    print(add_result)
    
    print('\n')
    
    mult_result = sess.run(mult_op,feed_dict={a:rand_a,b:rand_b})
    print(mult_result)
    
# Example Neural Network

n_features = 10
n_dense_neurons = 3

# Placeholder for x
x = tf.placeholder(tf.float32,(None,n_features))
# Variables for w and b
b = tf.Variable(tf.zeros([n_dense_neurons]))

W = tf.Variable(tf.random_normal([n_features,n_dense_neurons]))

# Operation Activation Function 
xW = tf.matmul(x,W)
z = tf.add(xW,b)

# tf.nn.relu() or tf.tanh()
a = tf.sigmoid(z)

# Variable Intializer! 

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    layer_out = sess.run(a,feed_dict={x : np.random.random([1,n_features])})
    
print(layer_out)

#Full Network Example

"""
Let's work on a regression example, we are trying to solve a very simple equation:

y = mx + b

y will be the y_labels and x is the x_data.
 We are trying to figure out the slope and
 the intercept for the line that best fits our data!
 """
 
# Artifical Data (Some Made Up Regression Data)
 
x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
y_label = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)

import matplotlib.pyplot as plt
plt.plot(x_data, y_label,'*')
plt.show

# Variables 

np.random.rand(2)
m = tf.Variable(0.39)
b = tf.Variable(0.2)

# Cost Function

error = 0

for x,y in zip(x_data,y_label):
    
    y_hat = m*x + b  #Our predicted value
    
    error += (y-y_hat)**2 # The cost we want to minimize (we'll need to use an optimization function for the minimization!)

# Optimizer

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

# Initialize Variables
init = tf.global_variables_initializer()

# Create Session and Run!

with tf.Session() as sess:
    
    sess.run(init)
    
    epochs = 100
    
    for i in range(epochs):
        
        sess.run(train)
        

    # Fetch Back Results
    final_slope , final_intercept = sess.run([m,b])
    
final_slope

final_intercept

# Evaluate Results

x_test = np.linspace(-1,11,10)
y_pred_plot = final_slope*x_test + final_intercept

plt.plot(x_test,y_pred_plot,'r')

plt.plot(x_data,y_label,'*')


