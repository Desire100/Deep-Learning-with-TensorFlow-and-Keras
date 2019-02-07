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
