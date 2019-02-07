#TensorFlow basics

# GRAPHS
"""
we are going to create the  graph which contains three nodes, 
two nodes are going to hold numbers 1 and 2 respectively and then
put them in an operation (add) of node 3 then out put their sum
"""

import tensorflow as tf 

n1 = tf.constant(1) # the first node which holds a number = 1
n2 = tf.constant(2)
n3 = n1 + n2

with tf.Session() as sess:
    result = sess.run(n3)
    
result

""" when you start a tensorflow, a default graph is  created 
but you can also create an additional graph
"""
print(tf.get_default_graph)

# a different graph
g = tf.Graph()

#create

graph_one = (tf.get_default_graph)
graph_two = (tf.Graph)

#Set graph_two as a default graph

with graph_two.as_default():
    print(graph_two is tf.get_default_graph())
    

# VARIABLES and PLACEHOLDERS

my_tensor = tf.random_uniform((4,4),0,1)
my_var = tf.Variable(initial_value = my_tensor)

# initialize the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    result = sess.run(my_var)
    
result

ph = tf.placeholder(tf.float32, shape(None,5))
