#!/usr/bin/python

import tensorflow as tf
import numpy as np

x_data = np.array([
[0,0], [0,1], [1,0], [1,1]
])
y_data = np.array([
[0], [1], [1], [0]
])

#hyperparameters
n_input = 2
n_hidden = 10
n_output = 1
learning_rate = 0.1
epochs = 10000

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#weights
W1 = tf.Variable(tf.random_uniform([n_input, n_hidden], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([n_hidden, n_output], -1.0, 1.0))

#bias
b1 = tf.Variable(tf.zeros([n_hidden]), name="Bias1")
b2 = tf.Variable(tf.zeros([n_output]), name="Bias2")

L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
hy = tf.sigmoid(tf.matmul(L2, W2) + b2)

cost = tf.reduce_mean(-Y*tf.log(hy) - (1-Y)*tf.log(1-hy))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(init)

    for step in xrange(epochs):
        session.run(optimizer, feed_dict={X: x_data, Y: y_data})

        if step % 1000 == 0:
            print session.run(cost, feed_dict={X: x_data, Y: y_data})

    answer = tf.equal(tf.floor(hy + 0.5), Y)
    accuracy = tf.reduce_mean(tf.cast(answer, "float"))

    print session.run([hy], feed_dict={X: x_data, Y: y_data})
    print "Accuracy: ", accuracy.eval({X: x_data, Y: y_data}) * 100 , "%"
