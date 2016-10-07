import tensorflow as tf
import numpy as np

from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

g = tf.get_default_graph()

X = tf.placeholder(tf.float32,[None,784])
Y = tf.placeholder(tf.float32,[None,10])

X_tensor = tf.reshape(X,[-1,28,28,1])

with tf.name_scope("layer1"):

    filter_size = 5
    num_filteres_in = 1
    num_filteres_out = 32

    W_1 = tf.get_variable(
        name='W1',
        shape=[filter_size,filter_size,num_filteres_in,num_filteres_out],
        initializer  =tf.random_normal_initializer()
        )

    b_1 = tf.get_variable(
        name='b1',
        shape=[num_filteres_out],
        initializer=tf.constant_initializer()
    )

    h_1 = tf.nn.relu(
        tf.nn.bias_add(
            tf.nn.conv2d(input=X_tensor,
                         filter=W_1,
                         strides=[1, 2, 2, 1],
                         padding='SAME'),
            b_1))



with tf.name_scope("layer2"):
    n_filters_in = 32
    n_filters_out = 64
    W_2 = tf.get_variable(
        name='W2',
        shape=[filter_size, filter_size, n_filters_in, n_filters_out],
        initializer=tf.random_normal_initializer())
    b_2 = tf.get_variable(
        name='b2',
        shape=[n_filters_out],
        initializer=tf.constant_initializer())
    h_2 = tf.nn.relu(
        tf.nn.bias_add(
            tf.nn.conv2d(input=h_1,
                     filter=W_2,
                     strides=[1, 2, 2, 1],
                     padding='SAME'),
            b_2))


# We'll now reshape so we can connect to a fully-connected/linear layer:
h_2_flat = tf.reshape(h_2, [-1, 7 * 7 * n_filters_out])

with tf.name_scope('fc_1'):
    W_3 = tf.get_variable(
                name="W_3",
                shape=[7 * 7 * n_filters_out, 128],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())

    b_3 = tf.get_variable(
        name='b_3',
        shape=[128],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0))

    h_3 = tf.nn.relu(
        tf.nn.bias_add(
        name='h_3',
        value=tf.matmul(h_2_flat, W_3),
        bias=b_3))

n_output = 10
with tf.name_scope('fc_2'):
    W_4 = tf.get_variable(
                name="W",
                shape=[128,n_output],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())

    b_4 = tf.get_variable(
        name='b',
        shape=[10],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0))

    Y_pred = tf.nn.softmax(
        tf.nn.bias_add(
        name='h',
        value=tf.matmul(h_3, W_4),
        bias=b_4))


cross_entropy = -tf.reduce_sum(Y * tf.log(Y_pred + 1e-12))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

predicted_y = tf.argmax(Y_pred, 1)
actual_y = tf.argmax(Y, 1)

correct_prediction = tf.equal(predicted_y, actual_y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

# o = [op.name for op in g.get_operations()]
# for i in o:
#     print i

n_epochs=3
for epoch in range(n_epochs):
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
        if i%300==0:
            print(sess.run(accuracy, feed_dict={X: mnist.test.images,Y: mnist.test.labels}))
        # W1 = g.get_tensor_by_name('W1:0')
        # W1 = np.array(W1.eval(session=sess))
        # print W1.shape
        # print W1[0][0][0][0]
        # print(W1[:,:,:,0].reshape(5,5).shape)
        #
        # print  W1[:,:,:,0].reshape(5,5)
        # print W1[:,:,:,0]
    print str(epoch) + "-------------------------------------"
    print(sess.run(accuracy, feed_dict={X: mnist.test.images,Y: mnist.test.labels}))


W1 = g.get_tensor_by_name('W1:0')
W1 = np.array(W1.eval(session=sess))

fig, ax = plt.subplots(1, 32, figsize=(20, 3))
for col_i in range(32):
    ax[col_i].imshow(W1[:,:,:,col_i].reshape((5,5)), cmap='coolwarm')
plt.show()
