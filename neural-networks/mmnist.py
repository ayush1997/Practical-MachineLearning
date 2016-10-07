from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print mnist.train.images.shape
print mnist.train.labels.shape

#
# mnist = fetch_mldata('MNIST original')
# X_train = mnist.data
# print X_train.shape
# y = mnist.target.reshape(-1,1)
# enc = OneHotEncoder()
# Y = enc.fit_transform(y).toarray()
# print Y.shape

# x_o = X_train
# y_o = Y
def plot_image():
    print X_train[0]
    plt.imshow(np.reshape(X_train[0],(28,28)),cmap="gray")
    plt.show()

# plot_image()
# X_train = X_train[:50000]
# Y = Y[:50000]
# X_valid = x_o[50000:]
# Y_valid = y_o[50000:]
# print X_valid.shape
#
# batch_x=[]
# batch_y=[]
# batch_size = 50
# n_epochs = 5
#
# i=0
# j=batch_size
# batch_x =[]
# batch_y = []
#
# for _ in range(X_train.shape[0]/batch_size):
#     batch_x.append(X_train[i:j])
#     batch_y.append(Y[i:j])
#     i = i+batch_size
#     j = j+batch_size
#
# print np.array(batch_x).shape
# print np.array(batch_y).shape
#
# for i in batch_x:
#     print i

n_input =784
n_output  =10

g = tf.get_default_graph()

X  = tf.placeholder(tf.float32,[None,n_input])
Y = tf.placeholder(tf.float32,[None,n_output])

with tf.name_scope('inits'):
    W = tf.get_variable(
                name="W",
                shape=[n_input, n_output],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())

    b = tf.get_variable(
        name='b',
        shape=[n_output],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0))

    h = tf.nn.bias_add(
        name='h',
        value=tf.matmul(X, W),
        bias=b)


Y_pred = tf.nn.softmax(h)

cross_entropy = -tf.reduce_sum(Y * tf.log(Y_pred + 1e-12))
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(Y_pred,Y)
optimizer = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

#Monitor accuracy
predicted_y = tf.argmax(Y_pred, 1)
actual_y = tf.argmax(Y, 1)

correct_prediction = tf.equal(predicted_y, actual_y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

n_epochs=3
for epoch in range(n_epochs):
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})


    print str(epoch) + "-------------------------------------"
    print(sess.run(accuracy, feed_dict={X: mnist.test.images,Y: mnist.test.labels}))

# o = [op.name for op in g.get_operations()]
# for i in o:
#     print i
W = g.get_tensor_by_name('W:0')
W_arr = np.array(W.eval(session=sess))
print(W_arr.shape)
print W_arr[:,0]
plt.imshow(W_arr[:,0].reshape(28,28),cmap='gray')
plt.show()
# fig, ax = plt.subplots(1, 10, figsize=(20, 3))
# for col_i in range(10):
#     ax[col_i].imshow(W_arr[:, col_i].reshape((28, 28)), cmap='coolwarm')
# plt.show()
