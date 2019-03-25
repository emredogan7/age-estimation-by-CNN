import numpy as np
import tensorflow as tf
import os


path_training = "./../data/training/" 
path_test = "./../data/test/"

filenames_training = os.listdir(path_training)
np.random.shuffle(filenames_training)
filenames_test = os.listdir(path_test)

filepaths_training = [path_training + f for f in filenames_training]
labels_training = [int(x[:3])-1 for x in filenames_training]

filepaths_test = [path_test + f for f in filenames_test]
labels_test = [int(x[:3])-1 for x in filenames_test]


learning_rate = 0.0001
epochs = 10
batch_size = 32
image_side = 91 
image_size = image_side * image_side
out_label_size = 80
one_hot_size = len(set(labels_test))

# declare the training data placeholders
# input x - for 28 x 28 pixels = 784 - this is the flattened image data that is drawn from 
# mnist.train.nextbatch()
x = tf.placeholder(tf.float32, [None, image_size])
# dynamically reshape the input
x_shaped = tf.reshape(x, [-1, image_side, image_side, 1])
# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, out_label_size])

def one_hot(y, size):
    return np.eye(size)[y]

def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                      num_filters]

    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                                      name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

    # add the bias
    out_layer += bias

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # now perform max pooling
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, 
                               padding='SAME')

    return out_layer

# create some convolutional layers
layer1 = create_new_conv_layer(x_shaped, 1, 32, [5, 5], [2, 2], name='layer1')
layer2 = create_new_conv_layer(layer1, 32, 64, [5, 5], [2, 2], name='layer2')
flattened = tf.reshape(layer2, [-1, 23 * 23 * 64])

# setup some weights and bias values for this layer, then activate with ReLU
wd1 = tf.Variable(tf.truncated_normal([23 * 23 * 64, 1000], stddev=0.03), name='wd1')
bd1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')
dense_layer1 = tf.matmul(flattened, wd1) + bd1
dense_layer1 = tf.nn.relu(dense_layer1)

# another layer with softmax activations
wd2 = tf.Variable(tf.truncated_normal([1000, out_label_size], stddev=0.03), name='wd2')
bd2 = tf.Variable(tf.truncated_normal([out_label_size], stddev=0.01), name='bd2')
dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
y_ = tf.nn.softmax(dense_layer2)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y))

# add an optimiser
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# setup the initialisation operator
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)
    total_batch = int(len(filepaths_training) / batch_size)


    feed_test_images = np.array([tf.image.decode_jpeg(tf.read_file(x)).eval().reshape(-1) for x in filepaths_test])
    feed_test_labels = one_hot(labels_test, one_hot_size)

    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x = np.array([tf.image.decode_jpeg(tf.read_file(x)).eval().reshape(-1) for x in filepaths_training[i*batch_size:i*batch_size+batch_size]])
            batch_y = one_hot(labels_training[i*batch_size:i*batch_size+batch_size], one_hot_size)
            _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch

        test_acc = sess.run(accuracy, feed_dict={x: feed_test_images, y: feed_test_labels})
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), "test accuracy: {:.3f}".format(test_acc))

    print("\nTraining complete!")
    print(sess.run(accuracy, feed_dict={x: feed_test_images, y: feed_test_labels}))

