import numpy as np
import tensorflow as tf
import os
import sys


# hiperparamaters
learning_rate = 0.001
epochs = 300
batch_size = 128
filter_count = 32

x = tf.placeholder(tf.float32, [None, 91*91])
x_shaped = tf.reshape(x, [-1, 91, 91, 1])
y = tf.placeholder(tf.int32, [None, 1])

conv1 = tf.contrib.layers.conv2d(
    inputs=x_shaped,
    num_outputs=filter_count,
    kernel_size=5,
    # stride=1,
    # padding='SAME',
    # activation_fn=tf.nn.relu,
    # weights_initializer=initializers.xavier_initializer(),
    # biases_initializer=tf.zeros_initializer(),
)
conv1_1 = tf.contrib.layers.conv2d(
    inputs=conv1,
    num_outputs=filter_count/2,
    kernel_size=1,
    # stride=1,
    # padding='SAME',
    # activation_fn=tf.nn.relu,
    # weights_initializer=initializers.xavier_initializer(),
    # biases_initializer=tf.zeros_initializer(),
)
conv2 = tf.contrib.layers.conv2d(
    inputs=conv1_1,
    num_outputs=filter_count,
    kernel_size=5,
    # stride=1,
    # padding='SAME',
    # activation_fn=tf.nn.relu,
    # weights_initializer=initializers.xavier_initializer(),
    # biases_initializer=tf.zeros_initializer(),
)
max_pool1 = tf.contrib.layers.max_pool2d(
    inputs=conv2,
    kernel_size=2,
    # stride=2,
    # padding='VALID',
)
conv3 = tf.contrib.layers.conv2d(
    inputs=max_pool1,
    num_outputs=filter_count*2,
    kernel_size=5,
    # stride=1,
    # padding='SAME',
    # activation_fn=tf.nn.relu,
    # weights_initializer=initializers.xavier_initializer(),
    # biases_initializer=tf.zeros_initializer(),
)
max_pool2 = tf.contrib.layers.max_pool2d(
    inputs=conv3,
    kernel_size=2,
    # stride=2,
    # padding='VALID',
)

# flattened = tf.reshape(max_pool2, [-1, 23 * 23 * 32])
flattened = tf.contrib.layers.flatten(max_pool2)

fc1 = tf.contrib.layers.fully_connected(
    inputs=flattened,
    num_outputs=100,
    # activation_fn=tf.nn.relu,
    # weights_initializer=initializers.xavier_initializer(),
    # biases_initializer=tf.zeros_initializer(),
)

fc2 = tf.contrib.layers.fully_connected(
    inputs=fc1,
    num_outputs=1,
    # activation_fn=tf.nn.relu,
    # weights_initializer=initializers.xavier_initializer(),
    # biases_initializer=tf.zeros_initializer(),
)

output = tf.cast(tf.round(fc2), tf.int32)

loss_mae = tf.losses.absolute_difference(y, fc2)
loss_mae_round = tf.losses.absolute_difference(y, output)

optimiser = tf.contrib.optimizer_v2.AdamOptimizer(
    learning_rate=learning_rate,
).minimize(loss_mae)

accuracy = tf.contrib.metrics.accuracy(
    labels=y,
    predictions=output
)

# setup the initialisation operator
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)

    images_train = np.load('images_train.npy')
    labels_training = np.load('labels_training.npy')
    images_test = np.load('images_test.npy')
    labels_test = np.load('labels_test.npy')

    total_batch = int(len(labels_training) / batch_size)

    for epoch in range(epochs):
        sum_loss = 0
        for i in range(total_batch):
            print("Epoch: ", (epoch + 1), "Batch: ",
                  i + 1, "/", total_batch, end="\r")

            batch_x = images_train[i*batch_size:i*batch_size+batch_size]
            batch_y = labels_training[i*batch_size:i*batch_size+batch_size]

            _, c = sess.run([optimiser, loss_mae_round], feed_dict={
                            x: batch_x, y: batch_y})
            sum_loss += c 

        avg_loss = sum_loss / total_batch

        # train_acc = sess.run(accuracy, feed_dict={
        #                     x: images_train[:500], y: labels_training[:500]})
        
        # Train acc after the epoch
        sum_acc = 0
        for i in range(total_batch):
            batch_x = images_train[i*batch_size:i*batch_size+batch_size]
            batch_y = labels_training[i*batch_size:i*batch_size+batch_size]
            train_acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})

            sum_acc += train_acc
        avg_acc = sum_acc / total_batch 

        print("Epoch:", (epoch + 1), ", loss:",
              "{:.3f}".format(avg_loss), ", train accuracy: {:.3f}".format(avg_acc))

    print("\nTraining complete!")


    total_batch = int(len(labels_test) / batch_size)

    sum_acc = 0
    for i in range(total_batch):
        batch_x = images_test[i*batch_size:i*batch_size+batch_size]
        batch_y = labels_test[i*batch_size:i*batch_size+batch_size]
        test_acc, prediction = sess.run([accuracy, output], feed_dict={x: batch_x, y: batch_y})

        sum_acc += test_acc
    avg_acc = sum_acc / total_batch 

    # acc, prediction = sess.run([accuracy, output], feed_dict={
    #                            x: images_test, y: labels_test})

    print("Test Accuracy:", avg_acc)
    print("prediction(last batch):", prediction.reshape(-1))
    print("labels_test(last batch:", batch_y.reshape(-1))
