from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import sys


images_train = np.load('images_training_oversampled.npy')
labels_training = np.load('labels_training_oversampled.npy')
images_validation = np.load('images_validation.npy')
labels_validation = np.load('labels_validation.npy')
images_test = np.load('images_test.npy')
labels_test = np.load('labels_test.npy')



labels_training_new = []

for label in labels_training:
    zeros_array = np.zeros(80)
    zeros_array[label-1] = 1
    labels_training_new.append(zeros_array)

labels_training_new = np.array(labels_training_new)



labels_validation_new = []

for label in labels_validation:
    zeros_array = np.zeros(80)
    zeros_array[label-1] = 1
    labels_validation_new.append(zeros_array)

labels_validation_new = np.array(labels_validation_new)




labels_test_new = []

for label in labels_test:
    zeros_array = np.zeros(80)
    zeros_array[label-1] = 1
    labels_test_new.append(zeros_array)

labels_test_new = np.array(labels_test_new)

# hyperparamaters
# learning_rate = 0.001
epochs = 300
# batch_size = 50
# filter_count = 32

for learning_rate in [0.001]:
    for batch_size in [100]:
        for filter_count in [32]:
            for maxpool_size in [2]:
                for hidden_nodes in [100]:

                    outfile_name = "lr" + str(learning_rate) + "_bs" + str(batch_size) + "_fc" + str(
                        filter_count) + "_mps"+str(maxpool_size) + "_hn"+str(hidden_nodes)

                    print("Starting " + outfile_name)

                    x = tf.placeholder(tf.float32, [None, 91*91])
                    x_shaped = tf.reshape(x, [-1, 91, 91, 1])
                    y = tf.placeholder(tf.int32, [None, 80])

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

                    conv2_bn = tf.contrib.layers.batch_norm(conv2,
                                                            center=True,
                                                            scale=True
                                                            )
                    max_pool1 = tf.contrib.layers.max_pool2d(
                        inputs=conv2_bn,
                        kernel_size=maxpool_size,
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
                    conv3_bn = tf.contrib.layers.batch_norm(conv3,
                                                            center=True,
                                                            scale=True
                                                            )
                    max_pool2 = tf.contrib.layers.max_pool2d(
                        inputs=conv3_bn,
                        kernel_size=maxpool_size,
                        # stride=2,
                        # padding='VALID',
                    )

                    # flattened = tf.reshape(max_pool2, [-1, 23 * 23 * 32])
                    flattened = tf.contrib.layers.flatten(max_pool2)

                    fc1 = tf.contrib.layers.fully_connected(
                        inputs=flattened,
                        num_outputs=hidden_nodes,
                        # activation_fn=tf.nn.relu,
                        # weights_initializer=initializers.xavier_initializer(),
                        # biases_initializer=tf.zeros_initializer(),
                    )
                    # batch norm layer between fully connected layers!
                    fc1_bn = tf.contrib.layers.batch_norm(fc1,
                                                          center=True,
                                                          scale=True
                                                          )
                    fc2 = tf.contrib.layers.fully_connected(
                        inputs=fc1_bn,
                        num_outputs=80,
                        activation_fn=tf.nn.softmax
                        # weights_initializer=initializers.xavier_initializer(),
                        # biases_initializer=tf.zeros_initializer(),
                    )

                    output = tf.argmax(fc2)

                    # y_new = []

                    # for label in y:
                    #     zeros_array = np.zeros((1, 80))
                    #     zeros_array[label-1] = 1
                    #     y_new.append(zeros_array)
                    
                    # y_new = np.array(y_new)

                    loss_mae = tf.losses.absolute_difference(y, fc2)
                    loss_mae_round = tf.losses.absolute_difference(tf.argmax(y), output)

                    optimiser = tf.contrib.optimizer_v2.AdamOptimizer(
                        learning_rate=learning_rate,
                    ).minimize(loss_mae)

                    # accuracy = tf.contrib.metrics.accuracy(
                    #     labels=y,
                    #     predictions=output
                    # )

                    train_total_batch = int(len(labels_training) / batch_size)
                    validation_total_batch = int(len(labels_validation) / batch_size)

                    # setup the initialisation operator
                    init_op = tf.global_variables_initializer()

                    with tf.Session() as sess:
                        # initialise the variables
                        sess.run(init_op)

                        train_losses = []
                        validation_losses = []
                        for epoch in range(epochs):
                            sum_loss = 0
                            for i in range(train_total_batch):
                                print("Epoch: ", (epoch + 1), "Batch: ",
                                      i + 1, "/", train_total_batch, end="\r")

                                batch_x = images_train[i *
                                                       batch_size:i*batch_size+batch_size]
                                batch_y = labels_training_new[i *
                                                          batch_size:i*batch_size+batch_size]

                                _, c = sess.run([optimiser, loss_mae_round], feed_dict={
                                                x: batch_x, y: batch_y})
                                sum_loss += c

                            train_avg_loss = sum_loss / train_total_batch

                            # Validation loss after the epoch
                            sum_loss = 0
                            for i in range(validation_total_batch):
                                batch_x = images_validation[i *
                                                            batch_size:i*batch_size+batch_size]
                                batch_y = labels_validation_new[i *
                                                            batch_size:i*batch_size+batch_size]
                                l = sess.run(loss_mae_round, feed_dict={
                                             x: batch_x, y: batch_y})

                                sum_loss += l
                            validation_avg_loss = sum_loss / validation_total_batch

                            train_losses.append(train_avg_loss)
                            validation_losses.append(validation_avg_loss)

                            print("Epoch:", (epoch + 1), ", train loss:", "{:.3f}".format(
                                train_avg_loss), ", validation loss: {:.3f}".format(validation_avg_loss))

                        with open("./results/" + outfile_name + '_train.txt', 'w') as f:
                            for item in train_losses:
                                f.write("%s\n" % item)

                        with open("./results/" + outfile_name + '_validation.txt', 'w') as f:
                            for item in validation_losses:
                                f.write("%s\n" % item)

                        print("\nTraining complete!")

                        test_total_batch = int(len(labels_test) / batch_size)

                        sum_loss = 0
                        for i in range(test_total_batch):
                            batch_x = images_test[i *
                                                  batch_size:i*batch_size+batch_size]
                            batch_y = labels_test_new[i *
                                                  batch_size:i*batch_size+batch_size]
                            test_loss, prediction = sess.run(
                                [loss_mae_round, output], feed_dict={x: batch_x, y: batch_y})

                            sum_loss += test_loss
                        test_avg_loss = sum_loss / test_total_batch

                        # acc, prediction = sess.run([accuracy, output], feed_dict={
                        #                            x: images_test, y: labels_test})

                        # print("Test Loss:", test_avg_loss)
                        # print("prediction(last batch):",
                        #       prediction.reshape(-1))
                        # print("labels_test(last batch):", batch_y.reshape(-1))


# sm_train_losses = savgol_filter(train_losses, 25, 3)
# sm_validation_losses = savgol_filter(validation_losses, 25, 3)

# plt.plot(sm_train_losses, 'r')
# plt.plot(sm_validation_losses, 'b')
# plt.show()
