from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import sys


images_train = np.load('./../data/images_train.npy')
labels_train = np.load('./../data/labels_train.npy')
images_validation = np.load('./../data/images_validation.npy')
labels_validation = np.load('./../data/labels_validation.npy')
images_test = np.load('./../data/images_test.npy')
labels_test = np.load('./../data/labels_test.npy')

def run_udacity_model(learning_rate = 0.001, epochs = 100, batch_size = 50, dr_keep=0.5, l2_scale=0.1, filter_coef=1):

  outfile_name = "udacity_epoch" + str(epochs) + "_lr" + str(learning_rate) + "_bs" + str(batch_size) + "_drk" + str(dr_keep) + "_l2" + str(l2_scale) + "_fc" + str(filter_coef)

  print("Starting " + outfile_name)

  x = tf.placeholder(tf.float32, [None, 91*91], name="placeholer_x")
  x_shaped = tf.reshape(x, [-1, 91, 91, 1])
  y = tf.placeholder(tf.int32, [None, 1], name="placeholer_y")
  train_mode = tf.placeholder(tf.bool, name="placeholer_train_mode")
  train_mode_dr = tf.placeholder(tf.bool, name="placeholer_train_mode_dr")

  conv1 = tf.contrib.layers.conv2d(
  inputs=x_shaped,
  num_outputs=24*filter_coef,
  kernel_size=5,
  stride=2,
  padding='VALID',
  # activation_fn=tf.nn.relu,
  # weights_initializer=initializers.xavier_initializer(),
  # biases_initializer=tf.zeros_initializer(),
  )
  
  bn1 = tf.layers.batch_normalization(conv1,
                                     training=train_mode
                                     )
  mp1 = tf.contrib.layers.max_pool2d(
                                      inputs=bn1,
                                      kernel_size=3,
                                      stride=1,
                                      padding='SAME'
                                     )
  
  conv2 = tf.contrib.layers.conv2d(
  inputs=bn1,
  num_outputs=36*filter_coef,
  kernel_size=5,
  stride=2,
  padding='VALID',
  # activation_fn=tf.nn.relu,
  # weights_initializer=initializers.xavier_initializer(),
  # biases_initializer=tf.zeros_initializer(),
  )
  
  conv3 = tf.contrib.layers.conv2d(
  inputs=conv2,
  num_outputs=48*filter_coef,
  kernel_size=5,
  stride=2,
  padding='VALID',
  # activation_fn=tf.nn.relu,
  # weights_initializer=initializers.xavier_initializer(),
  # biases_initializer=tf.zeros_initializer(),
  )
  
  bn2 = tf.layers.batch_normalization(conv3,
                                     training=train_mode
                                     )
  mp2 = tf.contrib.layers.max_pool2d(
                                    inputs=bn2,
                                    kernel_size=3,
                                    stride=1,
                                    padding='SAME'
                                   )
  
  conv4 = tf.contrib.layers.conv2d(
  inputs=bn2,
  num_outputs=64*filter_coef,
  kernel_size=3,
  stride=1,
  padding='VALID',
  # activation_fn=tf.nn.relu,
  # weights_initializer=initializers.xavier_initializer(),
  # biases_initializer=tf.zeros_initializer(),
  )
  
  conv5 = tf.contrib.layers.conv2d(
  inputs=conv4,
  num_outputs=64*filter_coef,
  kernel_size=3,
  stride=1,
  padding='VALID',
  # activation_fn=tf.nn.relu,
  # weights_initializer=initializers.xavier_initializer(),
  # biases_initializer=tf.zeros_initializer(),
  )
  
    
  bn3 = tf.layers.batch_normalization(conv5,
                                     training=train_mode
                                     )
                                     
  flattened = tf.contrib.layers.flatten(bn3)

  
  dr1 = tf.contrib.layers.dropout(
                                  inputs=flattened,
                                  keep_prob=dr_keep,
                                  is_training=train_mode_dr
                                 )
  
  fc1 = tf.contrib.layers.fully_connected(
  inputs=dr1,
  num_outputs=1164,
  # activation_fn=tf.nn.relu,
  # weights_initializer=initializers.xavier_initializer(),
  # biases_initializer=tf.zeros_initializer(),
  )
                                     
    
  bn4 = tf.layers.batch_normalization(fc1,
                                     training=train_mode
                                     )
      
  dr2 = tf.contrib.layers.dropout(
                                  inputs=bn4,
                                  keep_prob=dr_keep,
                                  is_training=train_mode_dr
                                 )
  
  fc2 = tf.contrib.layers.fully_connected(
  inputs=dr2,
  num_outputs=100,
  # activation_fn=tf.nn.relu,
  # weights_initializer=initializers.xavier_initializer(),
  # biases_initializer=tf.zeros_initializer(),
  )
                                     
    
  bn5 = tf.layers.batch_normalization(fc2,
                                     training=train_mode
                                     )
      
  dr3 = tf.contrib.layers.dropout(
                                  inputs=bn5,
                                  keep_prob=dr_keep,
                                  is_training=train_mode_dr
                                 )
  
  fc3 = tf.contrib.layers.fully_connected(
  inputs=dr3,
  num_outputs=50,
  # activation_fn=tf.nn.relu,
  # weights_initializer=initializers.xavier_initializer(),
  # biases_initializer=tf.zeros_initializer(),
  )
                                     
    
  bn6 = tf.layers.batch_normalization(fc3,
                                     training=train_mode
                                     )
  
  dr4 = tf.contrib.layers.dropout(
                                  inputs=bn6,
                                  keep_prob=dr_keep,
                                  is_training=train_mode_dr
                                 )
  
  fc4 = tf.contrib.layers.fully_connected(
  inputs=dr4,
  num_outputs=10,
  # activation_fn=tf.nn.relu,
  # weights_initializer=initializers.xavier_initializer(),
  # biases_initializer=tf.zeros_initializer(),
  )
                                     
    
  bn7 = tf.layers.batch_normalization(fc4,
                                     training=train_mode
                                     )
      
  dr5 = tf.contrib.layers.dropout(
                                  inputs=bn7,
                                  keep_prob=dr_keep,
                                  is_training=train_mode_dr
                                 )
  
  fc5 = tf.contrib.layers.fully_connected(
  inputs=dr5,
  num_outputs=1,
  # activation_fn=tf.nn.relu,
  # weights_initializer=initializers.xavier_initializer(),
  # biases_initializer=tf.zeros_initializer(),
  )

  #one = tf.constant(1, dtype=tf.int32)
  #eighty = tf.constant(80, dtype=tf.int32)
  
  output = tf.cast(tf.round(fc5), tf.int32)
  output = tf.clip_by_value(output, 1, 80)
  #output = tf.cond(tf.less(output, one), lambda: one, lambda: output)
  #output = tf.cond(tf.less(eighty, output), lambda: eighty, lambda: output)
  
  loss_mae = tf.losses.mean_squared_error(y, fc5)
  loss_mae_round = tf.losses.absolute_difference(y, output)
  
  
  reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  loss_mae += l2_scale * sum(reg_losses)

  optimiser = tf.contrib.optimizer_v2.AdamOptimizer(learning_rate=learning_rate,).minimize(loss_mae)

  train_total_batch = int(len(labels_train) / batch_size)
  validation_total_batch = int(len(labels_test) / batch_size)

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
            #print("Epoch: ", (epoch + 1), "Batch: ",
            #      i + 1, "/", train_total_batch, end="\r")

            batch_x = images_train[i*batch_size:i*batch_size+batch_size]
            batch_y = labels_train[i*batch_size:i*batch_size+batch_size]

            _, c = sess.run([optimiser, loss_mae_round], feed_dict={x: batch_x, y: batch_y, train_mode:True, train_mode_dr:True})
            sum_loss += c

        train_avg_loss = sum_loss / train_total_batch

        # Validation loss after the epoch

        batch_x = images_validation
        batch_y = labels_validation
        validation_avg_loss = sess.run(loss_mae_round, feed_dict={x: batch_x, y: batch_y, train_mode:True, train_mode_dr:False})

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

run_udacity_model(learning_rate = 0.01, epochs = 300, batch_size = 200, dr_keep=0.6, l2_scale=0.0, filter_coef=2)