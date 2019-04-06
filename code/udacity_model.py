import numpy as np
import tensorflow as tf
import os


images_train = np.load('./data/images_train.npy')
labels_train = np.load('./data/labels_train.npy')
images_validation = np.load('./data/images_validation.npy')
labels_validation = np.load('./data/labels_validation.npy')
images_test = np.load('./data/images_test.npy')
labels_test = np.load('./data/labels_test.npy')

def udacity_model(learning_rate = 0.001, dr_keep=0.5, l2_scale=0.1, filter_coef=1):
  x = tf.placeholder(tf.float32, [None, 91*91], name="features_placeholder")
  x_shaped = tf.reshape(x, [-1, 91, 91, 1])
  y = tf.placeholder(tf.int32, [None, 1], name="labels_placeholder")
  train_mode = tf.placeholder(tf.bool, name="train_mode_placeholder")

  conv1 = tf.contrib.layers.conv2d(
                                   inputs=x_shaped,
                                   num_outputs=24*filter_coef,
                                   kernel_size=5,
                                   stride=2,
                                   padding='VALID',
                                   # activation_fn=tf.nn.relu,
                                   # weights_initializer=initializers.xavier_initializer(),
                                   # biases_initializer=tf.zeros_initializer(),
                                   # name = "Conv2d_1"
                                  )
  
  bn1 = tf.layers.batch_normalization(
                                      inputs=conv1,
                                      training=train_mode,
                                      momentum=0.9,
                                      # name = "BatchNorm_1"
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
                                   # name = "Conv2d_2"
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
                                   # name = "Conv2d_3"
                                  )
  
  bn2 = tf.layers.batch_normalization(
                                      inputs=conv3,
                                      training=train_mode,
                                      momentum=0.9,
                                      # name = "BatchNorm_2"
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
                                   # name = "Conv2d_4"
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
                                   # name = "Conv2d_5"
                                  )
  
    
  bn3 = tf.layers.batch_normalization(
                                      inputs=conv5,
                                      training=train_mode,
                                      momentum=0.9,
                                      # name = "BatchNorm_3"
                                     )
                                     
  flattened = tf.contrib.layers.flatten(
                                        inputs=bn3,
                                        # name="Flatten"
                                       )

  
  dr1 = tf.contrib.layers.dropout(
                                  inputs=flattened,
                                  keep_prob=dr_keep,
                                  is_training=train_mode,
                                  # name="Dropout_1"
                                 )
  
  fc1 = tf.contrib.layers.fully_connected(
                                          inputs=dr1,
                                          num_outputs=1164,
                                          # activation_fn=tf.nn.relu,
                                          # weights_initializer=initializers.xavier_initializer(),
                                          # biases_initializer=tf.zeros_initializer(),
                                          # name="FullyConnected_1"
                                         )
                                     
  bn4 = tf.layers.batch_normalization(
                                      inputs=fc1,
                                      training=train_mode,
                                      momentum=0.9,
                                      # name="BatchNorm_4"
                                     )
      
  dr2 = tf.contrib.layers.dropout(
                                  inputs=bn4,
                                  keep_prob=dr_keep,
                                  is_training=train_mode,
                                  # name="Dropout_2"
                                 )
  
  fc2 = tf.contrib.layers.fully_connected(
                                          inputs=dr2,
                                          num_outputs=100,
                                          # activation_fn=tf.nn.relu,
                                          # weights_initializer=initializers.xavier_initializer(),
                                          # biases_initializer=tf.zeros_initializer(),
                                          # name="FullyConnected_2"
                                         )
                                     
    
  bn5 = tf.layers.batch_normalization(fc2,
                                      training=train_mode,
                                      momentum=0.9,
                                      # name="BatchNorm_5"
                                     )
      
  dr3 = tf.contrib.layers.dropout(
                                  inputs=bn5,
                                  keep_prob=dr_keep,
                                  is_training=train_mode,
                                  # name="Dropout_3"
                                 )
  
  fc3 = tf.contrib.layers.fully_connected(
                                          inputs=dr3,
                                          num_outputs=50,
                                          # activation_fn=tf.nn.relu,
                                          # weights_initializer=initializers.xavier_initializer(),
                                          # biases_initializer=tf.zeros_initializer(),
                                          # name="FullyConnected_3"
                                         )
                                     
    
  bn6 = tf.layers.batch_normalization(fc3,
                                      training=train_mode,
                                      momentum=0.9,
                                      # name="BatchNorm_6"
                                     )
  
  dr4 = tf.contrib.layers.dropout(
                                  inputs=bn6,
                                  keep_prob=dr_keep,
                                  is_training=train_mode,
                                  # name="Dropout_4"
                                 )
  
  fc4 = tf.contrib.layers.fully_connected(
                                          inputs=dr4,
                                          num_outputs=10,
                                          # activation_fn=tf.nn.relu,
                                          # weights_initializer=initializers.xavier_initializer(),
                                          # biases_initializer=tf.zeros_initializer(),
                                          # name="FullyConnected_4"
                                         )
    
  bn7 = tf.layers.batch_normalization(
                                      inputs=fc4,
                                      training=train_mode,
                                      momentum=0.9,
                                      # name="BatchNorm_7"
                                     )
      
  dr5 = tf.contrib.layers.dropout(
                                  inputs=bn7,
                                  keep_prob=dr_keep,
                                  is_training=train_mode,
                                  # name="Dropout_5"
                                 )
  
  fc5 = tf.contrib.layers.fully_connected(
                                          inputs=dr5,
                                          num_outputs=1,
                                          # activation_fn=tf.nn.relu,
                                          # weights_initializer=initializers.xavier_initializer(),
                                          # biases_initializer=tf.zeros_initializer(),
                                          # name="FullyConnected_5"
                                         )

  # Loss function for optimizer
  loss_mae = tf.losses.mean_squared_error(y, fc5)
  
  # L2 Regularization
  reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  loss_mae += l2_scale * sum(reg_losses)

  # Adam Optimizer
  optimizer = tf.contrib.optimizer_v2.AdamOptimizer(learning_rate=learning_rate)
  
  # For BatchNorm, the moving_mean and moving_variance need to be updated while training
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  
  # Optimize and group
  train_op = optimizer.minimize(loss_mae)  
  train_op = tf.group([train_op, update_ops])
  
  # For real output round and clip
  output = tf.round(fc5)
  output = tf.cast(output, tf.int32)
  output = tf.clip_by_value(output, 1, 80)
  
  # MAE for evaluating model 
  loss_mae_round = tf.losses.absolute_difference(y, output)
  
  return x, y, train_mode, train_mode_dr, train_op, loss_mae_round


def restore_model(model_name="model"):
  tf.reset_default_graph()
  x, y, train_mode, train_mode_dr, train_op, loss_mae_round = udacity_model(learning_rate= 0.01, dr_keep=0.6, l2_scale=0.0, filter_coef=2)

  log_directory = "./logs/" + model_name + "/"
  
  # Add ops to save and restore all the variables.
  saver = tf.train.Saver()

  # Later, launch the model, use the saver to restore variables from disk, and
  # do some work with the model.
  with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, log_directory + model_name + ".ckpt")
    print(model_name, "restored.")

    validation_loss = sess.run(loss_mae_round, feed_dict={x: images_validation, y: labels_validation, train_mode:False, train_mode_dr:False})
    test_loss = sess.run(loss_mae_round, feed_dict={x: images_test, y: labels_test, train_mode:False, train_mode_dr:False})
    
  return validation_loss, test_loss

def run_udacity_model(learning_rate = 0.001, epochs = 100, batch_size = 50, dr_keep=0.5, l2_scale=0.1, filter_coef=1, early_stop = 10, save_result=True):
  outfile_name = "udacity_lr" + str(learning_rate) + "_bs" + str(batch_size) + "_drk" + str(dr_keep) + "_l2" + str(l2_scale) + "_fc" + str(filter_coef)
  print("\nStarting " + outfile_name)
  log_directory = "./logs/" + outfile_name + "/"
  if not os.path.exists(log_directory):
    os.makedirs(log_directory)
  
  tf.reset_default_graph()
  x, y, train_mode, train_mode_dr, train_op, loss_mae_round = udacity_model(learning_rate, dr_keep, l2_scale, filter_coef)
  
  train_total_batch = int(len(labels_train) / batch_size)

  # setup the initialisation operator
  init_op = tf.global_variables_initializer()
  saver = None
  if save_result:
    saver = tf.train.Saver()
    
  with tf.Session() as sess:
    # initialize the variables
    sess.run(init_op)

    train_losses = []
    validation_losses = []
    
    best_validation_loss = np.inf
    best_validation_loss_epoch = 0

    early_stop_counter = 0
    
    for epoch in range(epochs):
      sum_loss = 0
      for i in range(train_total_batch):
        #print("Epoch: ", (epoch + 1), "Batch: ",
        #      i + 1, "/", train_total_batch, end="\r")

        batch_x = images_train[i*batch_size:i*batch_size+batch_size]
        batch_y = labels_train[i*batch_size:i*batch_size+batch_size]

        _, c= sess.run([train_op, loss_mae_round], feed_dict={x: batch_x, y: batch_y, train_mode:True, train_mode_dr:True})
        sum_loss += c

      train_avg_loss = sum_loss / train_total_batch

      # Validation loss after the epoch
      validation_loss = sess.run(loss_mae_round, feed_dict={x: images_validation, y: labels_validation, train_mode:True, train_mode_dr:False})

      train_losses.append(train_avg_loss)
      validation_losses.append(validation_loss)

      print("Epoch:", (epoch + 1), ", train loss:", "{:.3f}".format(train_avg_loss), ", validation loss: {:.3f}".format(validation_loss))

      if validation_loss - best_validation_loss < 0.01:
        early_stop_counter = 0
        best_validation_loss = validation_loss
        best_validation_loss_epoch = epoch + 1
        if save_result:
          save_path = saver.save(sess, log_directory + outfile_name + ".ckpt")
          print("Model saved in path: %s" % save_path)
      else:
        early_stop_counter += 1

      if early_stop_counter >= early_stop:
        print("Early stopping is trigger at step: {} loss:{}".format(epoch + 1, best_validation_loss))
        break
    if not save_result:
      test_loss = sess.run(loss_mae_round, feed_dict={x: images_test, y: labels_test, train_mode:False, train_mode_dr:False})
      print("\nTraining complete! Test loss: {:.3f}".format(test_loss))

  if save_result:
    validation_loss, test_loss = restore_model(outfile_name)

    #test_loss = sess.run(loss_mae_round, feed_dict={x: images_test, y: labels_test, train_mode:False, train_mode_dr:False})

    with open("./results/" + outfile_name + '.txt', 'w') as f:
      f.write("Epoch Train_Loss Validation_Loss\n")
      for i in range(len(train_losses)):
        f.write("{0} {1:.3f} {2:.3f}\n".format(i + 1, train_losses[i], validation_losses[i]))
      f.write("Training complete! Best validation at epoch {}. Validation loss: {:.3f}, Test loss: {:.3f}".format(best_validation_loss_epoch, validation_loss, test_loss))

    print("\nTraining complete! Best validation at epoch {}. Validation loss: {:.3f}, Test loss: {:.3f}".format(best_validation_loss_epoch, validation_loss, test_loss))

def graph_tensorboard():    
    tf.reset_default_graph()
    x, y, train_mode, train_mode_dr, train_op, loss_mae_round = udacity_model(learning_rate=0.01, dr_keep=0.6, l2_scale=0.1, filter_coef=2)

    # setup the initialisation operator
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        writer = tf.summary.FileWriter("output", sess.graph)
        print(sess.run([train_op, loss_mae_round], feed_dict={x: images_test, y: labels_test, train_mode:True}))
        writer.close()

run_udacity_model(learning_rate = 0.01, epochs = 300, batch_size = 200, dr_keep=0.6, l2_scale=0.1, filter_coef=2, early_stop=20, save_result = False)

#restore_model("udacity_lr0.01_bs200_drk0.6_l20.0_fc2")

# model_vars = tf.trainable_variables()
# tf.contrib.slim.model_analyzer.analyze_vars(model_vars, print_info=True)