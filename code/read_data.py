import numpy as np
import tensorflow as tf
import os

# np.random.seed(1)

path_training = "./../data/training/"
path_test = "./../data/test/"
path_validation = "./../data/validation/"

# path_training = "C:\\Users\\alper\\Workspaces\\cs559-homework\\data\\training\\"
# path_validation = "C:\\Users\\alper\\Workspaces\\cs559-homework\\data\\validation\\"
# path_test = "C:\\Users\\alper\\Workspaces\\cs559-homework\\data\\test\\"

# read train and test file names
filenames_training = os.listdir(path_training)
filenames_validation = os.listdir(path_validation)
filenames_test = os.listdir(path_test)

# shuffle file names
np.random.shuffle(filenames_training)
np.random.shuffle(filenames_validation)
np.random.shuffle(filenames_test)

# get a set of filenames for speeding up
filenames_training = filenames_training#[:1000]
filenames_validation = filenames_validation#[:400]
filenames_test = filenames_test#[:300]

# get paths of file names
filepaths_training = [path_training + f for f in filenames_training]
filepaths_validation = [path_validation + f for f in filenames_validation]
filepaths_test = [path_test + f for f in filenames_test]

# get labels of file names
labels_training = np.array([[int(x[:3])] for x in filenames_training])
labels_validation = np.array([[int(x[:3])] for x in filenames_validation])
labels_test = np.array([[int(x[:3])] for x in filenames_test])

np.save('labels_training.npy', labels_training)
np.save('labels_validation.npy', labels_validation)
np.save('labels_test.npy', labels_test)  

# sess = tf.InteractiveSession()  

with tf.Session() as sess:

    images_train = np.array([tf.image.decode_jpeg(tf.read_file(
        x)).eval().reshape(-1) for x in filepaths_training]) / 255
    images_validation = np.array([tf.image.decode_jpeg(tf.read_file(
        x)).eval().reshape(-1) for x in filepaths_validation]) / 255
    images_test = np.array([tf.image.decode_jpeg(tf.read_file(
        x)).eval().reshape(-1) for x in filepaths_test]) / 255

    np.save('images_train.npy', images_train)
    np.save('images_validation.npy', images_validation)
    np.save('images_test.npy', images_test)  
