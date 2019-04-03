import numpy as np
import tensorflow as tf
import os

# np.random.seed(1)

path_train = "./../data/train/"
path_test = "./../data/test/"
path_validation = "./../data/validation/"

# path_train = "C:\\Users\\alper\\Workspaces\\cs559-homework\\data\\train\\"
# path_validation = "C:\\Users\\alper\\Workspaces\\cs559-homework\\data\\validation\\"
# path_test = "C:\\Users\\alper\\Workspaces\\cs559-homework\\data\\test\\"

# read train and test file names
filenames_train = os.listdir(path_train)
filenames_validation = os.listdir(path_validation)
filenames_test = os.listdir(path_test)

# shuffle file names
np.random.shuffle(filenames_train)
np.random.shuffle(filenames_validation)
np.random.shuffle(filenames_test)

# get a set of filenames for speeding up
filenames_train = filenames_train#[:1000]
filenames_validation = filenames_validation#[:400]
filenames_test = filenames_test#[:300]

# get paths of file names
filepaths_train = [path_train + f for f in filenames_train]
filepaths_validation = [path_validation + f for f in filenames_validation]
filepaths_test = [path_test + f for f in filenames_test]

# get labels of file names
labels_train = np.array([[int(x[:3])] for x in filenames_train])
labels_validation = np.array([[int(x[:3])] for x in filenames_validation])
labels_test = np.array([[int(x[:3])] for x in filenames_test])

np.save('labels_train.npy', labels_train)
np.save('labels_validation.npy', labels_validation)
np.save('labels_test.npy', labels_test)  

# sess = tf.InteractiveSession()  

with tf.Session() as sess:

    images_train = np.array([tf.image.decode_jpeg(tf.read_file(
        x)).eval().reshape(-1) for x in filepaths_train]) / 255
    images_validation = np.array([tf.image.decode_jpeg(tf.read_file(
        x)).eval().reshape(-1) for x in filepaths_validation]) / 255
    images_test = np.array([tf.image.decode_jpeg(tf.read_file(
        x)).eval().reshape(-1) for x in filepaths_test]) / 255

    np.save('images_train.npy', images_train)
    np.save('images_validation.npy', images_validation)
    np.save('images_test.npy', images_test)  




from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(images_train, labels_train)

unison_array = [ np.append(X_resampled[i], y_resampled[i]) for i in range(len(y_resampled))]

np.random.shuffle(unison_array)

X_resampled = np.array([x[:-1] for x in unison_array])
y_resampled = np.array([[x[-1]] for x in unison_array])
np.save('images_train_oversampled.npy', X_resampled)
np.save('labels_train_oversampled.npy', y_resampled)
