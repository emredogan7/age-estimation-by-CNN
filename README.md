# Age Estimation by CNN Based Regression Model

- _Completed by: Hamdi Alperen Çetin & Emre Doğan_

- This is the README file of CS559 Deep Learning Course Homework, Bilkent University 2019.  
- We propose a Convolutional Neural Network Model to succesfully estimate the age of a person given her/his cropped face image. Different from the classical CNN models, our model ends up with a regression layer, not a classifier one. So, backpropagation process is done based on the regression output.  



## Dataset:
- We trained our model with a downsampled version of [UTKFace Dataset](http://aicip.eecs.utk.edu/wiki/UTKFace). 
- Due to its large size, we cannot share original dataset(all training + validation + test data). But you can find some samples of our dataset from [here](./data-samples/). Notice that the first 3 letters of any image corresponds to its output layer (age of the person in the image).


## Model:



Our project consists of source code, model log, results and our project report.

- results folder includes 
	* the training and validation losses for each epoch for our several trials.
	* best validation loss and regarding test loss in the  'LastVersion.txt' file.

- data folder includes .npy versions of our whole dataset. These files are created by read_data.py script in our local machine. 
  You can access read_data.py from our main directory.

- CS559_HW.ipynb Jupyter Notebook file includes our whole project work.

- CS559_HW.html file includes a final version of our Jupyter notebook if you want to access it easily. 
