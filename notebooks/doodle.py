
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import os
import tensorflow as tf
# :D


# In[11]:


file_names_training = os.listdir("./../data/training/")
file_names_test = os.listdir("./../data/test/")
file_names_validation = os.listdir("./../data/validation/")


# In[12]:


# age_training = [x[:3] for x in file_names_training]

dict_training = {x:int(x[:3]) for x in file_names_training}
dict_validation = {x:int(x[:3]) for x in file_names_validation}
dict_test = {x:int(x[:3]) for x in file_names_test}

labes_training = [int(x[:3]) for x in file_names_training]
# other 2 to be adeed.


# In[13]:


image_count_training = len(file_names_training)


# In[17]:


img_path = file_names_training[2]
img_raw = tf.read_file(img_path)
print(repr(img_raw)[:])

