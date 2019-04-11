import matplotlib.pyplot as plt
import numpy as np

handle1 = open('./elbow_data.txt')

ls = []
for eachdata in handle1:
    ls.append(eachdata)
ls_ = list(range(len(ls)))

plt.figure()
plt.plot(ls)
plt.savefig('loss_func.png')
