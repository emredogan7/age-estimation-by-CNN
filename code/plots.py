import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def plot_result(trial_name,  smooth = True):
  filename = "./results/" + trial_name + ".txt"
  with open(filename) as f:
    lines = f.readlines()
    
  train_losses = [float(line.split()[1]) for line in lines[1:-1]]
  validation_losses = [float(line.split()[2]) for line in lines[1:-1]]

  if smooth:
    train_losses = savgol_filter(train_losses, 3, 1)
    validation_losses = savgol_filter(validation_losses, 3, 1)
      


  plt.plot(train_losses, 'r')
  plt.plot(validation_losses, 'b')
  plt.show()