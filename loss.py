import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import pylab
from pylab import figure, show, legend
from mpl_toolkits.axes_grid1 import host_subplot

# read the log file
# fp = open('log.txt', 'r')

# train_iterations = [i for i in range(56250)]
train_iterations = [i+1 for i in range(12)]
train_loss = [0.78,0.39,0.28,0.22,0.16,0.151,0.141,0.143,0.142,0.132,0.140,0.137]
# train_loss = [i for i in range(52500) if i/7500]
test_iterations = [i+1 for i in range(12)]
test_accuracy = [0.13,0.69,0.78,0.83,0.86,0.885,0.89,0.901,0.91,0.913,0.915,0.916]
print(train_iterations,test_accuracy)

host = host_subplot(111)
plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
par1 = host.twinx()
# set labels
host.set_xlabel("iterations")
host.set_ylabel("loss")
par1.set_ylabel("validation accuracy")

# plot curves
p1, = host.plot(train_iterations, train_loss, label="training loss")
p2, = par1.plot(test_iterations, test_accuracy, label="validation accuracy")

# set location of the legend,
# 1->rightup corner, 2->leftup corner, 3->leftdown corner
# 4->rightdown corner, 5->rightmid ...
host.legend(loc=5)

# set label color
host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())
# set the range of x axis of host and y axis of par1
host.set_xlim([0, 13])
par1.set_ylim([0., 1.05])
plt.draw()
plt.show()
