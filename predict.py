import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import pylab
from pylab import figure, show, legend
from mpl_toolkits.axes_grid1 import host_subplot
from matplotlib import font_manager
# 设置展示图表的内容字体，可以从Windows字体里面拷贝字体文件到当前文件目录
my_font=font_manager.FontProperties(fname="C:\Windows\Fonts\msyh.ttc")

# train_iterations = [i for i in range(56250)]
wcl_iterations = [i+1 for i in range(30)]
wcl_rate = [0.8738,0.8681,0.8247,0.8269,0.9001,0.8917,0.878,0.8713,0.9655,0.9473,0.9602,0.9556,0.92,0.9628,0.8973,0.8485,0.8034,0.8147,0.8211,0.8983,0.8894,0.9233,0.9421,0.8956,0.9354,0.9102,0.9409,0.9311,0.8876,0.9514]
# train_loss = [i for i in range(52500) if i/7500]
yz_iterations = [i+1 for i in range(30)]
yz_accuracy = [0.764,0.7612,0.7164,0.7445,0.8573,0.8341,0.804,0.7962,0.8256,0.7577,0.8827,0.8661,0.8132,0.902,0.8112,0.7025,0.7067,0.7297,0.7247,0.7131,0.832,0.845,0.87,0.861,0.8204,0.8608,0.8902,0.8716,0.8322,0.8591]
print(wcl_rate,yz_accuracy)

# squares=[1, 4, 9, 16, 25]
# x=[1, 2, 3, 4, 5]
# 设置线宽
plt.plot(wcl_iterations, wcl_rate, color="red",linestyle='--',label='不超率',marker='o')
plt.plot(yz_iterations, yz_accuracy, color="blue",linestyle='-',label='一致率',marker='*')
# 设置图表标题，并给坐标轴添加标签
plt.title("模型线上表现", fontproperties=my_font)
plt.xlabel("Day")
plt.ylabel("Rate")

# 设置坐标轴刻度标记的大小
# plt.tick_params(axis='both',labelsize=30)
# 设置图例样式
plt.legend(loc='upper left', prop=my_font)  # loc代表图例所在的位置，upper right代表右上角
plt.show()
