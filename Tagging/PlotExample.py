# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 19:06:45 2017

@author: ramon
"""
"""
how to display an image and functions using matplotlib
"""

from skimage import io
import numpy as np
import matplotlib.pyplot as plt

im = io.imread('Images/0047.jpg')
plt.figure(1)
plt.imshow(im)
plt.axis('off')

x=np.array(np.arange(-10,10,0.01))
sin=np.sin(x)    
cos=np.cos(x)    
plt.figure(2)
plt.cla()
plt.plot(x,sin,label='funcio sinius')
plt.plot(x,cos,label='funcio cosinus')
plt.xlabel('x')
plt.ylabel('resposta')
plt.legend(loc='upper right')
plt.draw()
plt.pause(0.01)
