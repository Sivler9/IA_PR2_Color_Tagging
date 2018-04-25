# -*- coding: utf-8 -*-
"""

@author: ramon
"""
from skimage import io
import matplotlib.pyplot as plt


import Labels as lb


plt.close("all")
if __name__ == "__main__":

    #'colorspace': 'RGB', 'Lab' o 'ColorNaming'
    options = {'colorspace':'RGB', 'K':6, 'synonyms':False, 'single_thr':0.6, 'verbose':False, 'km_init':'first', 'metric':'basic'}

    ImageFolder = 'Images'
    GTFile = 'LABELSsmall.txt'
    
    GTFile = ImageFolder + '/' + GTFile
    GT = lb.loadGT(GTFile)

    DBcolors = []
    for gt in GT:
        print gt[0]
        im = io.imread(ImageFolder+"/"+gt[0])    
        colors,_,_ = lb.processImage(im, options)
        DBcolors.append(colors)
        
    encert,_ = lb.evaluate(DBcolors, GT, options)
    print "Encert promig: "+ '%.2f' % (encert*100) + '%'
