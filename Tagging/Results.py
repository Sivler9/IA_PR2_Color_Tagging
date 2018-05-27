#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Generador de resultados de Kmeans() y Labels()
"""

import sys

reload(sys)
sys.setdefaultencoding('utf8')

import time

import numpy as np
from skimage import io

import Labels as lb
from KMeans import KMeans

if __name__ == '__main__':
    options = {'colorspace': 'lab', 'K': 15, 'single_thr': 0.6, 'verbose': False, 'km_init': 'kmeans++',
               'fitting': 'calinski_harabaz', 'metric': 'basic'}  # , 'tolerance': 1, 'max_iter': 100}

    ImageFolder = 'Images'
    GTFile = 'LABELSlarge.txt'

    GTFile = ImageFolder + '/' + GTFile
    GT = lb.loadGT(GTFile)

    with open("res.txt", "w") as fil:
        for gt in GT[0:190:10]:
            print gt[0]
            im = io.imread(ImageFolder + "/" + gt[0])

            im2 = lb.space_change[options['colorspace']](im.astype(np.uint8))  # convertir a enteros [0, 255]
            kmeans = KMeans(im2, options['K'], options)

            for ini in KMeans.fit:
                options['fitting'] = ini
                kmeans._init_options(options)
                bes = kmeans.bestK()

                options['fitting'] = 'calinski_harabaz'
                fit = str(kmeans.fitting())

                if options['colorspace'] in lb.space_return:
                    c = lb.space_return[options['colorspace']](kmeans.centroids.reshape(1, -1, kmeans.centroids.shape[-1]))
                    kmeans.centroids = lb.cn.ImColorNamingTSELabDescriptor(c if options['colorspace'] == 'rgb' else c*255.0)[0]

                for t in xrange(55, 86, 5):
                    t = float(t)/100.0
                    options['single_thr'] = t
                    fil.write(gt[0][1:-4] + ', ' + ini + ', ' + str(t) + ', ' + str(bes) + ', ' + fit + ', ' + lb.similarityMetric(lb.getLabels(kmeans, options)[0], gt[1], {'metric': 'mat'}).__repr__()[1:-1] + '\n')
