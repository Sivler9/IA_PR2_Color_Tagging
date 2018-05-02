#!/usr/bin/python
# -*- coding: utf-8 -*-
"""@author: ramon, bojana"""

import sys

reload(sys)
sys.setdefaultencoding('utf8')

import re
import numpy as np
import ColorNaming as cn
from skimage import color
import KMeans as km


def NIUs():  # Faltan NIUs
    return 1325996, 142, 142


def loadGT(fileName):
    """Loads the file with groundtruth content

    :param str fileName: name of the file with groundtruth

    :rtype: list[(str, numpy.ndarray)]
    :return: list of tuples of ground truth data. (Name, [list-of-labels])
    """
    groundTruth = []
    fd = open(fileName, 'r')
    for line in fd:
        splitLine = line.split(' ')[:-1]
        labels = [''.join(sorted(filter(None, re.split('([A-Z][^A-Z]*)', l)))) for l in splitLine[1:]]
        groundTruth.append((splitLine[0], labels))

    return groundTruth


def evaluate(description, GT, options):
    """EVALUATION FUNCTION

    :param list description: lists of color name, contain one list of color labels for every images tested
    :param list GT: images to test and the real color names (see :func:`loadGT`)
    :param dict options: contains options to control metric, ...

    :rtype: (float, list[float])
    :return:
        mean_score: is the mean of the scores of each image\n
        scores: contain the similiraty between the ground truth list of color names and the obtained
    """
    #########################################################
    # YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    # AND CHANGE FOR YOUR OWN CODE TODO
    #########################################################
    scores = np.random.rand(len(description), 1)
    return sum(scores)/len(description), scores


def similarityMetric(Est, GT, options):
    """SIMILARITY METRIC

    :param list Est: list of color names estimated from the image ['red','green',..]
    :param list GT: list of color names from the ground truth
    :param dict options: contains options to control metric, ...

    :rtype: float
    :return: S similarity between label LISTs
    """
    if options is None:
        options = {}
    if 'metric' not in options:
        options['metric'] = 'basic'

    options['metric'] = options['metric'].lower()
    #########################################################
    # YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    # AND CHANGE FOR YOUR OWN CODE TODO
    #########################################################
    if options['metric'] == 'basic':
        import random
        return random.uniform(0, 1)
    else:
        return 0


def getLabels(kmeans, options):
    """Labels all centroids of kmeans object to their color names

    :param KMeans.KMeans kmeans: object of the class KMeans
    :param dict options: options necessary for labeling

    :rtype: (list[str], list[int])
    :returns:
        colors: labels of centroids of kmeans object\n
        ind: indexes of centroids with the same color label
    """
    #########################################################
    # YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    # AND CHANGE FOR YOUR OWN CODE TODO
    #########################################################
    # remind to create composed labels if the probability of
    # the best color label is less than  options['single_thr']
    meaningful_colors = ['color' + str(i) for i in xrange(kmeans.K)]
    unique = xrange(kmeans.K)
    return meaningful_colors, unique


def processImage(im, options):
    """Finds the colors present on the input image

    :param numpy.ndarray im: input image
    :param dict options: dictionary with options

    :rtype: (list[str], list[int], KMeans.KMeans)
    :returns:
        colors: name of colors of centroids of kmeans object\n
        indexes: indexes of centroids with the same label\n
        kmeans: object of the class KMeans
    """
    #########################################################
    # YOU MUST ADAPT THE CODE IN THIS FUNCTIONS TO: TODO
    #########################################################

    # 1- CHANGE THE IMAGE TO THE CORRESPONDING COLOR SPACE FOR KMEANS
    options['colorspace'] = options['colorspace'].lower()

    if options['colorspace'] == 'colornaming':
        pass
    elif options['colorspace'] == 'rgb':
        pass
    elif options['colorspace'] == 'lab':
        pass

    # 2- APPLY KMEANS ACCORDING TO 'OPTIONS' PARAMETER
    kmeans = km.KMeans(im, options['K'], options)
    kmeans.run()

    # 3- GET THE NAME LABELS DETECTED ON THE 11 DIMENSIONAL SPACE
    if options['colorspace'] == 'rgb':
        pass

    #########################################################
    # THE FOLLOWING 2 END LINES SHOULD BE KEPT UNMODIFIED
    #########################################################
    colors, which = getLabels(kmeans, options)
    return colors, which, kmeans
