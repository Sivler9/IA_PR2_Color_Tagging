#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Functions:
 - :func:`NIUs()<NIUs>`
 - :func:`loadGT(fileName)<loadGT>`
 - :func:`evaluate(description, GT, options)<evaluate>`
 - :func:`similarityMetric(Est, GT, options)<similarityMetric>`
 - :func:`getLabels(kmeans, options)<getLabels>`
 - :func:`processImage(im, options)<processImage>`
"""

import sys

reload(sys)
sys.setdefaultencoding('utf8')

import re
import numpy as np
import ColorNaming as cn
from skimage import color
from sklearn.metrics import jaccard_similarity_score  # No funcionan, GT tiene que tener mismo len() que Est
import KMeans as km


def NIUs():
    return 1325996, 1396552, 1424504


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
    scores = [similarityMetric(processImage(GT[i], options)[0], description[i], options) for i in xrange(len(GT))]
    return sum(scores)/len(description), scores


metrics = {'basic': lambda estimate, truth: sum(im in estimate for im in truth)/float(len(estimate)),
           '': None}  # TODO - Extras


def similarityMetric(Est, GT, options):
    """SIMILARITY METRIC

    :param list Est: list of color names estimated from the image ['red','green',..]
    :param list GT: list of color names from the ground truth
    :param dict options: contains options to control metric, ...

    :rtype: float
    :return: S similarity between label LISTs
    """
    if options is None:
        options = {'metric': 'basic'}

    if 'metric' not in options:
        options['metric'] = 'basic'

    options['metric'] = options['metric'].lower()

    if options['metric'] in metrics:
        return metrics[options['metric']](Est, GT)
    else:
        return np.random.rand()


# a, i = ("A", "D50", "D55", "D65", "D75", "E")[3], ("2", "10")[0]  # rgb2lab(im, a, i)
# rgb2gray no es reversible
space_change = {'rgb': lambda im:   im,
                'hsv':              color.rgb2hsv,  # No cartesianas
                'rgb cie':          color.rgb2rgbcie,
                'yiq':              color.rgb2yiq,
                'yuv':              color.rgb2yuv,
                'ycbcr':            color.rgb2ycbcr,
                'ypbpr':            color.rgb2ypbpr,
                'xyz':              color.rgb2xyz,
                'lab':              color.rgb2lab,
                'lch': lambda im:   color.lab2lch(color.rgb2lab(im)),  # No cartesianas
                'luv':              color.rgb2luv,
                'hed':              color.rgb2hed,
                'colornaming':      cn.ImColorNamingTSELabDescriptor}

space_return = {'rgb': lambda im:   im,
                'hsv':              color.hsv2rgb,  # No cartesianas
                'rgb cie':          color.rgbcie2rgb,
                'yiq':              color.yiq2rgb,
                'yuv':              color.yuv2rgb,
                'ycbcr':            color.ycbcr2rgb,
                'ypbpr':            color.ypbpr2rgb,
                'xyz':              color.xyz2rgb,
                'lab':              color.lab2rgb,
                'lch': lambda im:   color.lab2rgb(color.lch2lab(im)),  # No cartesianas
                'luv':              color.luv2rgb,
                'hed':              color.hed2rgb}

ucolor = [unicode(n) for n in cn.colors]


def getLabels(kmeans, options):
    """Labels all centroids of kmeans object to their color names

    :param KMeans.KMeans kmeans: object of the class KMeans
    :param dict options: options necessary for labeling

    :rtype: (list[str], list[int])
    :returns:
        colors: labels of centroids of kmeans object\n
        ind: indexes of centroids with the same color label
    """
    temp = []

    centers = kmeans.centroids.reshape(1, -1, kmeans.centroids.shape[-1])
    if options['colorspace'] in space_return:
        centers = space_return[options['colorspace']](centers)
        centers = cn.ImColorNamingTSELabDescriptor(centers)

    for c in centers[0]:
        # a = [index_sub_max, index_max]
        a = np.argpartition(c, -2)[-2:]
        l = ucolor[a[1]]
        if c[a[1]] < options['single_thr']:
            l = (l + ucolor[a[0]]) if ucolor[a[0]] > l else (ucolor[a[0]] + l)
        temp.append(l)

    meaningful_colors = np.unique(temp).tolist()
    unique = [np.where(np.array(temp) == c)[0].tolist() for c in meaningful_colors]

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

    if not isinstance(im, np.ndarray):  # EvaluationTest() envia informacion incorrecta
        return im[1], None, None

    # 1- CHANGE THE IMAGE TO THE CORRESPONDING COLOR SPACE FOR KMEANS
    options['colorspace'] = options['colorspace'].lower()

    if options['colorspace'] in space_change:
        im = space_change[options['colorspace']](im.astype(np.uint8))  # convertir a enteros [0, 255]
    else:
        options['colorspace'] = 'rgb'
        print("'colorspace' unspecified, using 'rgb'")

    # 2- APPLY KMEANS ACCORDING TO 'OPTIONS' PARAMETER
    kmeans = km.KMeans(im, options['K'], options)

    if options['K'] < 2:
        options['K'] = kmeans.bestK()

    kmeans.run()

    # 3- GET THE NAME LABELS DETECTED ON THE 11 DIMENSIONAL SPACE
    colors, which = getLabels(kmeans, options)

    return colors, which, kmeans
