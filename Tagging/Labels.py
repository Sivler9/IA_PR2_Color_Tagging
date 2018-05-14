# -*- coding: utf-8 -*-
"""

@author: ramon, bojana
"""
import re
import numpy as np
import ColorNaming as cn
from skimage import color
import KMeans as km

def NIUs():
    """@brief   Returns Authors NIUs

    @param  NONE

    @return NIU LIST    list of NIUs
    """

    return [1325996,1396552,1424504]


def loadGT(fileName):
    """@brief   Loads the file with groundtruth content
    
    @param  fileName  STRING    name of the file with groundtruth
    
    @return groundTruth LIST    list of tuples of ground truth data
                                (Name, [list-of-labels])
    """

    groundTruth = []
    fd = open(fileName, 'r')
    for line in fd:
        splitLine = line.split(' ')[:-1]
        labels = [''.join(sorted(filter(None,re.split('([A-Z][^A-Z]*)',l)))) for l in splitLine[1:]]
        groundTruth.append( (splitLine[0], labels) )
        
    return groundTruth


def evaluate(description, GT, options):
    """@brief   EVALUATION FUNCTION
    @param description LIST of color name lists: contain one lsit of color labels for every images tested
    @param GT LIST images to test and the real color names (see  loadGT)
    @options DICT  contains options to control metric, ...
    @return mean_score,scores mean_score FLOAT is the mean of the scores of each image
                              scores     LIST contain the similiraty between the ground truth list of color names and the obtained
    """
#########################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#########################################################
    scores = np.random.rand(len(description),1)        
    return sum(scores)/len(description), scores



def similarityMetric(Est, GT, options):
    """@brief   SIMILARITY METRIC
    @param Est LIST  list of color names estimated from the image ['red','green',..]
    @param GT LIST list of color names from the ground truth
    @param options DICT  contains options to control metric, ...
    @return S float similarity between label LISTs
    """
    
    if options == None:
        options = {}
    if not 'metric' in options:
        options['metric'] = 'basic'

    if options['metric'].lower() == 'basic'.lower():
        return float(sum(el in Est for el in GT))/float(len(Est))
    else:
        return 0
        
def getLabels(kmeans, options):
    """@brief   Labels all centroids of kmeans object to their color names
    
    @param  kmeans  KMeans      object of the class KMeans
    @param  options DICTIONARY  options necessary for labeling
    
    @return colors  LIST    colors labels of centroids of kmeans object
    @return ind     LIST    indexes of centroids with the same color label
    """
    sumCluster = [np.sum(kmeans.clusters==k) for k in range(kmeans.K)]
    sortClusterIndex = np.argsort(sumCluster)
    kmeans.centroids = kmeans.centroids[sortClusterIndex[::-1]] #Ordena de mas frecuente a menos frecuente
    colors = []
    ind = []
    for i in range(kmeans.K):
        indexes = np.argsort(kmeans.centroids[i])
        if kmeans.centroids[i][indexes[-1]] >= options['single_thr']:
            newColor = unicode(cn.colors[indexes[-1]])
        else:
            compoundColor = sorted([cn.colors[indexes[-1]],cn.colors[indexes[-2]]])
            newColor = unicode(compoundColor[0] + compoundColor[1])
        if newColor in colors:
            ind[colors.index(newColor)].append(i)
        else:
            colors.append(newColor)
            ind.append([i])

    return colors, ind


def processImage(im, options):
    """@brief   Finds the colors present on the input image
    
    @param  im      LIST    input image
    @param  options DICTIONARY  dictionary with options
    
    @return colors  LIST    colors of centroids of kmeans object
    @return indexes LIST    indexes of centroids with the same label
    @return kmeans  KMeans  object of the class KMeans
    """

#########################################################
##  YOU MUST ADAPT THE CODE IN THIS FUNCTIONS TO:
##  1- CHANGE THE IMAGE TO THE CORRESPONDING COLOR SPACE FOR KMEANS
##  2- APPLY KMEANS ACCORDING TO 'OPTIONS' PARAMETER
##  3- GET THE NAME LABELS DETECTED ON THE 11 DIMENSIONAL SPACE
#########################################################

    imP = np.copy(im).astype(np.float64)
##  1- CHANGE THE IMAGE TO THE CORRESPONDING COLOR SPACE FOR KMEANS
    if options['colorspace'].lower() == 'ColorNaming'.lower():  
        imP = cn.ImColorNamingTSELabDescriptor(imP)
    elif options['colorspace'].lower() == 'RGB'.lower():        
        pass #por defecto la imagen esta en RGB
    elif options['colorspace'].lower() == 'Lab'.lower():        
        imP = color.rgb2lab(imP/255.)
    elif options['colorspace'].lower() == 'HSV'.lower():
        #imP = color.rgb2hsv(imP.astype(np.uint8)/255.)
        imP = color.rgb2hsv(imP/255.)

##  2- APPLY KMEANS ACCORDING TO 'OPTIONS' PARAMETER
    if options['K']<2: # find the bes K
        kmeans = km.KMeans(imP, 0, options)
        kmeans.bestK()
    else:
        kmeans = km.KMeans(imP, options['K'], options)
        kmeans.run()

##  3- GET THE NAME LABELS DETECTED ON THE 11 DIMENSIONAL SPACE
    if options['colorspace'].lower() == 'Lab'.lower():
        tmpcentroids = color.lab2rgb(np.reshape(kmeans.centroids,(1,kmeans.K,3)))
        kmeans.centroids = tmpcentroids.reshape(kmeans.K,tmpcentroids.shape[2])*255
    elif options['colorspace'].lower() == 'HSV'.lower():
        '''tmpcentroids = color.hsv2rgb(np.reshape(kmeans.centroids,(1,kmeans.K,3)).astype(np.uint8))
        kmeans.centroids = tmpcentroids.reshape(kmeans.K,tmpcentroids.shape[2]).astype(np.float64)*255'''
        tmpcentroids = color.hsv2rgb(np.reshape(kmeans.centroids,(1,kmeans.K,3)))
        kmeans.centroids = tmpcentroids.reshape(kmeans.K,tmpcentroids.shape[2])*255

    if options['colorspace'].lower() != 'ColorNaming'.lower():
        tmpcentroids = cn.ImColorNamingTSELabDescriptor(np.reshape(kmeans.centroids,(1,kmeans.K,3)))
        #a,b,c,d = cn.ImColorNamingTSELab(np.reshape(kmeans.centroids,(1,kmeans.K,3)))
        kmeans.centroids = tmpcentroids.reshape(kmeans.K,tmpcentroids.shape[2])


#########################################################
##  THE FOLLOWING 2 END LINES SHOULD BE KEPT UNMODIFIED
#########################################################
    colors, which = getLabels(kmeans, options)   
    return colors, which, kmeans
