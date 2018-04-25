#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Contains:
 Class :class:`KMeans`\n
 Function :func:`distance`
"""

import sys

reload(sys)
sys.setdefaultencoding('utf8')

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import ColorNaming
import mpl_toolkits.mplot3d.axes3d as axes3d


def distance(X, C):
    """Calculates the distance between each pixcel and each centroid.

    :param numpy.ndarray X: PxD 1st set of data points (usually data points).
    :param numpy.ndarray C: KxD 2nd set of data points (usually cluster centroids points).

    :rtype: numpy.ndarray
    :return: PxK position ij is the distance between the i-th point of X and the j-th point of C.
    """
    def dist(f, t):
        assert f.size == t.size
        return np.sqrt(np.sum(np.square(f - t)))

    #########################################################
    # # YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    # # AND CHANGE FOR YOUR OWN CODE
    #########################################################
    PK = np.ndarray((X.shape[0], C.shape[0]))
    for i, x in enumerate(X):
        for j, c in enumerate(C):
            PK[i][j] = dist(x, c)
    return PK


class KMeans:
    def __init__(self, X, K, options=None):
        """Constructor of KMeans class

        :param numpy.ndarray X: input data
        :param int K: number of centroids
        :param dict or None options: dctionary with options
        """
        self.pca = None
        self._init_options(options if options else {})     # DICT options
        self._init_X(X)                                    # LIST data coordinates
        self._init_rest(K)                                 # Initializes de rest of the object

    def _init_X(self, X):
        """Initialization of all pixels.
        Sets X an as an array of data in vector form (PxD where P=N*M and D=3 in the above example).

        :param numpy.ndarray X: list of all pixel values. Usually a numpy array containing an image NxMx3.
        """
        self.X = X.reshape(-1, 3)

        # TODO color_space
        if self.options['color_space'] == 'rgb':
            pass
        elif self.options['color_space'] == 'color_naming':
            for i in xrange(self.X.shape[0]):
                self.X[i] = ColorNaming.RGB2Lab(self.X[i])  # TODO Usar la funcio que de las probabilidades
        else:
            print("'color_space' unspecified, using 'rgb'")

    def _init_options(self, options):
        """Initialization of options in case some fields are left undefined

        :param dict options: dictionary with options sets de options parameters
        """
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'fisher'
        if 'color_space' not in options:
            options['color_space'] = 'rgb'

        options['km_init'] = options['km_init'].lower()
        options['fitting'] = options['fitting'].lower()
        options['color_space'] = options['color_space'].lower()

        self.options = options

    def _init_rest(self, K):
        """Initialization of the remainig data in the class.

        :param int K: number of centroids
        """
        self.K = K                                              # INT number of clusters
        if self.K > 0:
            self._init_centroids()                              # LIST centroids coordinates
            self.old_centroids = np.empty_like(self.centroids)  # LIST coordinates of centroids from previous iteration
            self.clusters = np.zeros(len(self.X))           # LIST list that assignes each element of X into a cluster
            self._cluster_points()                              # sets the first cluster assignation
        self.num_iter = 0                                       # INT current iteration
        #############################################################
        # # THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):
        """Initialization of centroids depends on self.options['km_init']"""
        def rgb_to_line(n):
            assert n <= self.K
            n = format(n*(256**3)//(self.K + 1), '06x')
            return int(n[:2], 16), int(n[2:4], 16), int(n[4:], 16)

        if self.X.shape[0] < self.K:
            print("La imagen tiene menos de " + str(self.K) + " pixeles, usando K igual a " + str(self.X.shape[0]))
            self.K = self.X.shape[0]

        c = []
        if self.options['km_init'] == 'first':
            for k in self.X:
                k = k.tolist()
                if k not in c:
                    c.append(k)
                    if len(c) == self.K:
                        break
        elif self.options['km_init'] == 'random':
            i = self.K + int(np.sqrt(self.X.shape[0]))
            while i:
                i -= 1
                k = np.random.choice(self.X).tolist()
                if k not in c:
                    c.append(k)
                    if len(c) == self.K:
                        break
        elif self.options['km_init'] == 'uniform':
            c = [rgb_to_line(k + 1) for k in xrange(self.K)]
        else:  # TODO - Opciones extra. ej. puntos con distancia maxima en el espacio, separados uniformemente ...
            print("'km_init' unspecified, using 'really_random'")
            c = np.random.rand(self.K, self.X.shape[1])*255  # RGB

        if len(c) < self.K:
            print("La imagen tiene menos de " + str(self.K) + " colores, se han encontrado " + str(len(c)))
            self.K = len(c)

        self.centroids = np.array(c)

    def _cluster_points(self):
        """Calculates the closest centroid of all points in X"""
        PK = distance(self.X, self.centroids)
        self.clusters = np.array([np.where(p == max(p))[0][0] for p in PK])

    def _get_centroids(self):
        """Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid"""
        s = np.zeros(self.centroids.shape[0])
        c = np.zeros((self.centroids.shape[0], self.X.shape[1]))

        for i, x in enumerate(self.X):
            s[self.clusters[i]] += 1
            c[self.clusters[i]] += x

        self.old_centroids = self.centroids
        self.centroids = np.array([n/s[i] if s[i] else n for i, n in enumerate(c)])

    def _converges(self):
        """Checks if there is a difference between current and old centroids"""
        return any(n > self.options['tolerance'] for n in np.abs(self.centroids - self.old_centroids).reshape(-1))

    def _iterate(self, show_first_time=True):
        """One iteration of K-Means algorithm. This method should reassigne all the points from X
        to their closest centroids and based on that, calculate the new position of centroids."""
        self.num_iter += 1
        self._cluster_points()
        self._get_centroids()
        if self.options['verbose']:
            self.plot(show_first_time)

    def run(self):
        """Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations."""
        if not self.K:
            self.bestK()
            return

        self._iterate(True)
        self.options['max_iter'] = np.inf
        if self.options['max_iter'] > self.num_iter:
            while not self._converges():
                self._iterate(False)

    def bestK(self):
        """Runs K-Means multiple times to find the best K for the current data given the 'fitting' method.
        In case of Fisher elbow method is recommended.

        At the end, self.centroids and self.clusters contains the information for the best K.
        NO need to rerun KMeans.

        :return: the best K found.
        :rtype: int"""
        #######################################################
        # # YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        # # AND CHANGE FOR YOUR OWN CODE TODO
        #######################################################
        self._init_rest(4)
        self.run()
        fit = self.fitting()
        return 4

    def fitting(self):
        """:return: a value describing how well the current kmeans fits the data\n:rtype: float"""
        #######################################################
        # # YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        # # AND CHANGE FOR YOUR OWN CODE TODO
        #######################################################
        if self.options['fitting'] == 'fisher':
            return np.random.rand(1)
        else:
            return np.random.rand(1)

    def plot(self, first_time=True):
        """Plots the results"""
        # markers_shape = 'ov^<>1234sp*hH+xDd'
        markers_color = 'bgrcmybgrcmybgrcmyk'
        if first_time:
            plt.gcf().add_subplot(111, projection='3d')
            plt.ion()
            plt.show()

        if self.X.shape[1] > 3:
            if not self.pca:
                self.pca = PCA(n_components=3)
                self.pca.fit(self.X)
            Xt = self.pca.transform(self.X)
            Ct = self.pca.transform(self.centroids)
        else:
            Xt = self.X
            Ct = self.centroids

        for k in range(self.K):
            plt.gca().plot(Xt[self.clusters == k, 0], Xt[self.clusters == k, 1], Xt[self.clusters == k, 2],
                '.' + markers_color[k])
            plt.gca().plot(Ct[k, 0:1], Ct[k, 1:2], Ct[k, 2:3], 'o'+'k', markersize=12)

        if first_time:
            plt.xlabel('dim 1')
            plt.ylabel('dim 2')
            plt.gca().set_zlabel('dim 3')
        plt.draw()
        plt.pause(0.01)
