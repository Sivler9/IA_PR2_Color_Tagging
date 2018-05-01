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

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import mpl_toolkits.mplot3d.axes3d as axes3d
from sklearn.metrics import euclidean_distances


def distance(X, C):
    """Calculates the distance between each pixcel and each centroid.

    :param numpy.ndarray X: PxD 1st set of data points (usually data points).
    :param numpy.ndarray C: KxD 2nd set of data points (usually cluster centroids points).

    :rtype: numpy.ndarray
    :return: PxK position ij is the distance between the i-th point of X and the j-th point of C.
    """
    return euclidean_distances(X, C)


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
            Color-space position must be at deepest array.
        """
        self.X = X.reshape(-1, X.shape[-1])

        # TODO colorspace
        if self.options['colorspace'] == 'rgb':
            pass
            '''elif self.options['colorspace'] == 'lab':
            for i in xrange(self.X.shape[0]):
                self.X[i] = ColorNaming.RGB2Lab(self.X[i])'''  # TODO Usar la funcio que de las probabilidades
        else:
            print("'colorspace' unspecified, using 'rgb'")

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
        if 'colorspace' not in options:
            options['colorspace'] = 'rgb'

        options['km_init'] = options['km_init'].lower()
        options['fitting'] = options['fitting'].lower()
        options['colorspace'] = options['colorspace'].lower()

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
            # lento -> c = np.unique(self.X, axis=1)[:self.K]
            for k in self.X:
                k = k.tolist()
                if k not in c:
                    c.append(k)
                    if len(c) == self.K:
                        break
        elif self.options['km_init'] == 'random':
            for k in np.random.randint(0, self.X.shape[0], size=self.K + int(np.sqrt(self.X.shape[0]))):
                k = self.X[k].tolist()
                if k not in c:
                    c.append(k)
                    if len(c) == self.K:
                        break
        elif self.options['km_init'] == 'uniform':
            c = [rgb_to_line(k + 1) for k in xrange(self.K)]
        else:  # TODO - Opciones extra. ej. puntos con distancia maxima en el espacio, separados uniformemente ...
            print("'km_init' unspecified, using 'really_random'")
            c = np.random.rand(self.K, self.X.shape[-1])*255  # RGB

        if len(c) < self.K:
            print("La imagen tiene menos de " + str(self.K) + " colores, se han encontrado " + str(len(c)))
            self.K = len(c)

        self.centroids = np.array(c)

    def _cluster_points(self):
        """Calculates the closest centroid of all points in X"""
        self.clusters = np.argmin(distance(self.X, self.centroids), axis=1)

    def _get_centroids(self):
        """Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid"""
        self.old_centroids = self.centroids.copy()

        for k in xrange(self.K):
            a = np.mean(self.X[np.where(self.clusters == k)], axis=0)
            if not np.allclose(a, np.array([np.nan, np.nan, np.nan]), equal_nan=True):
                self.centroids[k] = a
            # else: self.centroids[k] = np.array([0]*self.X.shape[-1])  # TODO - cambiar por lo que tocaria

    def _converges(self):
        """Checks if there is a difference between current and old centroids"""
        return np.allclose(self.centroids, self.old_centroids, rtol=0, atol=self.options['tolerance'])

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

        if True:
            from sklearn.cluster import KMeans as camins
            # Los resultados de sklearn coinciden visualmente, pero falla mas que lo nuestro el test 5
            # Tambien es mas lento para algritmo 'full', que hay que usar para que coincida con los test

            if self.options['max_iter'] == np.inf:
                self.options['max_iter'] = 300

            # self.options['max_iter'] = 10
            # self.options['tolerance'] = 1

            kmeans = camins(n_clusters=self.K, init=self.centroids, n_init=1, max_iter=self.options['max_iter'],
                tol=self.options['tolerance'], algorithm='full').fit(self.X)
            self.centroids = kmeans.cluster_centers_
            self.clusters = kmeans.labels_
            if self.options['verbose']:
                self.plot()

            # self._init_rest(self.K)
            # self._iterate(True)
            # while self.options['max_iter'] > self.num_iter and not self._converges():
            #     self._iterate(False)
            return

        self._iterate(True)
        while self.options['max_iter'] > self.num_iter and not self._converges():
            self._iterate(False)

    def bestK(self):
        """Runs K-Means multiple times to find the best K for the current data given the 'fitting' method.
        In case of Fisher elbow method is recommended.

        At the end, self.centroids and self.clusters contains the information for the best K.
        TODO - NO need to rerun KMeans.

        :rtype: int
        :return: the best K found.
        """
        fit = 0
        best = 2
        for k in xrange(2, 10):
            self._init_rest(4)
            self.run()
            f = self.fitting()
            if f > fit:
                best = k
                fit = f
        return best

    def fitting(self):
        """:return: a value describing how well the current kmeans fits the data\n:rtype: float"""
        def fisher():
            media = np.mean(self.X, axis=0).reshape(-1, 3)
            media_k, between_k = self.centroids.copy(), []

            for k in xrange(self.K):
                cluster = self.X[np.where(self.clusters == k)]
                between_k.append(np.mean(distance(cluster, media_k[k].reshape(-1, 3))))

            within = np.mean(distance(media_k, media))
            between = np.mean(between_k)
            return within/between

        if self.options['fitting'] == 'fisher':
            if self.K == 0:
                return np.nan  # within/between = 0/0
            elif self.K == 1:
                return np.inf  # within/between = algo/0
            else:
                return fisher()
        else:
            return np.random.rand(1)

    def plot(self, first_time=True):
        """Plots the results"""
        # markers_shape = 'ov^<>1234sp*hH+xDd'
        markers_color = 'bgrcmy'*int(1 + self.K/7)
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
            x, y, z = Xt[self.clusters == k, 0], Xt[self.clusters == k, 1], Xt[self.clusters == k, 2]
            plt.gca().plot(x, y, z, '.' + markers_color[k])
            plt.gca().plot(Ct[k, 0:1], Ct[k, 1:2], Ct[k, 2:3], 'o'+'k', markersize=12)

        if first_time:
            plt.xlabel('dim 1')
            plt.ylabel('dim 2')
            plt.gca().set_zlabel('dim 3')
        plt.draw()
        plt.pause(0.01)
