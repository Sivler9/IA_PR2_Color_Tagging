#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Classes:
 - :class:`KMeans`\n
Functions:
 - :func:`distance(X, C)<distance>`
"""

import sys

reload(sys)
sys.setdefaultencoding('utf8')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import mpl_toolkits.mplot3d.axes3d as axes3d
from sklearn.cluster import KMeans as camins
from sklearn.metrics import euclidean_distances, calinski_harabaz_score, silhouette_score  # pairwise_distances_argmin


def gap(data, nrefs=3, maxClusters=15):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    for gap_index, k in enumerate(range(1, maxClusters)):

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):

            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)

            # Fit to it
            km = camins(k)
            km.fit(randomReference)

            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = camins(k)
        km.fit(data)

        origDisp = km.inertia_

        # Calculate gap statistic
        gap = np.mean(np.log(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

    return gaps.argmax() + 1  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal


def distance(X, C):
    """Calculates the distance between each pixcel and each centroid.

    :param numpy.ndarray X: PxD 1st set of data points (usually data points).
    :param numpy.ndarray C: KxD 2nd set of data points (usually cluster centroids points).

    :rtype: numpy.ndarray
    :return: PxK position ij is the distance between the i-th point of X and the j-th point of C.
    """
    # np.array([np.sqrt(np.sum((X - c) ** 2, axis=1)) for c in C])  # Mas lento y en _cluster_points() axis=0
    return euclidean_distances(X, C)


class KMeans:
    def __init__(self, X, K, options=None):
        """Constructor of KMeans class

        :param numpy.ndarray X: input data
        :param int K: number of centroids
        :param dict or None options: dctionary with options
        """
        self.pca = None
        self.rgb_centroids = None
        self._init_options(options if options else {})     # DICT options
        self._init_X(X)                                    # LIST data coordinates
        self._init_rest(K)                                 # Initializes de rest of the object

    def _init_X(self, X):
        """Initialization of all pixels.
        Sets X an as an array of data in vector form (PxD where P=N*M and D=3 in the above example).

        :param numpy.ndarray X: list of all pixel values. Usually a numpy array containing an image NxMx3.
            Color-space position (R, G , B) must be at deepest array.
        """
        self.X = X.reshape(-1, X.shape[-1]).astype(np.float64)

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

        options['km_init'] = options['km_init'].lower()
        options['fitting'] = options['fitting'].lower()

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

    init = {'first', 'random', 'kmeans++'}

    def _init_centroids(self):
        """Initialization of centroids depends on self.options['km_init']"""
        if self.X.shape[0] < self.K:
            print("La imagen tiene menos de " + str(self.K) + " pixeles, usando K igual a " + str(self.X.shape[0]))
            self.K = self.X.shape[0]

        centros = []
        if self.options['km_init'] == 'first':
            # c = np.unique(self.X, axis=1)[:self.K]  # Mas lento
            for pixel in self.X:
                pixel = pixel.tolist()
                if pixel not in centros:
                    centros.append(pixel)
                    if len(centros) == self.K:
                        break
        elif self.options['km_init'] == 'random':
            for pixel in np.random.randint(0, self.X.shape[0], size=self.K + int(np.sqrt(self.X.shape[0]))):
                pixel = self.X[pixel].tolist()
                if pixel not in centros:
                    centros.append(pixel)
                    if len(centros) == self.K:
                        break
        elif self.options['km_init'] == 'kmeans++':
            centros = camins(n_clusters=self.K, init='k-means++', n_init=1, max_iter=1).fit(self.X).cluster_centers_
        else:  # TODO - Opciones extra. ej. puntos con distancia maxima en el espacio, separados uniformemente ...
            print("'km_init' unspecified, using 'really_random'")
            centros = np.random.rand(self.K, self.X.shape[-1])*255  # RGB

        if len(centros) < self.K:
            print("La imagen tiene menos de " + str(self.K) + " colores, se han encontrado " + str(len(centros)))
            self.K = len(centros)

        self.centroids = np.array(centros)

    def _cluster_points(self):
        """Calculates the closest centroid of all points in X"""
        self.clusters = np.argmin(distance(self.X, self.centroids), axis=1)
        # pairwise_distances_argmin(self.X, self.centroids)  # Mas lento

    def _get_centroids(self):
        """Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid"""
        self.old_centroids = self.centroids.copy()

        for k in xrange(self.K):
            a = np.mean(self.X[np.where(self.clusters == k)], axis=0)
            if not np.allclose(a, np.array([np.nan]*self.X.shape[-1]), equal_nan=True):
                self.centroids[k] = a
            # else: self.centroids[k] = np.array([0]*self.X.shape[-1])

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

        #     from sklearn.cluster import KMeans as camins
        #     # Los resultados de sklearn coinciden visualmente, pero es mas lento para algritmo 'full'
        #
        #     if self.options['max_iter'] == np.inf:
        #         self.options['max_iter'] = 300
        #
        #     kmeans = camins(n_clusters=self.K, init=self.centroids, n_init=1, max_iter=self.options['max_iter'],
        #         tol=self.options['tolerance'], algorithm='full').fit(self.X)
        #     self.centroids = kmeans.cluster_centers_
        #     self.clusters = kmeans.labels_
        #     return

        self._iterate(True)
        while self.options['max_iter'] > self.num_iter and not self._converges():
            self._iterate(False)

    def bestK(self):
        """Runs K-Means multiple times to find the best K for the current data given the 'fitting' method.
        In case of Fisher elbow method is recommended.

        At the end, self.centroids and self.clusters contains the information for the best K.
        NO need to rerun KMeans.

        :rtype: int
        :return: the best K found.
        """
        best, center = 15, []
        if self.options['fitting'] == 'gap':
            best = gap(self.X)
            self._init_rest(best)
            self.run()
            return best
        if self.options['fitting'] == 'jump':
            return self.jumpMethod()
        elif self.options['fitting'] == 'fisher':  # elbow method
            fit, threshold = np.inf, 2.3

            self._init_rest(2)
            self.run()
            center.append([self.fitting(), self.centroids, self.clusters])

            self._init_rest(3)
            self.run()
            center.append([self.fitting(), self.centroids, self.clusters])

            for k in xrange(4, 16 + 1):
                self._init_rest(k)
                self.run()

                center.append([self.fitting(), self.centroids, self.clusters])
                if (center[-3][0] - center[-2][0]) > (center[-2][0] - center[-1][0])*threshold:
                    center = center[-2][1:]
                    best = k - 1
                    break
            else:
                center = center[-2][1:]
                best = 15
        else:
            fit = -np.inf
            for k in xrange(2, 16 + 1):
                self._init_rest(k)
                self.run()

                f = self.fitting()
                if f > fit:
                    fit, best, center = f, k, [self.centroids, self.clusters]

        self.centroids, self.clusters = center

        return best

    fit = {'gap': None, 'jump': None, 'fisher': None,  # Estan definidos dentro de fitting()
           'calinski_harabaz': calinski_harabaz_score}  # , 'silhouette': silhouette_score} peta memoria

    def fitting(self):
        """:return: a value describing how well the current kmeans fits the data\n:rtype: float"""
        def fisher():
            media = np.mean(self.X, axis=0).reshape(-1, self.X.shape[-1])
            media_k, between_k = self.centroids.copy(), []

            for k in xrange(self.K):
                cluster = self.X[np.where(self.clusters == k)].reshape(-1, self.X.shape[-1])
                between_k.append(np.mean(distance(cluster, media_k[k].reshape(-1, self.X.shape[-1]))))

            within = np.mean(distance(media_k, media))
            between = np.mean(between_k)
            return within/between

        if self.K == 0:
            return 0  # np.nan  # fisher -> within/between = 0/0
        elif self.K == 1:
            return 1  # np.inf  # fisher -> within/between = algo/0
        elif self.options['fitting'] == 'fisher':
            return fisher()
        elif self.options['fitting'] in KMeans.fit:
            return KMeans.fit[self.options['fitting']](self.X, self.clusters)
        else:
            return calinski_harabaz_score(self.X, self.clusters)  # np.random.rand(1)

    def jumpMethod(self):
        data = self.X
        clusters, centroids = [], []
        # dimension of 'data'; data.shape[0] would be size of 'data'
        p = data.shape[1]
        # vector of variances (1 by p)
        # using squared error rather than Mahalanobis distance' (SJ, p. 12)
        sigmas = np.var(data, axis=0)
        # by following the authors we assume 0 covariance between p variables (SJ, p. 12)
        # start with zero-matrix (p by p)
        Sigma = np.zeros((p, p), dtype=np.float32)
        # fill the main diagonal with variances for
        np.fill_diagonal(Sigma, val=sigmas)
        # calculate the inversed matrix
        Sigma_inv = np.linalg.inv(Sigma)

        cluster_range = range(1, 13 + 1)
        distortions = np.repeat(0, len(cluster_range) + 1).astype(np.float32)

        # for each k in cluster range implement
        for k in cluster_range:
            # initialize and fit the clusterer giving k in the loop
            self._init_rest(k)
            self.run()
            centroids.append(self.centroids)
            clusters.append(self.clusters)
            # calculate centers of suggested k clusters
            centers = self.centroids
            # since we need to calculate the mean of mins create dummy vec
            for_mean = np.repeat(0, len(data)).astype(np.float32)

            # for each observation (i) in data implement
            for i in range(len(data)):
                # dummy for vec of distances between i-th obs and k-center
                dists = np.repeat(0, k).astype(np.float32)

                # for each cluster in KMean clusters implement
                for cluster in range(k):
                    # calculate the within cluster dispersion
                    tmp = np.transpose(data[i] - centers[cluster])
                    # using squared error rather than Mahalanobis distance' (SJ, p. 12)
                    dists[cluster] = tmp.dot(Sigma_inv).dot(tmp)
                    # dists[cluster] = tmp.dot(tmp)

                # take the lowest distance to a class
                for_mean[i] = min(dists)

            # take the mean for mins for each observation
            distortions[k] = np.mean(for_mean) / p

        Y = p / 2
        # the first (by convention it is 0) and the second elements
        jumps = [0] + [distortions[1] ** (-Y) - 0]
        jumps += [distortions[k]**(-Y) - distortions[k-1]**(-Y) for k in range(2, len(distortions))]

        # calculate recommended number of clusters
        bestK = np.argmax(np.array(jumps))
        self.centroids = centroids[bestK-1]
        self.clusters = clusters[bestK-1]
        self.K = bestK

        return bestK

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
