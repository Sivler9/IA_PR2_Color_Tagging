"""

@author: ramon, bojana
"""
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances_argmin
import sklearn.metrics as metricas
import scipy
import scipy.cluster.vq
import scipy.spatial.distance
from sklearn.cluster import KMeans as camins

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
    """@brief   Calculates the distance between each pixcel and each centroid 

    @param  X  numpy array PxD 1st set of data points (usually data points)
    @param  C  numpy array KxD 2nd set of data points (usually cluster centroids points)

    @return dist: PxK numpy array position ij is the distance between the 
    	i-th point of the first set an the j-th point of the second set
    """
    return euclidean_distances(X,C)


class KMeans():

    def __init__(self, X, K, options=None):
        """@brief   Constructor of KMeans class
        
        @param  X   LIST    input data
        @param  K   INT     number of centroids
        @param  options DICT dctionary with options
        """

        self._init_X(X)  # LIST data coordinates
        self._init_options(options)  # DICT options
        self._init_rest(K)  # Initializes de rest of the object

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        """@brief Initialization of all pixels
        
        @param  X   LIST    list of all pixel values. Usually it will be a numpy 
                            array containing an image NxMx3

        sets X an as an array of data in vector form (PxD  where P=N*M and D=3 in the above example)
        """
        if len(X.shape) >= 3:
            self.X = X.reshape(-1, X.shape[2]).astype(np.float64)
        else:
            self.X = np.copy(X.astype(np.float64))

    def _init_options(self, options):
        """@brief Initialization of options in case some fields are left undefined
        
        @param  options DICT dctionary with options

			sets de options parameters
        """
        if options == None:
            options = {}
        if not 'km_init' in options:
            options['km_init'] = 'first'
        if not 'verbose' in options:
            options['verbose'] = False
        if not 'tolerance' in options:
            options['tolerance'] = 0
        if not 'max_iter' in options:
            options['max_iter'] = np.inf
        if not 'fitting' in options:
            options['fitting'] = 'Fisher'

        self.options = options

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_rest(self, K):
        """@brief   Initialization of the remainig data in the class.
        
        @param  options DICT dctionary with options
        """
        self.K = K  # INT number of clusters
        if self.K > 0:
            self._init_centroids()  # LIST centroids coordinates
            self.old_centroids = np.empty_like(self.centroids)  # LIST coordinates of centroids from previous iteration
            self.clusters = np.zeros(len(self.X))  # LIST list that assignes each element of X into a cluster
            self._cluster_points()  # sets the first cluster assignation
        self.num_iter = 0  # INT current iteration

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_centroids(self):
        """@brief Initialization of centroids
        depends on self.options['km_init']
        """
        if self.options['km_init'].lower() == 'first':
            unique, index = np.unique(self.X,axis=0, return_index=True)
            index = np.sort(index)
            self.centroids = np.array(self.X[index[:self.K]])
        elif self.options['km_init'].lower() == 'custom':
            self.centroids = np.zeros((self.K,self.X.shape[1]))
            for k in range(self.K): self.centroids[k,:] = k*255/(self.K-1)
        elif self.options['km_init'] == 'kmeans++':
            self.centroids = camins(n_clusters=self.K, init='k-means++', n_init=1, max_iter=1).fit(self.X).cluster_centers_
        else:
            maxtmp = self.X.max(axis=0)
            mintmp = self.X.min(axis=0)
            centroids = np.zeros((self.X.shape[1],self.K))
            for i in range(self.X.shape[1]):
                centroids[i] = np.random.uniform(low=mintmp[i],high=maxtmp[i],size=self.K)
            self.centroids = np.array(centroids.transpose())


    def _cluster_points(self):
        """@brief   Calculates the closest centroid of all points in X
        """
        self.clusters = pairwise_distances_argmin(self.X, self.centroids)

    def _get_centroids(self):
        """@brief   Calculates coordinates of centroids based on the coordinates 
                    of all the points assigned to the centroid
        """
        self.old_centroids = np.copy(self.centroids)
        self.centroids = np.array([self.X[self.clusters == i].mean(0) for i in range(self.K)])
        if np.isnan(self.centroids).any():
            mask = np.where(np.isnan(self.centroids).all(axis=1))[0]
            self.centroids[mask] = self.old_centroids[mask]


    def _converges(self):
        """@brief   Checks if there is a difference between current and old centroids
        """
        return np.allclose(self.centroids, self.old_centroids, self.options['tolerance'])

    def _iterate(self, show_first_time=True):
        """@brief   One iteration of K-Means algorithm. This method should 
                    reassigne all the points from X to their closest centroids
                    and based on that, calculate the new position of centroids.
        """
        self.num_iter += 1
        self._cluster_points()
        self._get_centroids()
        if self.options['verbose']:
            self.plot(show_first_time)

    def run(self):
        """@brief   Runs K-Means algorithm until it converges or until the number
                    of iterations is smaller than the maximum number of iterations.=
        """
        if self.K == 0:
            self.bestK()
            return

        self._iterate(True)
        self.options['max_iter'] = np.inf
        if self.options['max_iter'] > self.num_iter:
            while not self._converges():
                self._iterate(False)

    def bestK(self):
        """@brief   Runs K-Means multiple times to find the best K for the current 
                    data given the 'fitting' method. In cas of Fisher elbow method 
                    is recommended.
                    
                    at the end, self.centroids and self.clusters contains the 
                    information for the best K. NO need to rerun KMeans.
           @return B is the best K found.
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        centroids = []
        clusters = []
        bestk = 4
        #self.options['fitting'] ='gap'
        if self.options['fitting'].lower() == 'jump':
            return self.jumpMethod(clusters,centroids)

        elif self.options['fitting'].lower() == 'gap':
            bestk = gap(self.X, maxClusters=14)
            self._init_rest(bestk)
            self.run()
            return bestk
        elif self.options['fitting'].lower() == 'fisher':
            bestk, center = -1, []
            fit, threshold = np.inf, 2.3

            self._init_rest(2)
            self.run()
            center.append([self.fitting(), self.centroids, self.clusters])

            self._init_rest(3)
            self.run()
            center.append([self.fitting(), self.centroids, self.clusters])

            for k in xrange(4, 13 + 1):
                self._init_rest(k)
                self.run()

                center.append([self.fitting(), self.centroids, self.clusters])
                if (center[-3][0] - center[-2][0]) > (center[-2][0] - center[-1][0])*threshold:
                    self.centroids, self.clusters = center[-2][1:]
                    bestk = k - 1
                    break
            else:
                bestk = 4
                self.centroids, self.clusters = center[bestk-2][1:]

            self.K = bestk
            return bestk

        else:
            scores = []
            for k in range(2,14):
                self._init_rest(k)
                self.run()
                scores.append(self.fitting())
                centroids.append(self.centroids)
                clusters.append(self.clusters)
            if self.options['fitting'].lower() == 'calinski' or self.options['fitting'].lower() == 'silhouette':
                bestk = np.argmax(scores)+2

            self.centroids = centroids[bestk-2]
            self.clusters = clusters[bestk-2]
            self.K = bestk
            return bestk

    def fitting(self):
        """@brief  return a value describing how well the current kmeans fits the data
        """
        if self.K == 1:
            return 1
        elif self.options['fitting'].lower() == 'fisher' and self.K > 1:
            return 1/(metricas.calinski_harabaz_score(self.X, self.clusters)*(self.K -1)/(self.X.shape[0]-self.K)) #calinski = (Between_Variance/Whithin_Variance)*(N-k)/(K-1)
        elif self.options['fitting'].lower() == 'silhouette':
            return metricas.silhouette_score(self.X,self.clusters)
        elif self.options['fitting'].lower() == 'calinski':
            return metricas.calinski_harabaz_score(self.X, self.clusters)
        else:
            return np.random.rand(1)

    def jumpMethod(self, clusters, centroids):
        data = self.X
        # dimension of 'data'; data.shape[0] would be size of 'data'
        p = data.shape[1]
        # vector of variances (1 by p)
        #using squared error rather than Mahalanobis distance' (SJ, p. 12)
        sigmas = np.var(data, axis=0)
        ## by following the authors we assume 0 covariance between p variables (SJ, p. 12)
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
                    #using squared error rather than Mahalanobis distance' (SJ, p. 12)
                    dists[cluster] = tmp.dot(Sigma_inv).dot(tmp)
                    #dists[cluster] = tmp.dot(tmp)

                # take the lowest distance to a class
                for_mean[i] = min(dists)

            # take the mean for mins for each observation
            distortions[k] = np.mean(for_mean) / p

        Y = p / 2
        # the first (by convention it is 0) and the second elements
        jumps = [0] + [distortions[1] ** (-Y) - 0]
        jumps += [distortions[k] ** (-Y) \
                       - distortions[k-1] ** (-Y) \
                       for k in range(2, len(distortions))]

        # calculate recommended number of clusters
        bestK = np.argmax(np.array(jumps))
        self.centroids = centroids[bestK-1]
        self.clusters = clusters[bestK-1]
        self.K = bestK
        """plt.figure(2)
        plt.cla()
        plt.plot(range(16),jumps)
        plt.xlabel('K')
        plt.ylabel('fitting score')
        plt.draw()
        plt.pause(20)"""
        return bestK

    def plot(self, first_time=True):
        """@brief   Plots the results
        """

        # markersshape = 'ov^<>1234sp*hH+xDd'
        markerscolor = 'bgrcmybgrcmybgrcmyk'
        if first_time:
            plt.gcf().add_subplot(111, projection='3d')
            plt.ion()
            plt.show()

        if self.X.shape[1] > 3:
            if not hasattr(self, 'pca'):
                self.pca = PCA(n_components=3)
                self.pca.fit(self.X)
            Xt = self.pca.transform(self.X)
            Ct = self.pca.transform(self.centroids)
        else:
            Xt = self.X
            Ct = self.centroids

        for k in range(self.K):
            plt.gca().plot(Xt[self.clusters == k, 0], Xt[self.clusters == k, 1], Xt[self.clusters == k, 2],
                           '.' + markerscolor[k])
            plt.gca().plot(Ct[k, 0:1], Ct[k, 1:2], Ct[k, 2:3], 'o' + 'k', markersize=12)

        if first_time:
            plt.xlabel('dim 1')
            plt.ylabel('dim 2')
            plt.gca().set_zlabel('dim 3')
        plt.draw()
        plt.pause(0.01)
