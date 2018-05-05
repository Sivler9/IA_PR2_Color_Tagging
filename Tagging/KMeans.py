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
            self.X = X.reshape(-1, X.shape[2])
        else:
            self.X = np.copy(X)

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
            #self.centroids = np.unique(self.X, axis=0)[:self.K]  # copy first K elements of X
            '''x = np.random.rand(self.X.shape[1])
            y = self.X.dot(x) #"y" array de dimensiones Px1
            unique, index = np.unique(y.round(decimals=8), return_index=True)#Redondeamos los valores de "y" a 4 decimales y cogemos los valores unicos
            self.centroids = np.copy(self.X[np.sort(index)][:self.K])'''
            unique, index = np.unique(self.X,axis=0, return_index=True)
            index = np.sort(index)
            self.centroids = np.array(self.X[index[:self.K]])
            '''
            self.centroids = []
            i = 0
            for pixel in self.X:
                if pixel not in self.centroids:
                    self.centroids = np.append(self.centroids, pixel)
                    i += 1
                if i == self.K:
                    break
            '''
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
        '''
        distancias = distance(self.X,self.centroids)
        for i,distancia in enumerate(distancias):
            self.clusters[i] = np.argmin(distancia)
        '''
        self.clusters = pairwise_distances_argmin(self.X, self.centroids)

        '''for i in range(self.X.shape[0]):
            distanciaTemporal = np.inf
            valorFinal = -1
            for j, distanciaActual in enumerate(distancias[i]):
                if distanciaTemporal > distanciaActual:
                    distanciaTemporal = distanciaActual
                    valorFinal = j
            self.clusters[i] = valorFinal'''

    def _get_centroids(self):
        """@brief   Calculates coordinates of centroids based on the coordinates 
                    of all the points assigned to the centroid
        """
        self.old_centroids = np.copy(self.centroids)
        self.centroids = np.array([self.X[self.clusters == i].mean(0) for i in range(self.K)])
        if np.isnan(self.centroids).any():
            mask = np.where(np.isnan(self.centroids).all(axis=1))[0]
            self.centroids[mask] = self.old_centroids[mask]
        '''
            for i in range(self.centroids.shape[0]):
                tempIndex = np.where(self.clusters == i)[0]
                if len(tempIndex) > 0:
                    self.centroids[i] = np.mean(self.X[tempIndex], axis=0)'''


    def _converges(self):
        """@brief   Checks if there is a difference between current and old centroids
        """
        valor = np.allclose(self.centroids, self.old_centroids, self.options['tolerance'])
        return valor
        '''
        for i,centroide in enumerate(self.centroids):
            if(euclidean_distances(centroide.reshape(1, -1),self.old_centroids[i].reshape(1, -1)) > self.options['tolerance']):
                return False
        return True'''

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
        self._init_rest(4)
        self.run()
        #fit = self.fitting()
        return 4

    def fitting(self):
        """@brief  return a value describing how well the current kmeans fits the data
        """
        if self.options['fitting'].lower() == 'fisher' and self.K == 1:
            return 1
        elif self.options['fitting'].lower() == 'fisher' and self.K > 1:
            return 1/(metricas.calinski_harabaz_score(self.X, self.clusters)*(self.K -1)/(self.X.shape[0]-self.K)) #calinski = (Between_Variance/Whithin_Variance)*(N-k)/(K-1)
        elif self.options['fitting'].lower() == 'silhouette':
            return metricas.silhouette_score(self.X,self.clusters)
        else:
            return np.random.rand(1)

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
