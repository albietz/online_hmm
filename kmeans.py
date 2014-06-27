import copy
import distributions
import numpy as np
import matplotlib.pyplot as plt

def distortion(X, assignments, centers):
    deltas = X - centers[assignments,:]
    return np.sum(deltas * deltas)

class BadCentroids(Exception):
    pass

def kmeans(X, init_centroids):
    N = X.shape[0]
    K = len(init_centroids)
    centroids = copy.deepcopy(init_centroids)
    distortions = []
    dist = 1e9
    dist_prev = 2e9
    while True:
        dist_prev = dist

        # E-step
        distances = np.zeros((N, K))
        for k in range(K):
            distances[:,k] = centroids[k].distances(X)
        if np.any(np.isnan(distances)):
            raise BadCentroids

        assignments = np.argmin(distances, axis=1)
        dist = np.min(distances, axis=1).sum()
        distortions.append(dist)
        if dist == dist_prev:
            break

        # M-step
        for k in range(K):
            centroids[k].max_likelihood(X, assignments == k)

    return assignments, centroids, distortions

def kmeans_best_of_n(X, K, n_trials, dist_cls=None, debug=False):
    '''
    Tries 'n_trials' random initializations and returns
    result with lowest distortion.
    '''
    dist_cls = dist_cls or distributions.SquareDistance
    d_best = None
    assignments, centroids, distortions = None, None, None
    for i in range(n_trials):
        perm = np.random.permutation(X.shape[0])
        clusters = []
        for k in range(K):
            clusters.append(dist_cls(X[perm[k],:]))
        try:
            a, c, d = kmeans(X, clusters)
        except BadCentroids:
            print 'Bad centroids, skipping'
            continue

        if debug:
            print 'K-means trial {}: {}'.format(i+1, d[-1])
        if d_best is None or d[-1] < d_best:
            assignments, centroids, distortions = a, c, d

    return assignments, centroids, distortions

def plot_kmeans(X, assignments, centroids):
    centers = np.vstack(c.mean for c in centroids)
    plt.figure()
    plt.scatter(X[:,0], X[:,1], c=assignments)
    plt.scatter(centers[:,0], centers[:,1], color='green', s=100)
    plt.title('K-means, distortion: {}'.format(distortion(X, assignments, centers)))

if __name__ == '__main__':
    X = np.loadtxt('EMGaussian.data')
    N = X.shape[0]
    K = 4

    a, m, _ = kmeans_best_of_n(X, K, n_trials=1)
    plot_kmeans(X, a, m)
