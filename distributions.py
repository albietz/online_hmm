import numpy as np
from numpy import newaxis as nax
from numpy.linalg import det, inv

class Distribution(object):
    def log_pdf(self, X):
        pass

    def pdf(self, X, normalized):
        pass

    def distances(self, X):
        pass

    def max_likelihood(self, X, weights):
        pass

class Gaussian(Distribution):
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    def distances(self, X):
        diff = X - self.mean
        return 0.5 * np.diag(diff.dot(inv(self.cov)).dot(diff.T))

    def max_likelihood(self, X, weights):
        self.mean = np.sum(weights[:,nax] * X, axis=0) / np.sum(weights)
        diff = X - self.mean
        self.cov = diff.T.dot(weights[:,nax] * diff) / np.sum(weights)

    def log_pdf(self, X):
        d = self.mean.shape[0]
        diff = X - self.mean
        return - 0.5 * d * np.log(2*np.pi) - 0.5 * np.log(det(self.cov)) \
                - 0.5 * np.diag(diff.dot(inv(self.cov)).dot(diff.T))

    def pdf(self, X, normalized=True):
        d = self.mean.shape[0]
        diff = X - self.mean
        if normalized:
            return 1. / np.sqrt((2*np.pi)**d * det(self.cov)) \
                    * np.exp(-0.5 * np.diag(diff.dot(inv(self.cov)).dot(diff.T)))
        else:
            return np.exp(-0.5 * np.diag(diff.dot(inv(self.cov)).dot(diff.T)))


# for euclidian K-means or isotropic Gaussian
class SquareDistance(Distribution):
    def __init__(self, mean, sigma2=None):
        self.mean = mean
        self.sigma2 = sigma2

    @property
    def cov(self):
        return self.sigma2 * np.eye(2)

    def distances(self, X):
        diff = X - self.mean
        return np.sum(diff * diff, axis=1)

    def max_likelihood(self, X, weights):
        if weights.dtype == np.bool:
            self.mean = X[weights,:].mean(axis=0)
        else:
            self.mean = np.sum(weights[:,nax] * X, axis=0) / np.sum(weights)

        if self.sigma2 is not None:
            diff = X - self.mean
            dists = np.sum(diff * diff, axis=1)
            self.sigma2 = 0.5 * dists.dot(weights) / np.sum(weights)

    def log_pdf(self, X):
        assert self.sigma2 is not None, 'only for isotropic Gaussian'
        d = self.mean.shape[0]
        diff = X - self.mean
        dists = np.sum(diff*diff, axis=1)
        return - 0.5 * d * np.log(2*np.pi) - 0.5 * d * np.log(self.sigma2) \
                - 0.5 * dists / self.sigma2

    def pdf(self, X, normalized=True):
        assert self.sigma2 is not None, 'only for isotropic Gaussian'
        d = self.mean.shape[0]
        diff = X - self.mean
        dists = np.sum(diff*diff, axis=1)

        if normalized:
            return 1. / np.sqrt((2*np.pi*self.sigma2)**d) \
                    * np.exp(-0.5 * dists / self.sigma2)
        else:
            return np.exp(-0.5 * dists / self.sigma2)

class KL(Distribution):
    '''Basically a multinomial.'''
    def __init__(self, mean):
        self.mean = mean

    def distances(self, X):
        return - X.dot(np.log(self.mean))

    def max_likelihood(self, X, weights):
        if weights.dtype == np.bool:
            self.mean = X[weights,:].mean(axis=0)
        else:
            self.mean = np.sum(weights[:,nax] * X, axis=0) / np.sum(weights)

    def log_pdf(self, X):
        # log p(x|theta) = sum_j x_j log(theta_j)
        return X.dot(np.log(self.mean))

    def pdf(self, X, normalized=True):
        return np.exp(X.dot(np.log(self.mean)))

