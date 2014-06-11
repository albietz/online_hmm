import numpy as np
from numpy import newaxis as nax
from numpy.linalg import det, inv
from scipy import stats

class Distribution(object):
    def log_pdf(self, X):
        pass

    def pdf(self, X):
        pass

    def distances(self, X):
        pass

    def max_likelihood(self, X, weights):
        pass

class Gaussian(Distribution):
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    @property
    def dim(self):
        return len(self.mean)

    def distances(self, X):
        diff = X - self.mean
        return 0.5 * np.diag(diff.dot(inv(self.cov)).dot(diff.T))

    def max_likelihood(self, X, weights):
        self.mean = np.sum(weights[:,nax] * X, axis=0) / np.sum(weights)
        diff = X - self.mean
        self.cov = diff.T.dot(weights[:,nax] * diff) / np.sum(weights)

    def log_pdf(self, X):
        if len(X.shape) < 2:
            X = X[nax,:]
        d = self.mean.shape[0]
        diff = X - self.mean
        return - 0.5 * d * np.log(2*np.pi) - 0.5 * np.log(det(self.cov)) \
                - 0.5 * np.diag(diff.dot(inv(self.cov)).dot(diff.T))

    def pdf(self, X):
        if len(X.shape) < 2:
            X = X[nax,:]
        d = self.mean.shape[0]
        diff = X - self.mean
        return 1. / np.sqrt((2*np.pi)**d * det(self.cov)) \
                * np.exp(-0.5 * np.diag(diff.dot(inv(self.cov)).dot(diff.T)))

    def sample(self, size=1):
        return np.random.multivariate_normal(self.mean, self.cov, size)

    def new_sufficient_statistics_hmm(self, x, cluster_id, K):
        return GaussianSufficientStatisticsHMM(x, cluster_id, K, self.dim)

    def new_sufficient_statistics_hsmm(self, x, cluster_id, K, D):
        return GaussianSufficientStatisticsHSMM(x, cluster_id, K, D, self.dim)

    def online_max_likelihood(self, rho_obs, phi):
        s0, s1, s2 = rho_obs.get_statistics(phi)
        self.mean = s1 / s0
        self.cov = s2 / s0 - self.mean[:,nax] * self.mean

# for euclidian K-means or isotropic Gaussian
class SquareDistance(Distribution):
    def __init__(self, mean, sigma2=None):
        self.mean = mean
        self.sigma2 = sigma2

    @property
    def cov(self):
        return self.sigma2 * np.eye(2)

    @property
    def dim(self):
        return len(self.mean)

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

    def pdf(self, X):
        assert self.sigma2 is not None, 'only for isotropic Gaussian'
        d = self.mean.shape[0]
        diff = X - self.mean
        dists = np.sum(diff*diff, axis=1)

        return 1. / np.sqrt((2*np.pi*self.sigma2)**d) \
                * np.exp(-0.5 * dists / self.sigma2)

class KL(Distribution):
    '''Basically a multinomial.'''
    def __init__(self, mean):
        self.mean = mean

    @property
    def dim(self):
        return len(self.mean)

    def distances(self, X):
        return - X.dot(np.log(self.mean))

    def max_likelihood(self, X, weights):
        if weights.dtype == np.bool:
            self.mean = X[weights,:].mean(axis=0)
        else:
            self.mean = np.sum(weights[:,nax] * X, axis=0) / np.sum(weights)

    def online_update(self, x, step):
        self.mean = (1 - step) * self.mean + step * x

    def log_pdf(self, X):
        # log p(x|theta) = sum_j x_j log(theta_j)
        return X.dot(np.log(self.mean))

    def pdf(self, X):
        return np.exp(X.dot(np.log(self.mean)))

    def new_sufficient_statistics_hmm(self, x, cluster_id, K):
        return KLSufficientStatisticsHMM(x, cluster_id, K, self.dim)

    def new_sufficient_statistics_hsmm(self, x, cluster_id, K, D):
        return KLSufficientStatisticsHSMM(x, cluster_id, K, D, self.dim)

    def online_max_likelihood(self, rho_obs, phi):
        s0, s1 = rho_obs.get_statistics(phi)
        self.mean = s1 / s0

class DurationDistribution(Distribution):
    def __init__(self, D):
        self.D = D

    def log_pmf(self, X):
        pass

    def log_vec(self):
        return self.log_pmf(np.arange(1,self.D+1))

    def d_frac(self):
        v = self.log_vec()
        D = np.hstack((np.cumsum(v[::-1])[::-1], 0.))
        return D[1:] / D[:-1]

class PoissonDuration(DurationDistribution):
    def __init__(self, lmbda, D):
        super(PoissonDuration, self).__init__(D)
        self.lmbda = lmbda

    def log_pmf(self, X):
        return stats.poisson.logpmf(X, self.lmbda)

    def sample(self, size=None):
        return stats.poisson.rvs(self.lmbda, size=size)

class NegativeBinomial(DurationDistribution):
    def __init__(self, r, p, D):
        super(NegativeBinomial, self).__init__(D)
        self.r = r
        self.p = p

    def log_pmf(self, X):
        return stats.nbinom.logpmf(X, self.r, self.p)

    def sample(self, size=None):
        return stats.nbinom.rvs(self.r, self.p, size=size)

# Sufficient Statistics classes
class SufficientStatistics(object):
    def online_update(self, x, r, step):
        raise NotImplementedError

    def get_statistics(self, phi):
        raise NotImplementedError

class SufficientStatisticsHMM(SufficientStatistics):
    def __init__(self, cluster_id, K):
        self.cluster_id = cluster_id
        self.K = K

class GaussianSufficientStatisticsHMM(SufficientStatisticsHMM):
    def __init__(self, x, cluster_id, K, size):
        super(GaussianSufficientStatisticsHMM, self).__init__(cluster_id, K)
        # 1{Z_t = i}
        self.rho0 = np.zeros(self.K)
        self.rho0[self.cluster_id] = 1.
        # 1{Z_t = i} x_t
        self.rho1 = np.zeros((size, self.K))
        self.rho1[:,self.cluster_id] = x
        # 1{Z_t = i} x_t x_t'
        self.rho2 = np.zeros((size, size, self.K))
        self.rho2[:,:,self.cluster_id] = x[:,nax] * x

    def online_update(self, x, r, step):
        self.rho0 = (1 - step) * self.rho0.dot(r)
        self.rho0[self.cluster_id] += step
        self.rho1 = (1 - step) * self.rho1.dot(r)
        self.rho1[:,self.cluster_id] += step * x
        self.rho2 = (1 - step) * self.rho2.dot(r)
        self.rho2[:,:,self.cluster_id] += step * x[:,nax] * x

    def get_statistics(self, phi):
        return self.rho0.dot(phi), self.rho1.dot(phi), self.rho2.dot(phi)

class KLSufficientStatisticsHMM(SufficientStatisticsHMM):
    def __init__(self, x, cluster_id, K, size):
        super(KLSufficientStatisticsHMM, self).__init__(cluster_id, K)
        # 1{Z_t = i}
        self.rho0 = np.zeros(self.K)
        self.rho0[self.cluster_id] = 1.
        # 1{Z_t = i} x_t
        self.rho = np.zeros((size, self.K))
        self.rho[:,self.cluster_id] = x

    def online_update(self, x, r, step):
        self.rho0 = (1 - step) * self.rho0.dot(r)
        self.rho0[self.cluster_id] += step
        self.rho1 = (1 - step) * self.rho.dot(r)
        self.rho1[:,self.cluster_id] += step * x

    def get_statistics(self, phi):
        return self.rho0.dot(phi), self.rho1.dot(phi)

class SufficientStatisticsHSMM(SufficientStatistics):
    def __init__(self, cluster_id, K, D):
        self.cluster_id = cluster_id
        self.K = K
        self.D = D

class GaussianSufficientStatisticsHSMM(SufficientStatisticsHSMM):
    def __init__(self, x, cluster_id, K, D, size):
        super(GaussianSufficientStatisticsHSMM, self).__init__(cluster_id, K, D)
        # 1{Z_t = i}
        self.rho0 = np.zeros((self.K, self.D))
        self.rho0[self.cluster_id] = 1.
        # 1{Z_t = i} x_t
        self.rho1 = np.zeros((size, self.K, self.D))
        self.rho1[:,self.cluster_id] = x[:,nax]
        # 1{Z_t = i} x_t x_t'
        self.rho2 = np.zeros((size, size, self.K, self.D))
        self.rho2[:,:,self.cluster_id] = (x[:,nax] * x)[:,:,nax]

    def online_update(self, x, r, step):
        rho0 = np.zeros(self.rho0.shape)
        rho0[:,0] = (1 - step) * np.tensordot(self.rho0, r)
        rho0[self.cluster_id,0] += step
        rho0[:,1:] = (1 - step) * self.rho0[:,:-1]
        self.rho0 = rho0

        rho1 = np.zeros(self.rho1.shape)
        rho1[:,:,0] = (1 - step) * np.tensordot(self.rho1, r)
        rho1[:,self.cluster_id,0] += step * x[:,nax]
        rho1[:,:,1:] = (1 - step) * self.rho1[:,:,:-1]
        self.rho1 = rho1

        rho2 = np.zeros(self.rho2.shape)
        rho2[:,:,:,0] = (1 - step) * np.tensordot(self.rho2, r)
        rho2[:,:,self.cluster_id,0] += step * (x[:,nax] * x)[:,nax]
        rho2[:,:,:,1:] = (1 - step) * self.rho2[:,:,:,:-1]
        self.rho2 = rho2

    def get_statistics(self, phi):
        return np.tensordot(self.rho0, phi), np.tensordot(self.rho1, phi),
            np.tensordot(self.rho2, phi)

class KLSufficientStatisticsHSMM(SufficientStatisticsHSMM):
    def __init__(self, x, cluster_id, K, size):
        super(KLSufficientStatisticsHSMM, self).__init__(cluster_id, K)
        # 1{Z_t = i}
        self.rho0 = np.zeros((self.K, self.D))
        self.rho0[self.cluster_id] = 1.
        # 1{Z_t = i} x_t
        self.rho1 = np.zeros((size, self.K, self.D))
        self.rho1[:,self.cluster_id] = x[:,nax]

    def online_update(self, x, r, step):
        rho0 = np.zeros(self.rho0.shape)
        rho0[:,0] = (1 - step) * np.tensordot(self.rho0, r)
        rho0[self.cluster_id,0] += step
        rho0[:,1:] = (1 - step) * self.rho0[:,:-1]
        self.rho0 = rho0

        rho1 = np.zeros(self.rho1.shape)
        rho1[:,:,0] = (1 - step) * np.tensordot(self.rho1, r)
        rho1[:,self.cluster_id,0] += step * x[:,nax]
        rho1[:,:,1:] = (1 - step) * self.rho1[:,:,:-1]
        self.rho1 = rho1

    def get_statistics(self, phi):
        return np.tensordot(self.rho0, phi), np.tensordot(self.rho1, phi)
