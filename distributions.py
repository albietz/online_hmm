import numpy as np
from numpy import newaxis as nax
from numpy.linalg import det, inv
from scipy import stats

class Distribution(object):
    def log_pdf(self, X):
        raise NotImplementedError

    def pdf(self, X):
        raise NotImplementedError

    def distances(self, X):
        raise NotImplementedError

    def max_likelihood(self, X, weights):
        raise NotImplementedError

class Gaussian(Distribution):
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    def __repr__(self):
        return '<Gaussian: mean={}, cov={}>'.format(repr(self.mean), repr(self.cov))

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

    def new_incremental_sufficient_statistics(self, x, phi, cluster_id):
        return GaussianISufficientStatistics(x, phi, cluster_id, self.dim)

    def online_max_likelihood(self, rho_obs, phi=None, t=None):
        if phi is None:
            s0, s1, s2 = rho_obs.get_statistics()
        else:
            s0, s1, s2 = rho_obs.get_statistics(phi)
        self.mean = s1 / s0
        self.cov = s2 / s0 - self.mean[:,nax] * self.mean

# for euclidian K-means or isotropic Gaussian
class SquareDistance(Distribution):
    def __init__(self, mean, sigma2=None, tau=None, kappa=None):
        self.mean = mean
        self.sigma2 = sigma2

        self.tau = tau
        self.kappa = kappa

        if tau is None or kappa is None:
            self.map = False
            self.tau = 0
            self.kappa = 0
        else:
            assert tau is not None and kappa is not None
            self.map = True

    def __repr__(self):
        return '<SquareDistance: mean={}, sigma2={}>'.format(repr(self.mean), repr(self.sigma2))

    @property
    def cov(self):
        s2 = self.sigma2 or 1
        return s2 * np.eye(len(self.mean))

    def to_gaussian(self):
        if self.sigma2 is not None:
            return Gaussian(self.mean, self.cov)
        else:
            return Gaussian(self.mean, np.eye(len(self.mean)))

    @property
    def dim(self):
        return len(self.mean)

    def distances(self, X):
        if len(X.shape) < 2:
            X = X[nax,:]
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
        if len(X.shape) < 2:
            X = X[nax,:]
        if self.sigma2 is None:
            return -self.distances(X)

        d = self.mean.shape[0]
        diff = X - self.mean
        dists = np.sum(diff*diff, axis=1)
        return - 0.5 * d * np.log(2*np.pi) - 0.5 * d * np.log(self.sigma2) \
                - 0.5 * dists / self.sigma2

    def pdf(self, X):
        if len(X.shape) < 2:
            X = X[nax,:]
        if self.sigma2 is None:
            return np.exp(-self.distances(X))

        d = self.mean.shape[0]
        diff = X - self.mean
        dists = np.sum(diff*diff, axis=1)

        return 1. / np.sqrt((2*np.pi*self.sigma2)**d) \
                * np.exp(-0.5 * dists / self.sigma2)

    def sample(self, size=1):
        return np.random.multivariate_normal(self.mean, self.cov, size)

    def new_sufficient_statistics_hmm(self, x, cluster_id, K):
        return KLSufficientStatisticsHMM(x, cluster_id, K, self.dim)

    def new_sufficient_statistics_hsmm(self, x, cluster_id, K, D):
        return KLSufficientStatisticsHSMM(x, cluster_id, K, D, self.dim)

    def new_incremental_sufficient_statistics(self, x, phi, cluster_id):
        return KLISufficientStatistics(x, phi, cluster_id, self.dim)

    def online_max_likelihood(self, rho_obs, phi=None, t=None):
        if phi is None:
            s0, s1 = rho_obs.get_statistics()
        else:
            s0, s1 = rho_obs.get_statistics(phi)

        if self.map: # MAP
            assert t is not None
            self.mean = (self.tau * self.kappa + t * s1) / (self.tau + t * s0)
        else:  # MLE
            self.mean = s1 / s0


class KL(Distribution):
    '''Basically a multinomial.'''
    def __init__(self, mean, tau=None, kappa=None, n=100):
        self.mean = mean
        self.tau = tau
        self.kappa = kappa

        if tau is None or kappa is None:
            self.map = False
            self.tau = 0
            self.kappa = 0
        else:
            assert tau is not None and kappa is not None
            self.map = True

        self.n = n  # number of trials for sampling a multinomial

    def __repr__(self):
        return '<KL: mean={}>'.format(repr(self.mean))

    @property
    def dim(self):
        return len(self.mean)

    def distances(self, X):
        return - X.dot(np.log(self.mean))

    def max_likelihood(self, X, weights):
        if weights.dtype == np.bool and not self.map:
            self.mean = X[weights,:].mean(axis=0)
        elif self.map:
            self.mean = (self.tau * self.kappa + np.sum(weights[:,nax] * X, axis=0)) \
                / (self.tau + np.sum(weights))
        else:
            self.mean = np.sum(weights[:,nax] * X, axis=0) / np.sum(weights)

    def online_update(self, x, step):
        self.mean = (1 - step) * self.mean + step * x

    def log_pdf(self, X):
        # log p(x|theta) = sum_j x_j log(theta_j)
        return X.dot(np.log(self.mean))

    def pdf(self, X):
        return np.exp(X.dot(np.log(self.mean)))

    def sample(self, size=1):
        Z = self.mean.sum()
        x = np.random.multinomial(self.n, self.mean/Z, size=int(size))
        return Z * x.astype('float64') / x.sum(1)[:,nax]

    def new_sufficient_statistics_hmm(self, x, cluster_id, K):
        return KLSufficientStatisticsHMM(x, cluster_id, K, self.dim)

    def new_sufficient_statistics_hsmm(self, x, cluster_id, K, D):
        return KLSufficientStatisticsHSMM(x, cluster_id, K, D, self.dim)

    def new_incremental_sufficient_statistics(self, x, phi, cluster_id):
        return KLISufficientStatistics(x, phi, cluster_id, self.dim)

    def online_max_likelihood(self, rho_obs, phi=None, t=None):
        if phi is None:
            s0, s1 = rho_obs.get_statistics()
        else:
            s0, s1 = rho_obs.get_statistics(phi)

        if self.map: # MAP
            assert t is not None
            self.mean = (self.tau * self.kappa + t * s1) / (self.tau + t * s0)
        else:  # MLE
            self.mean = s1 / s0

class ItakuraSaito(Distribution):
    def __init__(self, mean):
        self.mean = mean

    def __repr__(self):
        return '<IS: mean={}>'.format(repr(self.mean))

    def distances(self, X):
        xy = X / self.mean[nax,:]
        return np.sum(xy - np.log(xy) - 1, axis=1)

    def log_pdf(self, X):
        return -self.distances(X)

    def pdf(self, X):
        return np.exp(-self.distances(X))

    def max_likelihood(self, X, weights):
        if weights.dtype == np.bool:
            self.mean = X[weights,:].mean(axis=0)
        else:
            self.mean = np.sum(weights[:,nax] * X, axis=0) / np.sum(weights)

class DurationDistribution(Distribution):
    def __init__(self, D):
        self.D = D
        self.d_frac_vec = None

    def log_pmf(self, X):
        raise NotImplementedError

    def pmf(self, X):
        raise NotImplementedError

    def log_vec(self):
        return self.log_pmf(np.arange(1,self.D+1))

    def vec(self):
        return self.pmf(np.arange(1,self.D+1))

    def d_frac(self):
        if self.d_frac_vec is not None:
            return self.d_frac_vec
        v = self.log_vec()
        D = np.hstack((np.cumsum(v[::-1])[::-1], 0.))
        self.d_frac_vec = np.clip(D[1:] / D[:-1], 0, 1 - 1e-16)
        return self.d_frac_vec

class PoissonDuration(DurationDistribution):
    def __init__(self, lmbda, D):
        super(PoissonDuration, self).__init__(D)
        self.lmbda = lmbda

    def log_pmf(self, X):
        return stats.poisson.logpmf(X, self.lmbda)

    def pmf(self, X):
        return stats.poisson.pmf(X, self.lmbda)

    def __repr__(self):
        return '<Poisson: lambda={}>'.format(self.lmbda)

    def max_likelihood(self, probs):
        assert self.D == len(probs)
        self.lmbda = np.arange(1., self.D + 1).dot(probs)

    def sample(self, size=None):
        return stats.poisson.rvs(self.lmbda, size=size)

    def new_sufficient_statistics_hsmm(self, cluster_id, K, D):
        return PoissonSufficientStatisticsHSMM(cluster_id, K, D)

    def online_max_likelihood(self, rho_dur, phi):
        s0, s1 = rho_dur.get_statistics(phi)
        self.lmbda = s1 / s0

class NegativeBinomial(DurationDistribution):
    def __init__(self, r, p, D):
        super(NegativeBinomial, self).__init__(D)
        self.r = r
        self.p = p

    def log_pmf(self, X):
        return stats.nbinom.logpmf(X, self.r, self.p)

    def pmf(self, X):
        return stats.nbinom.pmf(X, self.r, self.p)

    def __repr__(self):
        return '<NegativeBinomial: r={}, p={}>'.format(self.r, self.p)

    def max_likelihood(self, probs):
        # fixed r, this only estimates p
        k = np.arange(1., self.D + 1).dot(probs)
        self.p = float(self.r) / (self.r + k)

    def sample(self, size=None):
        return stats.nbinom.rvs(self.r, self.p, size=size)

    def new_sufficient_statistics_hsmm(self, cluster_id, K, D):
        return NegativeBinomialSufficientStatisticsHSMM(cluster_id, K, D)

    def online_max_likelihood(self, rho_dur, phi):
        s0, s1 = rho_dur.get_statistics(phi)
        k = s1 / s0
        self.p = float(self.r) / (self.r + k)

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
        self.rho1 = np.zeros((size, self.K))
        self.rho1[:,self.cluster_id] = x

    def online_update(self, x, r, step):
        self.rho0 = (1 - step) * self.rho0.dot(r)
        self.rho0[self.cluster_id] += step
        self.rho1 = (1 - step) * self.rho1.dot(r)
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
        rho0[:,1:] = (1 - step) * self.rho0[:,:-1]
        rho0[self.cluster_id,:] += step
        self.rho0 = rho0

        rho1 = np.zeros(self.rho1.shape)
        rho1[:,:,0] = (1 - step) * np.tensordot(self.rho1, r)
        rho1[:,:,1:] = (1 - step) * self.rho1[:,:,:-1]
        rho1[:,self.cluster_id,:] += step * x[:,nax]
        self.rho1 = rho1

        rho2 = np.zeros(self.rho2.shape)
        rho2[:,:,:,0] = (1 - step) * np.tensordot(self.rho2, r)
        rho2[:,:,:,1:] = (1 - step) * self.rho2[:,:,:,:-1]
        rho2[:,:,self.cluster_id,:] += step * (x[:,nax] * x)[:,:,nax]
        self.rho2 = rho2

    def get_statistics(self, phi):
        return np.tensordot(self.rho0, phi), np.tensordot(self.rho1, phi), \
            np.tensordot(self.rho2, phi)

class KLSufficientStatisticsHSMM(SufficientStatisticsHSMM):
    def __init__(self, x, cluster_id, K, D, size):
        super(KLSufficientStatisticsHSMM, self).__init__(cluster_id, K, D)
        # 1{Z_t = i}
        self.rho0 = np.zeros((self.K, self.D))
        self.rho0[self.cluster_id] = 1.
        # 1{Z_t = i} x_t
        self.rho1 = np.zeros((size, self.K, self.D))
        self.rho1[:,self.cluster_id] = x[:,nax]

    def online_update(self, x, r, step):
        rho0 = np.zeros(self.rho0.shape)
        rho0[:,0] = (1 - step) * np.tensordot(self.rho0, r)
        rho0[:,1:] = (1 - step) * self.rho0[:,:-1]
        rho0[self.cluster_id,:] += step
        self.rho0 = rho0

        rho1 = np.zeros(self.rho1.shape)
        rho1[:,:,0] = (1 - step) * np.tensordot(self.rho1, r)
        rho1[:,:,1:] = (1 - step) * self.rho1[:,:,:-1]
        rho1[:,self.cluster_id,:] += step * x[:,nax]
        self.rho1 = rho1

    def get_statistics(self, phi):
        return np.tensordot(self.rho0, phi), np.tensordot(self.rho1, phi)

class TransitionSufficientStatisticsHSMM(SufficientStatistics):
    def __init__(self, K, D):
        self.K = K
        self.D = D
        # 1{Z_{t-1} = i, Z_t = j, Z_t^D = 1}
        # rho[i, j, k, d]
        self.rho_pairs = np.zeros((K,K,K,D))

    def online_update(self, r, r_marginal, step):
        rho_pairs = np.zeros(self.rho_pairs.shape)
        rho_pairs[:,:,:,0] = (1 - step) * np.tensordot(self.rho_pairs, r) + \
                step * np.eye(self.K)[nax,:,:] * r_marginal[:,:,nax]
        rho_pairs[:,:,:,1:] = (1 - step) * self.rho_pairs[:,:,:,:-1]
        self.rho_pairs = rho_pairs

    def get_statistics(self, phi):
        return np.tensordot(self.rho_pairs, phi)

class DurationSufficientStatistics(SufficientStatisticsHSMM):
    def online_update(self, r, r_marginal, step):
        raise NotImplementedError

class PoissonSufficientStatisticsHSMM(SufficientStatisticsHSMM):
    def __init__(self, cluster_id, K, D):
        super(PoissonSufficientStatisticsHSMM, self).__init__(cluster_id, K, D)
        # 1{Z_{t-1} = i, Z_t^D = 1}
        self.rho0 = np.zeros((self.K, self.D))
        # 1{Z_{t-1} = i, Z_t^D = 1} Z_{t-1}
        self.rho1 = np.zeros((self.K, self.D))

    def online_update(self, r, r_marginal, step):
        rho0 = np.zeros(self.rho0.shape)
        rho0[:,0] = (1 - step) * np.tensordot(self.rho0, r) + \
                step * r_marginal[self.cluster_id]
        rho0[:,1:] = (1 - step) * self.rho0[:,:-1]
        self.rho0 = rho0

        rho1 = np.zeros(self.rho1.shape)
        rho1[:,0] = (1 - step) * np.tensordot(self.rho1, r) + \
                step * np.arange(1., self.D + 1).dot(r[self.cluster_id])
        rho1[:,1:] = (1 - step) * self.rho1[:,:-1]
        self.rho1 = rho1

    def get_statistics(self, phi):
        return np.tensordot(self.rho0, phi), np.tensordot(self.rho1, phi)

class NegativeBinomialSufficientStatisticsHSMM(PoissonSufficientStatisticsHSMM):
    pass  # same as Poisson

# Sufficient statistics for incremental EM
class IncrementalSufficientStatistics(object):
    def __init__(self, cluster_id):
        self.cluster_id = cluster_id

    def online_update(self, x, phi, step):
        raise NotImplementedError

    def get_statistics(self):
        raise NotImplementedError

class GaussianISufficientStatistics(IncrementalSufficientStatistics):
    def __init__(self, x, phi, cluster_id, size):
        super(GaussianISufficientStatistics, self).__init__(cluster_id)
        # 1{Z_t = i}
        self.s0 = phi[self.cluster_id]
        # 1{Z_t = i} x_t
        self.s1 = phi[self.cluster_id] * x
        # 1{Z_t = i} x_t x_t'
        self.s2 = phi[self.cluster_id] * x[:,nax] * x

    def online_update(self, x, phi, step):
        self.s0 = (1 - step) * self.s0 + step * phi[self.cluster_id]
        self.s1 = (1 - step) * self.s1 + step * phi[self.cluster_id] * x
        self.s2 = (1 - step) * self.s2 + step * phi[self.cluster_id] * x[:,nax] * x

    def get_statistics(self):
        return self.s0, self.s1, self.s2

class KLISufficientStatistics(IncrementalSufficientStatistics):
    def __init__(self, x, phi, cluster_id, size):
        super(KLISufficientStatistics, self).__init__(cluster_id)
        # 1{Z_t = i}
        self.s0 = phi[self.cluster_id]
        # 1{Z_t = i} x_t
        self.s1 = phi[self.cluster_id] * x

    def online_update(self, x, phi, step):
        self.s0 = (1 - step) * self.s0 + step * phi[self.cluster_id]
        self.s1 = (1 - step) * self.s1 + step * phi[self.cluster_id] * x

    def get_statistics(self):
        return self.s0, self.s1

class TransitionISufficientStatistics(IncrementalSufficientStatistics):
    def __init__(self, K):
        self.K = K
        # 1{Z_{t-1} = i, Z_t = j}
        # s[i,j]
        self.s = np.zeros((K,K))

    def online_update(self, phi_q, step):
        # phi_q[i,j] = phi_{t-1}[i] q_t[i,j], with q_t[i,j] = q_t(j|i)
        self.s = (1 - step) * self.s + step * phi_q

    def get_statistics(self):
        return self.s
