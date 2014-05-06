import copy
import distributions
import kmeans
import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.mlab import bivariate_normal
from numpy import newaxis as nax
from numpy.linalg import det, inv

def log_likelihood(X, obs_distr, pi):
    N = X.shape[0]
    K = len(obs_distr)
    pdfs = np.zeros((N,K))
    for j in range(K):
        pdfs[:,j] = obs_distr[j].pdf(X)

    # ll = sum_i log(sum_j pi_j p(x_i|theta_j))
    ll = sum(np.log(np.dot(pi, pdfs[i,:])) for i in range(N))
    return ll

def em(X, init_obs_distr, assignments=None, n_iter=10, Xtest=None):
    N = X.shape[0]
    K = len(init_obs_distr)

    if assignments is not None:
        pi = np.array([np.sum(assignments == j) for j in range(K)], dtype=np.float)
        pi = pi / np.sum(pi)
    else:
        pi = np.ones(K) / K
    obs_distr = copy.deepcopy(init_obs_distr)
    tau = np.zeros((N, K))  # tau[i,j] = p(z_i = j | x_i)

    ll_train = []
    ll_test = []

    for i in range(n_iter):
        # E-step
        for j in range(K):
            tau[:,j] = pi[j] * obs_distr[j].pdf(X)

        # normalize each line
        tau = tau / np.sum(tau, axis=1)[:,nax]

        # M-step
        pi = np.sum(tau, axis=0) / N

        for j in range(K):
            obs_distr[j].max_likelihood(X, tau[:,j])

        ll_train.append(log_likelihood(X, obs_distr, pi))
        if Xtest is not None:
            ll_test.append(log_likelihood(Xtest, obs_distr, pi))

    return tau, obs_distr, pi, ll_train, ll_test

def plot_em(X, tau, obs_distr, contours=False):
    means = np.vstack(d.mean for d in obs_distr)
    K = means.shape[0]
    plt.figure()
    plt.scatter(X[:,0], X[:,1], c=np.argmax(tau, axis=1))
    plt.scatter(means[:,0], means[:,1], color='green', s=100)
    
    if contours:
        sigmas = [d.cov for d in obs_distr]
        for j in range(K):
            x, y = np.arange(-10., 10., 0.04), np.arange(-15., 15., 0.04)
            xx, yy = np.meshgrid(x, y)
            sx = np.sqrt(sigmas[j][0,0])
            sy = np.sqrt(sigmas[j][1,1])
            sxy = sigmas[j][1,0]
            z = bivariate_normal(xx, yy, sx, sy, means[j,0], means[j,1], sxy)
            cs = plt.contour(xx, yy, z, [0.01])

if __name__ == '__main__':
    X = np.loadtxt('EMGaussian.data')
    Xtest = np.loadtxt('EMGaussian.test')
    K = 4
    iterations = 40

    assignments, centers, _ = kmeans.kmeans_best_of_n(X, K, n_trials=5)
    for k in range(K):
        centers[k].sigma2 = 1.

    # Isotropic
    tau, obs_distr, pi, ll_train_iso, ll_test_iso = \
            em(X, centers, assignments, n_iter=iterations, Xtest=Xtest)
    plot_em(X, tau, obs_distr, contours=True)
    plt.title('EM with covariance matrices proportional to identity')

    # General
    new_centers = [distributions.Gaussian(c.mean, c.sigma2*np.eye(2)) \
                for c in centers]
    tau, obs_distr, pi, ll_train_gen, ll_test_gen = \
            em(X, new_centers, assignments, n_iter=iterations, Xtest=Xtest)
    plot_em(X, tau, obs_distr, contours=True)
    plt.title('EM with general covariance matrices')

    # log-likelihood plot
    plt.figure()
    plt.plot(ll_train_iso, label='isotropic, training')
    plt.plot(ll_test_iso, label='isotropic, test')
    plt.plot(ll_train_gen, label='general, training')
    plt.plot(ll_test_gen, label='general, test')
    plt.xlabel('iterations')
    plt.ylabel('log-likelihood')
    plt.title('Comparison of learning curves')
    plt.legend()
