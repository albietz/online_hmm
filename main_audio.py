import distributions
import em
import kmeans
import hmm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys

from numpy import newaxis as nax
from scipy.io import loadmat

def plot_segmentation(X, assignments):
    gs = gridspec.GridSpec(4, 4)
    plt.subplot(gs[0,:])
    x,y = np.meshgrid(np.arange(len(assignments)),(0,1))

    c = assignments[nax,:].astype(np.float) / np.max(assignments)

    plt.pcolor(x,y,c,vmin=0,vmax=1)
    plt.ylim(0,1)
    plt.xlim(0, len(assignments))
    plt.yticks([])
    # plt.bar(np.arange(len(assignments)), assignments+1, width=1.)

    plt.subplot(gs[1:,:])
    plt.imshow(np.log(X.T), aspect='auto')

if __name__ == '__main__':
    # if not sys.argv[1:]:
    #     print 'please specify the .mat file with the STFT data'
    #     sys.exit(0)

    X = loadmat('ravel-fft.mat')['X'].T
    X0 = X / np.sum(X,axis=1)[:,nax]
    X = 5 * X0

    # number of clusters
    K = 10

    # K-means
    assignments, centroids, dists = \
            kmeans.kmeans_best_of_n(X, K, n_trials=4, dist_cls=distributions.KL)

    fig = plt.figure(1)
    plot_segmentation(X, assignments)
    fig.suptitle('K-means')

    # EM
    iterations = 10
    tau, obs_distr, pi, em_ll_train, _ = em.em(X, centroids, n_iter=iterations)

    fig = plt.figure(2)
    plot_segmentation(X, np.argmax(tau, axis=1))
    fig.suptitle('EM')
    
    # HMM
    tau, A, obs_distr, pi, ll_train, _ = hmm.em_hmm(X, pi, obs_distr)

    fig = plt.figure(3)
    plot_segmentation(X, np.argmax(tau, axis=1))
    fig.suptitle('HMM smoothing')

    seq = hmm.viterbi(X, pi, A, obs_distr)
    fig = plt.figure(4)
    plot_segmentation(X, np.argmax(tau, axis=1))
    fig.suptitle('HMM viterbi')
