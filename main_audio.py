import distributions
import em
import kmeans
import hmm
import hsmm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import time

from numpy import newaxis as nax
from scipy.io import loadmat

def plot_segmentation(X, assignments):
    gs = gridspec.GridSpec(2 + len(assignments), 4)
    for i, (title, ass) in enumerate(assignments):
        plt.subplot(gs[i,:])
        x,y = np.meshgrid(np.arange(len(ass)),(0,1))

        c = ass[nax,:].astype(np.float) / np.max(ass)

        plt.pcolor(x,y,c,vmin=0,vmax=1)
        plt.ylim(0,1)
        plt.xlim(0, len(ass))
        plt.yticks([])
        plt.title(title)
        # plt.bar(np.arange(len(ass)), ass+1, width=1.)

    plt.subplot(gs[-2:,:])
    plt.imshow(np.log(X.T), aspect='auto')
    plt.title('Spectrogram')

if __name__ == '__main__':
    if not sys.argv[1:]:
        print 'no SFTF file provided, using ravel-fft.mat'
        filename = 'ravel-fft.mat'
    else:
        filename = sys.argv[1]

    X = loadmat(filename)['X'].T
    X0 = X / np.sum(X,axis=1)[:,nax]
    X = 5 * X0

    # number of clusters
    K = 10

    ass_plots = []

    # K-means
    t = time.time()
    assignments, centroids, dists = \
            kmeans.kmeans_best_of_n(X, K, n_trials=4, dist_cls=distributions.KL)
    print 'K-means: {}s'.format(time.time() - t)

    ass_plots.append(('K-means', assignments))

    # EM
    iterations = 10
    t = time.time()
    tau, obs_distr, pi, em_ll_train, _ = em.em(X, centroids, n_iter=iterations)
    print 'EM: {}s'.format(time.time() - t)

    ass_plots.append(('EM', np.argmax(tau, axis=1)))

    # initialize with K-means
    # init_pi = np.ones(K) / K
    # init_obs_distr = centroids

    # initialize with EM
    init_pi = pi
    init_obs_distr = obs_distr

    # HMM
    t = time.time()
    tau, A, obs_distr, pi, ll_train, _ = hmm.em_hmm(X, init_pi, init_obs_distr)
    print 'HMM EM: {}s'.format(time.time() - t)

    ass_plots.append(('HMM smoothing', np.argmax(tau, axis=1)))

    seq, _ = hmm.viterbi(X, pi, A, obs_distr)
    ass_plots.append(('HMM viterbi', np.array(seq)))

    t = time.time()
    seq, obs_distr, energies = hmm.map_em_hmm(X, init_obs_distr)
    print 'HMM MAP-EM: {}s'.format(time.time() - t)
    ass_plots.append(('HMM MAP-EM', np.array(seq)))

    # HSMM
    dur_distr = [distributions.NegativeBinomial(5, 0.95, D=200) for _ in range(K)]
    t = time.time()
    tau, A, obs_distr, dur_distr, pi, ll_train, _ = \
            hsmm.em_hsmm(X, init_pi, init_obs_distr, dur_distr)
    print 'HSMM EM: {}s'.format(time.time() - t)

    ass_plots.append(('HSMM smoothing', np.argmax(tau, axis=1)))

    seq, _ = hsmm.viterbi(X, pi, A, obs_distr, dur_distr)
    ass_plots.append(('HSMM viterbi', np.array(seq)))

    t = time.time()
    seq, obs_distr, dur_distr, energies = hsmm.map_em_hsmm(X, init_obs_distr, dur_distr)
    print 'HSMM MAP-EM: {}s'.format(time.time() - t)
    ass_plots.append(('HSMM MAP-EM', np.array(seq)))

    plt.figure()
    plot_segmentation(X, ass_plots)
