import distributions
import em
import kmeans
import hmm
import hsmm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import optparse
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

class algos:
    kmeans = 1
    em = 2
    hmm = 3
    hsmm = 4
    map_hmm = 5
    map_hsmm = 6

if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-f', '--file', dest='filename',
            help='STFT input file (default: ravel-fft.mat)', default='ravel-fft.mat')
    parser.add_option('-a', '--algos', dest='algos',
            help='''list of algorithms to run (default:all):\n
1: K-means\n
2: EM\n
3: HMM\n
4: HSMM\n
5: MAP-HMM\n
6: MAP-HSMM''',
                      default='1,2,3,4,5,6')
    parser.add_option('--init', dest='init',
                      help='initialization (em (default)/kmeans)', default='em')
    parser.add_option('-k', '--nclusters', dest='k', type='int',
                      help='number of clusters', default=10)
    options, args = parser.parse_args()

    X = loadmat(options.filename)['X'].T
    X0 = X / np.sum(X,axis=1)[:,nax]
    X = 5 * X0

    # algos to run
    algs = map(int, options.algos.split(','))
    print algs

    # number of clusters
    K = 10

    ass_plots = []

    # K-means
    t = time.time()
    assignments, centroids, dists = \
            kmeans.kmeans_best_of_n(X, K, n_trials=4, dist_cls=distributions.KL)
    print 'K-means: {}s'.format(time.time() - t)


    if options.init == 'em' or algos.em in algs:
        # EM
        iterations = 10
        t = time.time()
        tau_em, obs_distr, pi, em_ll_train, _ = em.em(X, centroids, n_iter=iterations)
        print 'EM: {}s'.format(time.time() - t)


    if options.init == 'em':
        # initialize with EM
        init_pi = pi
        init_obs_distr = obs_distr
    else:
        # initialize with K-means
        init_pi = np.ones(K) / K
        init_obs_distr = centroids

    # run algorithms
    for alg in algs:
        if alg == algos.kmeans:
            ass_plots.append(('K-means', assignments))

        elif alg == algos.em:
            ass_plots.append(('EM', np.argmax(tau_em, axis=1)))

        elif alg == algos.hmm:
            t = time.time()
            tau, A, obs_distr, pi, ll_train, _ = hmm.em_hmm(X, init_pi, init_obs_distr)
            print 'HMM EM: {}s'.format(time.time() - t)

            ass_plots.append(('HMM smoothing', np.argmax(tau, axis=1)))

            seq, _ = hmm.viterbi(X, pi, A, obs_distr)
            ass_plots.append(('HMM viterbi', np.array(seq)))

        elif alg == algos.map_hmm:
            t = time.time()
            seq, obs_distr, energies = hmm.map_em_hmm(X, init_obs_distr)
            print 'HMM MAP-EM: {}s'.format(time.time() - t)
            ass_plots.append(('HMM MAP-EM', np.array(seq)))

        elif alg == algos.hsmm:
            dur_distr = [distributions.NegativeBinomial(5, 0.95, D=200) for _ in range(K)]
            t = time.time()
            tau, A, obs_distr, dur_distr, pi, ll_train, _ = \
                    hsmm.em_hsmm(X, init_pi, init_obs_distr, dur_distr)
            print 'HSMM EM: {}s'.format(time.time() - t)

            ass_plots.append(('HSMM smoothing', np.argmax(tau, axis=1)))

            seq, _ = hsmm.viterbi(X, pi, A, obs_distr, dur_distr)
            ass_plots.append(('HSMM viterbi', np.array(seq)))

        elif alg == algos.map_hsmm:
            t = time.time()
            seq, obs_distr, dur_distr, energies = hsmm.map_em_hsmm(X, init_obs_distr, dur_distr)
            print 'HSMM MAP-EM: {}s'.format(time.time() - t)
            ass_plots.append(('HSMM MAP-EM', np.array(seq)))

    plt.figure()
    plot_segmentation(X, ass_plots)
