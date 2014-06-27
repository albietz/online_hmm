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
        plt.xticks([])
        plt.title(title)
        # plt.bar(np.arange(len(ass)), ass+1, width=1.)

    plt.subplot(gs[-2:,:])
    plt.imshow(np.log(X.T), aspect='auto')
    plt.title('Spectrogram')
    plt.gcf().set_tight_layout(True)

class algos:
    kmeans = 1
    em = 2
    hmm = 3
    hsmm = 4
    map_hmm = 5
    map_hsmm = 6
    online_opt_hmm = 7
    online_opt_hsmm = 8
    online_em_hmm = 9
    online_em_hsmm = 10

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
                      help='initialization (kmeans(default)/em/randex)', default='kmeans')
    parser.add_option('--kmeans_trials', dest='kmeans_trials', type='int',
                      default=4, help='number of kmeans trials')
    parser.add_option('-k', '--nclusters', dest='k', type='int',
                      help='number of clusters', default=10)
    parser.add_option('-r', '--repeat', dest='repeat', default=1, type='int',
                      help='repeat input')
    parser.add_option('--iter', dest='n_iter', default=10, type='int',
                      help='EM iterations')
    options, args = parser.parse_args()

    X = loadmat(options.filename)['X'].T
    X0 = X / np.sum(X,axis=1)[:,nax]
    X = 5 * X0

    if options.repeat > 1:
        X = np.vstack((X,) * options.repeat)

    # algos to run
    algs = map(int, options.algos.split(','))

    # number of clusters
    K = options.k

    ass_plots = []

    # K-means
    if options.init == 'kmeans' or algos.kmeans in algs:
        t = time.time()
        assignments, centroids, dists = \
                kmeans.kmeans_best_of_n(X, K, n_trials=4, dist_cls=distributions.KL)
        print 'K-means: {}s'.format(time.time() - t)

    # EM
    if options.init == 'em' or algos.em in algs:
        iterations = 10
        t = time.time()
        tau_em, obs_distr, pi, em_ll_train, _ = em.em(X, centroids, n_iter=iterations)
        print 'EM: {}s'.format(time.time() - t)


    # initialization
    if options.init == 'em':
        # initialize with EM
        init_pi = pi
        init_obs_distr = obs_distr
    elif options.init == 'kmeans':
        # initialize with K-means
        init_pi = np.ones(K) / K
        init_obs_distr = centroids
    elif options.init == 'randex':
        init_pi = np.ones(K) / K
        perm = np.random.permutation(X.shape[0])
        init_obs_distr = [distributions.KL(X[perm[k]]) for k in range(K)]
    elif options.init == 'firstex':
        init_pi = np.ones(K) / K
        init_obs_distr = [distributions.KL(X[k]) for k in range(K)]
    elif options.init == 'rand':
        init_pi = np.ones(K) / K
        norm = X[0].sum()
        init_obs_distr = [distributions.KL(norm*np.random.dirichlet(np.ones(X.shape[1])))
                for k in range(K)]
    elif options.init == 'prev':
        print 'using existing variables, must be in IPython interactive mode! (%run -i)'
    else:
        print '{} initialization is not available'.format(options.init)
        sys.exit(0)

    # run algorithms
    for alg in algs:
        if alg == algos.kmeans:
            ass_plots.append(('K-means', assignments))

        elif alg == algos.em:
            ass_plots.append(('EM', np.argmax(tau_em, axis=1)))

        elif alg == algos.hmm:
            t = time.time()
            tau, A, obs_distr, pi, ll_train, _ = hmm.em_hmm(X, init_pi, init_obs_distr, n_iter=options.n_iter)
            print 'HMM EM: {}s, final loglikelihood: {}'.format(time.time() - t, ll_train[-1])

            ass_plots.append(('HMM smoothing', np.argmax(tau, axis=1)))

            seq, _ = hmm.viterbi(X, pi, A, obs_distr)
            ass_plots.append(('HMM viterbi', np.array(seq)))

        elif alg == algos.map_hmm:
            t = time.time()
            seq, obs_distr, energies = hmm.map_em_hmm(X, init_obs_distr)
            print 'HMM MAP-EM: {}s, final energy: {}'.format(time.time() - t, energies[-1])
            ass_plots.append(('HMM MAP-EM', np.array(seq)))

        elif alg == algos.hsmm:
            # init_dur_distr = [distributions.PoissonDuration(60, D=200) for _ in range(K)]
            init_dur_distr = [distributions.NegativeBinomial(5, 0.95, D=200) for _ in range(K)]
            # init_dur_distr = [distributions.NegativeBinomial(100, 0.05, D=200) for _ in range(K)]
            t = time.time()
            tau, A, obs_distr, dur_distr, pi, ll_train, _ = \
                    hsmm.em_hsmm(X, init_pi, init_obs_distr, init_dur_distr, n_iter=options.n_iter, fit_durations=False)
            print 'HSMM EM: {}s, final loglikelihood: {}'.format(time.time() - t, ll_train[-1])

            ass_plots.append(('HSMM smoothing', np.argmax(tau, axis=1)))

            seq, _ = hsmm.viterbi(X, pi, A, obs_distr, dur_distr)
            ass_plots.append(('HSMM viterbi', np.array(seq)))

        elif alg == algos.map_hsmm:
            t = time.time()
            seq, obs_distr, dur_distr, energies = hsmm.map_em_hsmm(X, init_obs_distr, init_dur_distr)
            print 'HSMM MAP-EM: {}s, final energy: {}'.format(time.time() - t, energies[-1])
            ass_plots.append(('HSMM MAP-EM', np.array(seq)))

        elif alg == algos.online_opt_hmm:
            t = time.time()
            seq, obs_distr, cost = hmm.online_opt_hmm(X, 1.5, 250.0, init_obs_distr, dist_cls=distributions.KL)
            print 'HMM online opt: {}s, cost: {}'.format(time.time() - t, cost)

            ass_plots.append(('HMM online opt', seq))

        elif alg == algos.online_opt_hsmm:
            lcost = 1.
            t = time.time()
            seq, obs_distr, cost = hsmm.online_opt_hsmm(X, 2.0, 250.0, init_obs_distr, dist_cls=distributions.KL)
            print 'HMM online opt: {}s'.format(time.time() - t)

            ass_plots.append(('HMM online opt', seq))

        elif alg == algos.online_em_hmm:
            step = lambda t: 1. / (t ** 0.6)
            t = time.time()
            seq, tau, A, obs_distr = hmm.online_em_hmm(X, init_pi, init_obs_distr, t_min=100, step=step)
            print 'HMM online EM: {}s'.format(time.time() - t)

            ass_plots.append(('HMM online EM', seq))

        elif alg == algos.online_em_hsmm:
            step = lambda t: 1. / (t ** 0.6)
            init_dur_distr = [distributions.NegativeBinomial(5, 0.95, D=200) for _ in range(K)]
            # init_dur_distr = [distributions.PoissonDuration(150, D=200) for _ in range(K)]
            t = time.time()
            seq, A, obs_distr, dur_distr = hsmm.online_em_hsmm(X, init_pi, init_obs_distr, init_dur_distr, t_min=100, step=step)
            print 'HSMM online EM: {}s'.format(time.time() - t)

            ass_plots.append(('HSMM online EM', seq))

    plt.figure()
    plot_segmentation(X, ass_plots)
