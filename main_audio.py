import distributions
import em
import evaluation
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

def plot_segmentation(X, assignments, start=0, end=None, freq_start=0, freq_end=None, gt=None, K=None):
    if end is None:
        end = X.shape[0]
    gs = gridspec.GridSpec(2 + len(assignments), 4)
    for i, (title, ass) in enumerate(assignments):
        plt.subplot(gs[i,:])
        x,y = np.meshgrid(np.arange(end-start),(0,1))

        c = (ass[nax,start:end].astype(np.float) - np.min(ass)) \
                / (max(1,np.max(ass)) - np.min(ass))

        plt.pcolor(x,y,c,vmin=0,vmax=1)
        plt.ylim(0,1)
        plt.xlim(0, end-start)
        plt.yticks([])
        plt.xticks([])
        plt.title(title)
        # plt.bar(np.arange(len(ass)), ass+1, width=1.)

    plt.subplot(gs[-2:,:])
    if freq_end is None:
        freq_end = X.shape[1]
    plt.imshow(np.log(X[start:end, freq_start:freq_end].T),
        extent=(start, end, freq_end*44100./4096, freq_start*44100./4096),
        aspect='auto')
    plt.title('Spectrogram')
    plt.grid(False)
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
    incremental_em_hmm = 11
    incremental_em_hsmm = 12
    incremental_em_hmm_add = 13

if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-f', '--file', dest='filename',
            help='STFT matlab input file (default: ravel-fft.mat)', default='ravel-fft.mat')
    parser.add_option('-a', '--algos', dest='algos',
            help='''list of algorithms to run (default:3,4):\n
1: K-means\n
2: EM\n
3: HMM\n
4: HSMM\n
5: MAP-HMM (non-probabilistic)\n
6: MAP-HSMM (non-probabilistic)\n
7: online non-probabilistic HMM\n
8: online non-probabilistic HSMM\n
9: online EM HMM (Cappe)\n
10: online EM HSMM\n
11: incremental EM HMM\n
12: incremental EM HSMM\n
13: incremental EM HMM with new states''',
                      default='3,4')
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
    parser.add_option('-n', dest='n', default=5, type='float',
                      help='spectrum normalization constant')
    parser.add_option('--start', dest='start', default=0, type='int')
    parser.add_option('--end', dest='end', default=None, type='int')
    parser.add_option('--ground_truth', dest='ground_truth', default=None,
                      help='ground truth file')
    options, args = parser.parse_args()

    matfile = loadmat(options.filename)

    ground_truth = None
    if 'label' in matfile:
        ground_truth = matfile['label'].flatten() - 1
    elif options.ground_truth:
        ground_truth = np.load(options.ground_truth)

    X = matfile['X'].T
    X0 = X / np.sum(X,axis=1)[:,nax]
    X = options.n * X0
    X = X[options.start:options.end]

    if ground_truth is not None:
        ground_truth = ground_truth[options.start:options.end]

    if options.repeat > 1:
        X = np.vstack((X,) * options.repeat)
        if ground_truth is not None:
            ground_truth = np.hstack((ground_truth,) * options.repeat)

    # algos to run
    algs = map(int, options.algos.split(','))

    # number of clusters
    K = options.k

    ass_plots = []
    seqs = {}
    results = {}  # data/results obtained, indexed by algorithm

    # K-means
    if options.init == 'kmeans' or algos.kmeans in algs:
        t = time.time()
        assignments, centroids, dists = \
                kmeans.kmeans_best_of_n(X, K, n_trials=4, dist_cls=distributions.KL)
        print 'K-means: {}s'.format(time.time() - t)
        results[algos.kmeans] = {
            'seq': assignments,
            'centroids': centroids,
        }
        seqs[algos.kmeans] = assignments

    # EM
    if options.init == 'em' or algos.em in algs:
        iterations = 10
        t = time.time()
        tau_em, obs_distr, pi, em_ll_train, _ = em.em(X, centroids, n_iter=options.n_iter)
        print 'EM: {}s'.format(time.time() - t)
        results[algos.em] = {
            'seq': np.argmax(tau_em, axis=1),
            'obs_distr': obs_distr,
            'll_train': em_ll_train,
            'tau': tau_em,
        }
        seqs[algos.em] = np.argmax(tau_em, axis=1)


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
        # init_obs_distr = [distributions.KL(X[k], tau=5., kappa=options.n * np.ones(X.shape[1])/X.shape[1]) for k in range(K)]
    elif options.init == 'rand':
        init_pi = np.ones(K) / K
        norm = X[0].sum()
        init_obs_distr = [distributions.KL(norm*np.random.dirichlet(np.ones(X.shape[1])))
                for k in range(K)]
    elif options.init == 'randunif':
        init_pi = np.ones(K) / K
        p = X.shape[1]
        def gen_perturbed_unif():
            m = np.ones(p) + 0.1 * np.random.rand(p)
            return distributions.KL(options.n * m / m.sum())
        init_obs_distr = [gen_perturbed_unif() for k in range(K)]
    elif options.init == 'prev':
        print 'using existing variables, must be in IPython interactive mode! (%run -i)'
    else:
        print '{} initialization is not available'.format(options.init)
        sys.exit(0)


    # run algorithms
    for alg in algs:
        if alg == algos.kmeans:
            ass_plots.append(('K-means', results[alg]['seq']))

        elif alg == algos.em:
            ass_plots.append(('EM', results[alg]['seq']))

        elif alg == algos.hmm:
            t = time.time()
            tau, A, obs_distr, pi, ll_train, _ = hmm.em_hmm(X, init_pi, init_obs_distr, n_iter=options.n_iter)
            print 'HMM EM: {}s, final loglikelihood: {}'.format(time.time() - t, ll_train[-1])

            seq_smoothing = np.argmax(tau, axis=1)
            ass_plots.append(('HMM smoothing', seq_smoothing))

            seq_viterbi, _ = hmm.viterbi(X, pi, A, obs_distr)
            ass_plots.append(('HMM viterbi', seq_viterbi))
            results[alg] = {
                'tau': tau,
                'A': A,
                'obs_distr': obs_distr,
                'pi': pi,
                'll_train': ll_train,
                'seq_smoothing': seq_smoothing,
                'seq_viterbi': seq_viterbi,
            }
            seqs[alg] = (seq_smoothing, seq_viterbi)

        elif alg == algos.map_hmm:
            t = time.time()
            seq, obs_distr, energies = hmm.map_em_hmm(X, init_obs_distr)
            print 'HMM MAP-EM: {}s, final energy: {}'.format(time.time() - t, energies[-1])
            results[alg] = {
                'obs_distr': obs_distr,
                'energies': energies,
                'seq': seq,
            }
            ass_plots.append(('HMM MAP-EM', seq))
            seqs[alg] = seq

        elif alg == algos.hsmm:
            # init_dur_distr = [distributions.PoissonDuration(60, D=200) for _ in range(K)]
            # init_dur_distr = [distributions.NegativeBinomial(100, 0.2, D=400) for _ in range(K)]
            init_dur_distr = [distributions.NegativeBinomial(5, 5. / (5 + 20), D=100) for _ in range(K)]
            # init_dur_distr = [distributions.NegativeBinomial(100, 0.05, D=200) for _ in range(K)]
            # init_dur_distr = [distributions.NegativeBinomial(5, 0.75, D=200) for _ in range(K)]
            # init_dur_distr = [distributions.NegativeBinomial(5, 0.95, D=200) for _ in range(K)]
            t = time.time()
            tau, A, obs_distr, dur_distr, pi, ll_train, _ = \
                    hsmm.em_hsmm(X, init_pi, init_obs_distr, init_dur_distr, n_iter=options.n_iter, fit_durations=False)
            print 'HSMM EM: {}s, final loglikelihood: {}'.format(time.time() - t, ll_train[-1])

            seq_smoothing = np.argmax(tau, axis=1)
            ass_plots.append(('HSMM smoothing', seq_smoothing))

            seq_viterbi, _ = hsmm.viterbi(X, pi, A, obs_distr, dur_distr)
            ass_plots.append(('HSMM viterbi', seq_viterbi))
            results[alg] = {
                'tau': tau,
                'A': A,
                'obs_distr': obs_distr,
                'dur_distr': dur_distr,
                'pi': pi,
                'll_train': ll_train,
                'seq_smoothing': seq_smoothing,
                'seq_viterbi': seq_viterbi,
            }
            seqs[alg] = (seq_smoothing, seq_viterbi)

        elif alg == algos.map_hsmm:
            t = time.time()
            seq, obs_distr, dur_distr, energies = hsmm.map_em_hsmm(X, init_obs_distr, init_dur_distr)
            print 'HSMM MAP-EM: {}s, final energy: {}'.format(time.time() - t, energies[-1])
            seqs[alg] = np.array(seq)
            ass_plots.append(('HSMM MAP-EM', seqs[alg]))

        elif alg == algos.online_opt_hmm:
            t = time.time()
            seq, obs_distr, cost = hmm.online_opt_hmm(X, 1.5, 250.0, init_obs_distr, dist_cls=distributions.KL)
            print 'HMM online opt: {}s, cost: {}'.format(time.time() - t, cost)

            seqs[alg] = seq
            ass_plots.append(('HMM online opt', seq))

        elif alg == algos.online_opt_hsmm:
            lcost = 1.
            t = time.time()
            seq, obs_distr, cost = hsmm.online_opt_hsmm(X, 2.0, 250.0, init_obs_distr, dist_cls=distributions.KL)
            print 'HMM online opt: {}s'.format(time.time() - t)

            seqs[alg] = seq
            ass_plots.append(('HMM online opt', seq))

        elif alg == algos.online_em_hmm:
            step = lambda t: 1. / (t ** 0.6)
            t = time.time()
            seq, tau, A, obs_distr, _, _ = hmm.online_em_hmm(X, init_pi, init_obs_distr, t_min=100, step=step, m_step_delta=10)
            ll_final = hmm.log_likelihood(*hmm.alpha_beta(X, init_pi, A, obs_distr))
            print 'HMM online EM: {}s, final loglikelihood: {}'.format(time.time() - t, ll_final)

            seq_mpm = hmm.mpm_sequence(X, init_pi, A, obs_distr)
            seq_viterbi, _ = hmm.viterbi(X, init_pi, A, obs_distr)

            seqs[alg] = (seq, seq_mpm, seq_viterbi)
            ass_plots.append(('HMM online EM filter', seq))
            ass_plots.append(('HMM online EM smoothing', seq_mpm))
            ass_plots.append(('HMM online EM viterbi', seq_viterbi))
            results[alg] = {
                'tau': tau,
                'A': A,
                'obs_distr': obs_distr,
                'pi': init_pi,
                'll_final': ll_final,
                'seq_online': seq,
                'seq_smoothing': seq_mpm,
                'seq_viterbi': seq_viterbi,
            }

        elif alg == algos.online_em_hsmm:
            step = lambda t: 1. / (t ** 0.6)
            # init_dur_distr = [distributions.NegativeBinomial(5, 0.95, D=200) for _ in range(K)]
            # init_dur_distr = [distributions.NegativeBinomial(30, 30. / (30 + 20), D=60) for _ in range(K)]
            init_dur_distr = [distributions.NegativeBinomial(5, 5. / (5 + 20), D=100) for _ in range(K)]
            # init_dur_distr = [distributions.PoissonDuration(40, D=200) for _ in range(K)]
            t = time.time()
            seq, A, obs_distr, dur_distr = hsmm.online_em_hsmm(X, init_pi, init_obs_distr, init_dur_distr, t_min=100, step=step)
            print 'HSMM online EM: {}s'.format(time.time() - t)

            seq_mpm = hsmm.mpm_sequence(X, init_pi, A, obs_distr, dur_distr)
            seq_viterbi, _ = hsmm.viterbi(X, init_pi, A, obs_distr, dur_distr)
            seqs[alg] = (seq, seq_mpm, seq_viterbi)
            ass_plots.append(('HSMM online EM filter', seq))
            ass_plots.append(('HSMM online EM smoothing', seq_mpm))
            ass_plots.append(('HSMM online EM viterbi', seq_viterbi))

            _, lalphastar, _, lbetastar = hsmm.alpha_beta(X, init_pi, A, obs_distr, dur_distr)
            ll_final = hsmm.log_likelihood(lalphastar, lbetastar)
            results[alg] = {
                'tau': tau,
                'A': A,
                'obs_distr': obs_distr,
                'dur_distr': dur_distr,
                'pi': init_pi,
                'll_final': ll_final,
                'seq_online': seq,
                'seq_smoothing': seq_mpm,
                'seq_viterbi': seq_viterbi,
            }

        elif alg == algos.incremental_em_hmm:
            step = lambda t: 1. / (t ** 0.6)
            # step = lambda t: 1. / t
            t = time.time()
            seq, tau, A, obs_distr, _, _ = hmm.incremental_em_hmm(X, init_pi, init_obs_distr, t_min=100, step=step, m_step_delta=10)
            ll_final = hmm.log_likelihood(*hmm.alpha_beta(X, init_pi, A, obs_distr))
            print 'HMM incremental EM: {}s, final loglikelihood: {}'.format(time.time() - t, ll_final)

            seq_mpm = hmm.mpm_sequence(X, init_pi, A, obs_distr)
            seq_viterbi, _ = hmm.viterbi(X, init_pi, A, obs_distr)

            seqs[alg] = (seq, seq_mpm, seq_viterbi)
            ass_plots.append(('HMM incremental EM filter', seq))
            ass_plots.append(('HMM incremental EM smoothing', seq_mpm))
            ass_plots.append(('HMM incremental EM viterbi', seq_viterbi))

            results[alg] = {
                'tau': tau,
                'A': A,
                'obs_distr': obs_distr,
                'pi': init_pi,
                'll_final': ll_final,
                'seq_online': seq,
                'seq_smoothing': seq_mpm,
                'seq_viterbi': seq_viterbi,
            }

        elif alg == algos.incremental_em_hsmm:
            step = lambda t: 1. / (t ** 0.6)
            # step = lambda t: 1. / t
            # init_dur_distr = [distributions.NegativeBinomial(5, 0.95, D=200) for _ in range(K)]
            # init_dur_distr = [distributions.NegativeBinomial(30, 30. / (30 + 20), D=200) for _ in range(K)]
            init_dur_distr = [distributions.NegativeBinomial(5, 5. / (5 + 20), D=200) for _ in range(K)]
            # init_dur_distr = [distributions.PoissonDuration(40, D=200) for _ in range(K)]

            t = time.time()
            seq, A, obs_distr, dur_distr = hsmm.incremental_em_hsmm(X, init_pi, init_obs_distr, init_dur_distr, t_min=100, step=step)
            _, lalphastar, _, lbetastar = hsmm.alpha_beta(X, init_pi, A, obs_distr, dur_distr)
            ll_final = hsmm.log_likelihood(lalphastar, lbetastar)
            print 'HSMM incremental EM: {}s, final loglikelihood: {}'.format(time.time() - t, ll_final)

            seq_mpm = hsmm.mpm_sequence(X, init_pi, A, obs_distr, dur_distr)
            seq_viterbi, _ = hsmm.viterbi(X, init_pi, A, obs_distr, dur_distr)
            seqs[alg] = (seq, seq_mpm, seq_viterbi)
            ass_plots.append(('HSMM incremental EM filter', seq))
            ass_plots.append(('HSMM incremental EM smoothing', seq_mpm))
            ass_plots.append(('HSMM incremental EM viterbi', seq_viterbi))

            results[alg] = {
                'tau': tau,
                'A': A,
                'obs_distr': obs_distr,
                'dur_distr': dur_distr,
                'pi': init_pi,
                'll_final': ll_final,
                'seq_online': seq,
                'seq_smoothing': seq_mpm,
                'seq_viterbi': seq_viterbi,
            }

        elif alg == algos.incremental_em_hmm_add:
            step = lambda t: 1. / (t ** 0.6)
            # step = lambda t: 1. / t
            t = time.time()
            params = {'tau': 2, 'kappa': options.n * np.ones(X.shape[1])/X.shape[1]}
            seq, A, obs_distr, _, _ = hmm.incremental_em_hmm_add(X, lmbda=1.2, Kmax=20, dist_cls=distributions.KL, dist_params=params, step=step)
            print 'HMM incremental EM (add new states): {}s'.format(time.time() - t)

            seqs[alg] = seq
            ass_plots.append(('HMM incremental EM add', seq))

    prfs = {}
    if ground_truth is not None:
        ass_plots.append(('ground truth', ground_truth))

        for alg in algs:
            if isinstance(seqs[alg], tuple):
                prfs[alg] = tuple(evaluation.evaluate(ground_truth, s, K) for s in seqs[alg])
            else:
                prfs[alg] = evaluation.evaluate(ground_truth, seqs[alg], K)

    plt.figure()
    plot_segmentation(X, ass_plots, gt=ground_truth)
