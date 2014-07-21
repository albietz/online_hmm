import collections
import distributions
import em
import evaluation
import kmeans
import hmm
import hsmm
import matplotlib.pyplot as plt
import numpy as np
import sys

from numpy import newaxis as nax
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix

from IPython.core.debugger import Tracer

def cluster(X, K, divergence, debug=False):
    if divergence == 'KL':
        dist_cls = distributions.KL
        if np.any(X <= 0):  # for MFCCs...
            X = X - X.min() + 1e-8
        X = 5. * X / X.sum(1)[:,nax]
    elif divergence == 'IS':
        dist_cls = distributions.ItakuraSaito
        if np.any(X <= 0):  # for MFCCs...
            X = X - X.min() + 1e-8
        # X = X / X.sum(1)[:,nax]
    elif divergence == 'EU':
        dist_cls = distributions.SquareDistance
    else:
        print 'Wrong divergence'
        sys.exit(0)

    assignments, centroids, _ = kmeans.kmeans_best_of_n(X, K, n_trials=10,
            dist_cls=dist_cls, debug=debug)
    init_pi = np.ones(K) / K
    init_obs_distr = centroids

    tau_em, obs_distr, pi, em_ll_train, _ = em.em(X, centroids, n_iter=10)
    # tau_hmm, A, obs_distr, pi, ll_train, _ = hmm.em_hmm(X, init_pi, init_obs_distr, n_iter=10)
    # seq_hmm, _ = hmm.viterbi(X, pi, A, obs_distr)
    Tracer()()

    return {'kmeans': assignments,
            'EM': np.argmax(tau_em, axis=1),
            # 'hmm_smoothing': np.argmax(tau_hmm, axis=1),
            # 'hmm_viterbi': seq_hmm,
           }

if __name__ == '__main__':
    db_file = sys.argv[1]
    db = loadmat(db_file, squeeze_me=True)['store']['s'].item()

    PRF = collections.namedtuple('PRF', 'p r f pcw rcw fcw')
    divergences = ['KL', 'IS', 'EU']
    seqs = {}
    prfs = collections.defaultdict(lambda: collections.defaultdict(list))
    for i in range(db.shape[0]):
        print db[i]['name'].item()
        K = len(db[i]['classNames'].item())
        ground_truth = db[i]['trueLabel'].item() - 1  # for 0-indexing
        X = db[i]['feature'].item()

        for div in divergences:
            seqs[div] = cluster(X, K, div, debug=False)
            for algo in seqs[div]:
                prfs[algo][div].append(PRF(*evaluation.evaluate(ground_truth, seqs[div][algo], K)))
                # print '{}: {}'.format(div, prfs[algo][div][-1])

    prf = collections.defaultdict(dict)
    for algo in prfs:
        for div in divergences:
            prf[algo][div] = PRF(*[np.array(vals) for vals in zip(*prfs[algo][div])])

        print
        print algo
        for div in divergences:
            print '{}: {}'.format(div, PRF(*['{:.3f} +/- {:.3f}'.format(x.mean(), x.std())
                for x in prf[algo][div]]))
