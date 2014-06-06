import copy
import distributions
import em
import kmeans
import numpy as np
import matplotlib.pyplot as plt
import sys
from numpy import newaxis as nax
from numpy.linalg import det, inv

def alpha_beta(X, pi, A, obs_distr, dur_distr):
    '''A[i,j] = p(z_{t+1} = j | z_t = i)'''
    T = X.shape[0]
    K = pi.shape[0]
    D = len(dur_distr)
    lA = np.log(A)
    lemissions = np.zeros((T,K))
    for k in range(K):
        lemissions[:,k] = obs_distr[k].log_pdf(X)

    # lD[d,i] = log p(d|i)
    lD = np.hstack(d.log_vec()[:,nax] for d in dur_distr)
    D = lD.shape[0]

    # Forward messages
    lalpha = np.zeros((T, K))       # 1 to T
    lalphastar = np.zeros((T, K))   # 0 to T-1

    lalphastar[0] = np.log(pi)
    for t in range(T-1):
        dmax = min(D,t+1)
        a = lalphastar[t+1-dmax:t+1] + lD[:dmax][::-1] + np.cumsum(lemissions[t+1-dmax:t+1][::-1], axis=0)[::-1]
        lalpha[t] = np.logaddexp.reduce(a, axis=0)

        a = lalpha[t][:,nax] + lA
        lalphastar[t+1] = np.logaddexp.reduce(a, axis=0)
    t = T-1
    dmax = min(D,t+1)
    a = lalphastar[t+1-dmax:t+1] + lD[:dmax][::-1] + np.cumsum(lemissions[t+1-dmax:t+1][::-1], axis=0)[::-1]
    lalpha[t] = np.logaddexp.reduce(a, axis=0)

    # Backward messages
    lbeta = np.zeros((T, K))        # 1 to T
    lbetastar = np.zeros((T, K))    # 0 to T-1

    lbeta[T-1] = np.zeros(K)
    for t in reversed(range(T-1)):
        # TODO: right-censoring
        dmax = min(D, T-t)
        b = lbeta[t:t+dmax] + lD[:dmax] + np.cumsum(lemissions[t:t+dmax], axis=0)
        lbetastar[t] = np.logaddexp.reduce(b, axis=0)
        if t > 0:
            b = lbetastar[t] + lA
            lbeta[t-1] = np.logaddexp.reduce(b, axis=1)

    return lalpha, lalphastar, lbeta, lbetastar

def viterbi(X, pi, A, obs_distr, dur_distr, use_distance=False):
    T = X.shape[0]
    K = pi.shape[0]

    lA = np.log(A)
    lemissions = np.zeros((T,K))
    for k in range(K):
        if use_distance:
            lemissions[:,k] = - obs_distr[k].distances(X)
        else:
            lemissions[:,k] = obs_distr[k].log_pdf(X)

    # lD[d,i] = log p(d|i)
    lD = np.hstack(d.log_vec()[:,nax] for d in dur_distr)
    D = lD.shape[0]

    lgamma = np.zeros((T,K))
    lgammastar = np.zeros((T,K))
    back = np.zeros((T,K))  # back-pointers
    backstar = np.zeros((T,K))

    lgammastar[0] = np.log(pi)
    for t in range(T-1):
        dmax = min(D,t+1)
        a = lgammastar[t+1-dmax:t+1] + lD[:dmax][::-1] + np.cumsum(lemissions[t+1-dmax:t+1][::-1], axis=0)[::-1]
        a = a[::-1]
        lgamma[t] = np.max(a, axis=0)
        back[t] = np.argmax(a, axis=0)

        a = lgamma[t][:,nax] + lA
        lgammastar[t+1] = np.max(a, axis=0)
        backstar[t+1] = np.argmax(a, axis=0)

    t = T-1
    dmax = min(D,t+1)
    a = lgammastar[t+1-dmax:t+1] + lD[:dmax][::-1] + np.cumsum(lemissions[t+1-dmax:t+1][::-1], axis=0)[::-1]
    a = a[::-1]
    lgamma[t] = np.max(a, axis=0)
    back[t] = np.argmax(a, axis=0)

    # Tracer()()
    # recover MAP from back-pointers
    t = T - 1
    seq = []
    i = int(np.argmax(lgamma[t]))
    d = int(back[t,i])
    while t > 0:
        seq.extend([i] * (d+1))
        i = int(backstar[t-d,i])
        t = t - d - 1
        d = int(back[t,i])

    return np.array(list(reversed(seq))), lgamma

def smoothing(lalpha, lalphastar, lbeta, lbetastar):
    '''Computes all the p(q_t | u_1, ..., u_T)'''
    T, K = lalpha.shape

    lgamma = lalpha + lbeta
    lgammastar = lalphastar + lbetastar
    gamma = np.exp(lgamma - np.logaddexp.reduce(lgamma, axis=1)[:,nax])
    gammastar = np.exp(lgammastar - np.logaddexp.reduce(lgammastar, axis=1)[:,nax])

    tau = np.cumsum(gammastar - np.vstack((np.zeros(K), gamma[:-1])), axis=0)
    return tau

def pairwise_smoothing(X, lalpha, lbetastar, A):
    '''returns log_p[t,i,j] = log p(q_t = i, q_{t+1} = j|u)'''
    T, K = lalpha.shape
    lA = np.log(A)

    log_p = np.zeros((T,K,K))
    for t in range(T-1):
        log_p[t,:,:] = lalpha[t][:,nax] + lA + lbetastar[t]

    log_p2 = log_p.reshape(T, K*K)
    log_p = np.reshape(log_p2 - np.logaddexp.reduce(log_p2, axis=1)[:,nax],
                       (T,K,K))

    return log_p

def log_likelihood(pi, lbetastar):
    '''p(u_1, ..., u_T) = \sum_i pi(i) beta*_0(i)'''
    return np.logaddexp.reduce(np.log(pi) + lbetastar[0])

def em_hsmm(X, pi, init_obs_distr, dur_distr, n_iter=10, Xtest=None, fit_durations=False):
    pi = pi.copy()
    obs_distr = copy.deepcopy(init_obs_distr)
    T = X.shape[0]
    K = len(obs_distr)

    A = 1. / K * np.ones((K,K))

    ll_train = []
    ll_test = []

    lalpha, lalphastar, lbeta, lbetastar = alpha_beta(X, pi, A, obs_distr, dur_distr)
    ll_train.append(log_likelihood(pi, lbetastar))
    if Xtest is not None:
        _, _, _, lbetastar_test = \
                alpha_beta(Xtest, pi, A, obs_distr, dur_distr)
        ll_test.append(log_likelihood(pi, lbetastar_test))

    for it in range(n_iter):
        # E-step
        tau = np.exp(smoothing(lalpha, lalphastar, lbeta, lbetastar))
        tau_pairs = np.exp(pairwise_smoothing(X, lalpha, lbetastar, A))

        # M-step
        pi = tau[0,:] / np.sum(tau[0,:])

        A = np.sum(tau_pairs, axis=0)
        A = A / A.sum(axis=1)[:,nax]

        for j in range(K):
            obs_distr[j].max_likelihood(X, tau[:,j])
            if fit_durations:
                raise NotImplementedError

        lalpha, lalphastar, lbeta, lbetastar = \
                alpha_beta(X, pi, A, obs_distr, dur_distr)
        ll_train.append(log_likelihood(pi, lbetastar))
        if Xtest is not None:
            _, _, _, lbetastar_test = \
                    alpha_beta(Xtest, pi, A, obs_distr, dur_distr)
            ll_test.append(log_likelihood(pi, lbetastar_test))

    return tau, A, obs_distr, dur_distr, pi, ll_train, ll_test

def map_em_hsmm(X, init_obs_distr, dur_distr, A=None, n_iter=10):
    obs_distr = copy.deepcopy(init_obs_distr)
    T = X.shape[0]
    K = len(obs_distr)

    pi = np.ones(K)
    if A is None:
        A = 1. / K * np.ones((K,K))

    energies = []
    for it in range(n_iter):
        # E-step
        seq, lgamma = viterbi(X, pi, A, obs_distr, dur_distr, use_distance=True)
        energy = np.max(lgamma[T-1])
        energies.append(energy)

        # M-step
        for k in range(K):
            if np.sum(seq == k) > 0:
                obs_distr[k].max_likelihood(X, seq == k)

    return seq, obs_distr, dur_distr, energies

def online_opt_hsmm(X, lambda1, lambda2, lcost, init_obs_distr=None, dist_cls=distributions.SquareDistance):
    if init_obs_distr is None:
        obs_distr = [dist_cls(X[0])]
    else:
        obs_distr = copy.deepcopy(init_obs_distr)
    T = X.shape[0]
    seq = -np.ones(T)

    costs = np.array([d.distances(X[0]) for d in obs_distr])
    best = np.argmin(costs)
    seq[0] = best
    counts = collections.defaultdict(int)
    counts[best] = 1

    cost = lcost[1]
    curr_seg_length = 1

    for t in range(1, T):
        costs = np.array([d.distances(X[t]) for d in obs_distr])
        if curr_seg_length + 1 < len(lcost):
            costs[seq[t-1]] += lambda1 * (lcost[curr_seg_length + 1] - lcost[curr_seg_length])
        # 1 == cost of transition
        costs[np.arange(len(obs_distr)) != seq[t-1]] += lambda1 * (1 + lcost[1])

        best = np.argmin(costs)

        if best == seq[t-1]:
            curr_seg_length += 1
        else:
            curr_seg_length = 1

        if costs[best] < lambda1 * (1 + lcost[1]) + lambda2:
            seq[t] = best
            counts[best] += 1
            obs_distr[best].online_update(X[t], 1. / counts[best])
            cost += costs[best]
        else:
            best = len(obs_distr)
            seq[t] = best
            counts[best] = 1
            obs_distr.append(dist_cls(X[t]))
            cost += lambda1 * (1 + lcost[1]) + lambda2

    return seq, obs_distr, cost

if __name__ == '__main__':
    X = np.loadtxt('EMGaussian.data')
    Xtest = np.loadtxt('EMGaussian.test')
    K = 4

    # Run simple EM (no HMM)
    iterations = 40
    assignments, centers, _ = kmeans.kmeans_best_of_n(X, K, n_trials=5)
    new_centers = [distributions.Gaussian(c.mean, np.eye(2)) \
                for c in centers]
    tau, obs_distr, pi, gmm_ll_train, gmm_ll_test = \
            em.em(X, new_centers, assignments, n_iter=iterations, Xtest=Xtest)

    # example with fixed parameters
    A = 1. / 6 * np.ones((K,K))
    A[np.diag(np.ones(K)) == 1] = 0.5

    lalpha, lbeta = alpha_beta(Xtest, pi, A, obs_distr)
    log_p = smoothing(lalpha, lbeta)
    p = np.exp(log_p)

    def plot_traj(p):
        plt.figure()
        ind = np.arange(100)
        for k in range(K):
            plt.subplot(K,1,k+1)
            plt.bar(ind, p[:100,k])

    plot_traj(p)

    # EM for the HMM
    tau, A, obs_distr, pi, ll_train, ll_test = \
            em_hmm(X, pi, obs_distr, Xtest=Xtest)

    plt.figure()
    plt.plot(ll_train, label='training')
    plt.plot(ll_test, label='test')
    plt.xlabel('iterations')
    plt.ylabel('log-likelihood')
    plt.legend()

    # print all log-likelihoods
    print '{:<14} {:>14} {:>14}'.format('', 'train', 'test')
    print '{:<14} {:>14.3f} {:>14.3f}'.format('General GMM', gmm_ll_train[-1], gmm_ll_test[-1])
    print '{:<14} {:>14.3f} {:>14.3f}'.format('HMM', ll_train[-1], ll_test[-1])

    # Viterbi
    seq, _ = viterbi(X, pi, A, obs_distr)
    plt.figure()
    plt.scatter(X[:,0], X[:,1], c=seq)
    plt.title('most likely sequence, training')
    seq_test, _ = viterbi(Xtest, pi, A, obs_distr)
    plt.figure()
    plt.scatter(Xtest[:,0], Xtest[:,1], c=seq_test)
    plt.title('most likely sequence, test')

    # marginals in each state
    lalpha, lbeta = alpha_beta(Xtest, pi, A, obs_distr)
    log_p = smoothing(lalpha, lbeta)
    plot_traj(np.exp(log_p))

    def plot_traj(p):
        plt.figure()
        ind = np.arange(100)
        for k in range(K):
            plt.subplot(K,1,k+1)
            plt.bar(ind, p[:100,k])

    # most likely state according to marginals vs viterbi
    plt.figure()
    ind = np.arange(100)
    c = ['b', 'g', 'r', 'y']
    plt.subplot(211)
    for k in range(K):
        plt.bar(ind, k == np.argmax(log_p[:100], axis=1), color=c[k])
    plt.title('marginals')

    plt.subplot(212)
    for k in range(K):
        plt.bar(ind, k == np.array(seq_test[:100]), color=c[k])
    plt.title('marginals')
