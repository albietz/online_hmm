import collections
import copy
import distributions
import em
import kmeans
import numpy as np
import matplotlib.pyplot as plt
import sys
from numpy import newaxis as nax
from numpy.linalg import det, inv

from IPython.core.debugger import Tracer

def alpha_beta(X, pi, A, obs_distr):
    '''A[i,j] = p(z_{t+1} = j | z_t = i)'''
    T = X.shape[0]
    K = pi.shape[0]
    lalpha = np.zeros((T, K))
    lbeta = np.zeros((T, K))
    lA = np.log(A)
    lemissions = np.zeros((T,K))
    for k in range(K):
        lemissions[:,k] = obs_distr[k].log_pdf(X)

    lalpha[0,:] = np.log(pi) + lemissions[0,:]
    for t in range(1,T):
        a = lalpha[t-1:t,:].T + lA + lemissions[t,:]
        lalpha[t,:] = np.logaddexp.reduce(a, axis=0)

    lbeta[T-1,:] = np.zeros(K)
    for t in reversed(range(T-1)):
        b = lbeta[t+1,:] + lA + lemissions[t+1,:]
        lbeta[t,:] = np.logaddexp.reduce(b, axis=1)

    return lalpha, lbeta

def viterbi(X, pi, A, obs_distr, use_distance=False):
    T = X.shape[0]
    K = pi.shape[0]

    # gamma_t(j) = max_{q_1, ..., q_{t-1}} p(q_1, ..., q_{t-1}, q_t=j, u_1, ..., u_t)
    # lgamma[t,j] = log gamma_t(j)
    lgamma = np.zeros((T,K))
    back = np.zeros((T,K), dtype=int)  # back-pointers
    lA = np.log(A)
    lemissions = np.zeros((T,K))
    for k in range(K):
        if use_distance:
            lemissions[:,k] = - obs_distr[k].distances(X)
        else:
            lemissions[:,k] = obs_distr[k].log_pdf(X)

    lgamma[0,:] = np.log(pi) + lemissions[0,:]
    for t in range(1,T):
        a = lgamma[t-1:t,:].T + lA + lemissions[t,:]
        lgamma[t,:] = np.max(a, axis=0)
        ss = np.sum(lgamma[t,:] == a, axis=0)
        if np.max(ss) > 1:
            print ss, t
        back[t,:] = np.argmax(a, axis=0)

    # recover MAP from back-pointers
    seq = [int(np.argmax(lgamma[T-1,:]))]
    for t in reversed(range(1, T)):
        seq.append(back[t,seq[-1]])

    return np.array(list(reversed(seq)), dtype=int), lgamma

def smoothing(lalpha, lbeta):
    '''Computes all the p(q_t | u_1, ..., u_T)'''
    log_p = lalpha + lbeta
    return log_p - np.logaddexp.reduce(log_p, axis=1)[:,nax]

def mpm_sequence(X, pi, A, obs_distr):
    # sequence of maximum posterior marginal (smoothing)
    lalpha, lbeta = alpha_beta(X, pi, A, obs_distr)
    tau = np.exp(smoothing(lalpha, lbeta))
    return np.argmax(tau, axis=1)

def pairwise_smoothing(X, lalpha, lbeta, A, obs_distr):
    '''returns log_p[t,i,j] = log p(q_t = i, q_{t+1} = j|u)'''
    T, K = lalpha.shape
    lA = np.log(A)
    lemissions = np.zeros((T,K))
    for k in range(K):
        lemissions[:,k] = obs_distr[k].log_pdf(X)

    log_p = lalpha[:T-1,:,nax] + lA[nax,:,:] + lemissions[1:,nax,:] + lbeta[1:,nax,:]

    log_p2 = log_p.reshape(T-1, K*K)
    log_p = np.reshape(log_p2 - np.logaddexp.reduce(log_p2, axis=1)[:,nax],
                       (T-1,K,K))

    return log_p

def log_likelihood(lalpha, lbeta):
    '''p(u_1, ..., u_T) = \sum_i alpha_T(i) beta_T(i)'''
    T = lalpha.shape[0]
    return np.logaddexp.reduce(lalpha[T-1,:] + lbeta[T-1,:])

def em_hmm(X, pi, init_obs_distr, n_iter=10, Xtest=None):
    pi = pi.copy()
    obs_distr = copy.deepcopy(init_obs_distr)
    T = X.shape[0]
    K = len(obs_distr)

    A = 1. / K * np.ones((K,K))

    ll_train = []
    ll_test = []

    lalpha, lbeta = alpha_beta(X, pi, A, obs_distr)
    ll_train.append(log_likelihood(lalpha, lbeta))
    if Xtest is not None:
        lalpha_test, lbeta_test = alpha_beta(Xtest, pi, A, obs_distr)
        ll_test.append(log_likelihood(lalpha_test, lbeta_test))

    for it in range(n_iter):
        # E-step
        tau = np.exp(smoothing(lalpha, lbeta))
        tau_pairs = np.exp(pairwise_smoothing(X, lalpha, lbeta, A, obs_distr))

        # M-step
        prior = 1e-6
        pi = (prior + tau[0,:]) / np.sum(prior + tau[0,:])

        A = prior + np.sum(tau_pairs, axis=0)
        A = A / A.sum(axis=1)[:,nax]

        for j in range(K):
            obs_distr[j].max_likelihood(X, tau[:,j])

        lalpha, lbeta = alpha_beta(X, pi, A, obs_distr)
        ll_train.append(log_likelihood(lalpha, lbeta))
        if Xtest is not None:
            lalpha_test, lbeta_test = alpha_beta(Xtest, pi, A, obs_distr)
            ll_test.append(log_likelihood(lalpha_test, lbeta_test))

    return tau, A, obs_distr, pi, ll_train, ll_test

def map_em_hmm(X, init_obs_distr, n_iter=10):
    ''' Same as EM for the HMM, but the E-step is replaced by the
    current MAP estimate of the hidden chain (given by Viterbi).
    The transition probabilities aren't estimated.
    '''
    obs_distr = copy.deepcopy(init_obs_distr)
    T = X.shape[0]
    K = len(obs_distr)

    A = 0.99 * np.eye(K)
    A[A == 0] = 0.01
    A = A / A.sum(axis=1)[:,nax]
    pi = np.ones(K)

    energies = []
    for it in range(n_iter):
        # E-step
        seq, lgamma = viterbi(X, pi, A, obs_distr, use_distance=True)
        energy = np.max(lgamma[T-1])
        energies.append(energy)

        # M-step
        # NOTE: estimate A?
        for k in range(K):
            if np.sum(seq == k) > 0:
                obs_distr[k].max_likelihood(X, seq == k)

    return seq, obs_distr, energies

def online_opt_hmm(X, lambda1, lambda2, init_obs_distr=None, dist_cls=distributions.SquareDistance):
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

    cost = 0.

    for t in range(1, T):
        costs = np.array([d.distances(X[t]) for d in obs_distr])
        costs[np.arange(len(obs_distr)) != seq[t-1]] += lambda1
        best = np.argmin(costs)

        if costs[best] < lambda1 + lambda2:
            seq[t] = best
            counts[best] += 1
            obs_distr[best].online_update(X[t], 1. / counts[best])
            cost += costs[best]
        else:
            best = len(obs_distr)
            seq[t] = best
            counts[best] = 1
            obs_distr.append(dist_cls(X[t]))
            cost += lambda1 + lambda2

    return seq, obs_distr, cost

def online_em_hmm(X, init_pi, init_obs_distr, t_min=100, step=None, m_step_delta=1, Xtest=None, monitor=None):
    pi = init_pi.copy()
    obs_distr = copy.deepcopy(init_obs_distr)

    if step is None:
        step = lambda t: 1. / t

    T = X.shape[0]
    K = len(obs_distr)

    A = 1. / K * np.ones((K,K))
    seq = np.zeros(T, dtype=int)
    tau = np.zeros((T, K))

    emissions = np.array([d.pdf(X[0]) for d in obs_distr]).flatten()
    phi = pi * emissions
    phi /= phi.sum()
    tau[0] = phi
    seq[0] = np.argmax(phi)

    # rho[i, j, k]
    rho_pairs = np.zeros((K,K,K))
    rho_obs = [d.new_sufficient_statistics_hmm(X[0], i, K) for i, d in enumerate(obs_distr)]

    ll_test = []
    if Xtest is not None:
        lalpha_test, lbeta_test = alpha_beta(Xtest, pi, A, obs_distr)
        ll_test.append(log_likelihood(lalpha_test, lbeta_test))

    monitor_vals = []
    if monitor:
        monitor_vals.append(monitor(A, obs_distr))

    for t in range(1,T):
        # r[i,j] = p(Z_{t-1} = i | Z_t = j, x_{1:t-1})
        r = phi[:,nax] * A
        r /= r.sum(0)

        emissions = np.array([d.pdf(X[t]) for d in obs_distr]).flatten()
        phi = emissions * A.T.dot(phi)
        phi /= phi.sum()
        tau[t] = phi
        seq[t] = np.argmax(phi)

        # SA E-step
        s = step(t)
        rho_pairs = (1 - s) * rho_pairs.dot(r) + s * np.eye(K)[nax,:,:] * r[:,:,nax]

        for k in range(K):
            rho_obs[k].online_update(X[t], r, s)

        # M-step
        if t < t_min or t % m_step_delta != 0:
            continue
        # Tracer()()
        A = rho_pairs.dot(phi)
        A /= A.sum(axis=1)[:,nax]

        for k in range(K):
            obs_distr[k].online_max_likelihood(rho_obs[k], phi, t=t)

        if Xtest is not None and t % 100 == 0:
            lalpha_test, lbeta_test = alpha_beta(Xtest, pi, A, obs_distr)
            ll_test.append(log_likelihood(lalpha_test, lbeta_test))
        if monitor:
            monitor_vals.append(monitor(A, obs_distr))

    return seq, tau, A, obs_distr, ll_test, monitor_vals

def incremental_em_hmm(X, init_pi, init_obs_distr, t_min=100, step=None, m_step_delta=1, Xtest=None, monitor=None):
    pi = init_pi.copy()
    obs_distr = copy.deepcopy(init_obs_distr)

    if step is None:
        step = lambda t: 1. / t

    T = X.shape[0]
    K = len(obs_distr)

    A = 1. / K * np.ones((K,K))
    seq = np.zeros(T, dtype=int)  # online filtering sequence
    tau = np.zeros((T, K))        # filter

    emissions = np.array([d.pdf(X[0]) for d in obs_distr]).flatten()
    phi = pi * emissions  # current marginal in q-distribution
    phi /= phi.sum()
    tau[0] = phi
    seq[0] = np.argmax(phi)

    s_pairs = distributions.TransitionISufficientStatistics(K)
    s_obs = [d.new_incremental_sufficient_statistics(X[0], phi, i) for i, d in enumerate(obs_distr)]

    ll_test = []
    if Xtest is not None:
        lalpha_test, lbeta_test = alpha_beta(Xtest, pi, A, obs_distr)
        ll_test.append(log_likelihood(lalpha_test, lbeta_test))

    monitor_vals = []
    if monitor:
        monitor_vals.append(monitor(A, obs_distr))

    for t in range(1,T):
        # E-step
        emissions = np.array([d.pdf(X[t]) for d in obs_distr]).flatten()
        q = A * emissions  # new q_t transition
        q /= q.sum(axis=1)[:,nax]
        phi_q = phi[:,nax] * q  # for transition statistics
        phi = phi_q.sum(axis=0)

        # compute filter
        tau[t] = emissions * A.T.dot(tau[t-1])
        tau[t] /= tau[t].sum()
        seq[t] = np.argmax(tau[t])

        # sufficient statistics updates
        s = step(t)
        s_pairs.online_update(phi_q, s)

        for k in range(K):
            s_obs[k].online_update(X[t], phi, s)

        # M-step
        if t < t_min or t % m_step_delta != 0:
            continue
        # Tracer()()
        A = s_pairs.get_statistics()
        A /= A.sum(axis=1)[:,nax]

        for k in range(K):
            obs_distr[k].online_max_likelihood(s_obs[k], t=t)

        if Xtest is not None and t % 100 == 0:
            lalpha_test, lbeta_test = alpha_beta(Xtest, pi, A, obs_distr)
            ll_test.append(log_likelihood(lalpha_test, lbeta_test))
        if monitor:
            monitor_vals.append(monitor(A, obs_distr))

    return seq, tau, A, obs_distr, ll_test, monitor_vals

def incremental_em_hmm_add(X, lmbda, alpha=1.5, Kmax=30, init_pi=None, init_obs_distr=None,
                           dist_cls=distributions.KL, dist_params=None,
                           t_min=100, step=None, Xtest=None, monitor=None):
    if dist_params is None:
        dist_params = {}
    if init_obs_distr is None:
        obs_distr = [dist_cls(X[0], **dist_params)]
        pi = 1.
    else:
        obs_distr = copy.deepcopy(init_obs_distr)
        pi = copy.deepcopy(init_pi)

    if step is None:
        step = lambda t: 1. / t

    T = X.shape[0]

    K = 1
    A = np.zeros((K,K))
    A[0,0] = 1.
    seq = np.zeros(T, dtype=int)

    emissions = np.array([d.pdf(X[0]) for d in obs_distr]).flatten()
    phi = pi * emissions
    phi /= phi.sum()
    tau = phi
    seq[0] = np.argmax(phi)

    s_pairs = distributions.TransitionISufficientStatistics(Kmax)
    s_obs = [d.new_incremental_sufficient_statistics(X[0], phi, i) for i, d in enumerate(obs_distr)]
    t_init = [0 for _ in range(len(obs_distr))]

    ll_test = []
    if Xtest is not None:
        lalpha_test, lbeta_test = alpha_beta(Xtest, np.ones(K)/K, A, obs_distr)
        ll_test.append(log_likelihood(lalpha_test, lbeta_test))

    monitor_vals = []
    if monitor:
        monitor_vals.append(monitor(A, obs_distr))

    for t in range(1,T):
        # E-step
        new_distr = dist_cls(X[0], **dist_params)
        new_distr.max_likelihood(X[t][nax,:], np.ones(1))
        emissions = np.array([d.pdf(X[t]) for d in obs_distr + [new_distr]]).flatten()

        q = A * emissions[:K]
        Z = q.sum(axis=1)
        q /= Z[:,nax]
        phi_q = phi[:,nax] * q

        added_state = False
        if K < Kmax:  # test new model with K+1 states
            Anew = alpha - 1 + s_pairs.get_statistics()[:K,:K+1]
            Anew[np.eye(K) == 1] += 0.5*alpha
            Anew /= Anew.sum(axis=1)[:,nax]
            q_new = Anew * emissions
            Z_new = q_new.sum(axis=1)
            q_new /= Z_new[:,nax]
            phi_q_new = phi[:,nax] * q_new

            loglik = phi.dot(np.log(Z))
            loglik_new = phi.dot(np.log(Z_new))

            if loglik_new - lmbda > loglik:
                A = Anew
                q = q_new
                phi_q = phi_q_new

                added_state = True
                K = K + 1

                obs_distr.append(new_distr)
                t_init.append(t)

        phi = phi_q.sum(axis=0)

        # compute filter
        tau = emissions[:K] * A.T.dot(tau)
        tau /= tau.sum()
        seq[t] = np.argmax(tau)

        # sufficient statistics updates
        s = step(t)
        phq = np.zeros((Kmax,Kmax))
        phq[:phi_q.shape[0],:phi_q.shape[1]] = phi_q
        s_pairs.online_update(phq, s)

        for k in range(len(s_obs)):
            s_obs[k].online_update(X[t], phi, step(t - t_init[k]+1))

        if added_state:
            s_obs.append(new_distr.new_incremental_sufficient_statistics(X[t], phi, K-1))

        # M-step
        # if t < t_min:
        #     continue
        # Tracer()()
        A = alpha - 1 + s_pairs.get_statistics()[:K,:K]
        A[np.eye(K) == 1] += 0.5*alpha
        A /= A.sum(axis=1)[:,nax]

        for k in range(K):
            obs_distr[k].online_max_likelihood(s_obs[k], t=t-t_init[k]+1)

        if Xtest is not None and t % 100 == 0:
            lalpha_test, lbeta_test = alpha_beta(Xtest, np.ones(K)/K, A, obs_distr)
            ll_test.append(log_likelihood(lalpha_test, lbeta_test))
        if monitor:
            monitor_vals.append(monitor(A, obs_distr))

    Tracer()()
    return seq, A, obs_distr, ll_test, monitor_vals

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
