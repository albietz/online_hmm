import distributions
import gen_data
import hmm
import hsmm
import numpy as np
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
    if sys.argv[1] == 'HMM':
        # HMM
        K = 2
        pi = np.array([0.3, 0.7])
        A = np.array([[0.1, 0.9],
                      [0.2, 0.8]])
        obs_distr = [distributions.Gaussian(np.array([3., 0.]),
                        np.array([[2., 1.], [1., 4.]])),
                     distributions.Gaussian(np.array([-2., 3.]),
                         np.array([[3., -1.], [-1., 2.]]))]

        seq, X = gen_data.gen_hmm(pi, A, obs_distr, 10000)
        seq_test, Xtest = gen_data.gen_hmm(pi, A, obs_distr, 1000)

        init_pi = np.ones(K) / K
        init_obs_distr = [distributions.Gaussian(np.array([1., 0.]), np.eye(2)),
            distributions.Gaussian(np.array([0., 1.]), np.eye(2))]

        # HMM
        print 'HMM - batch EM'
        tau, A_batch, obs_distr_batch, pi_batch, ll_train, ll_test = hmm.em_hmm(X, init_pi, init_obs_distr, Xtest=Xtest)
        em_hmm_seq = np.argmax(tau, axis=1)

        step = lambda t: 1. / (t ** 0.7)
        print 'HMM - online EM'
        def monitor(A, obs_distr):
            return obs_distr[0].mean[0]

        online_hmm_seq, tau, A_online, obs_distr_online, ll_test_online, mon_online = hmm.online_em_hmm(X, init_pi, init_obs_distr, t_min=80, step=step, Xtest=None, monitor=monitor)
        online_hmm_ll = hmm.log_likelihood(*hmm.alpha_beta(X, pi_batch, A_online, obs_distr_online))

        incr_hmm_seq, tau, A_incr, obs_distr_incr, ll_test_incr, mon_incr = hmm.incremental_em_hmm(X, init_pi, init_obs_distr, t_min=80, step=step, Xtest=None, monitor=monitor)
        incr_hmm_ll = hmm.log_likelihood(*hmm.alpha_beta(X, pi_batch, A_incr, obs_distr_incr))

    elif sys.argv[1] == 'HSMM':
        # HSMM
        K = 3
        pi = np.array([0.3, 0.6, 0.1])
        A = np.array([[0.01, 0.85, 0.14],
                      [0.25, 0.01, 0.74],
                      [0.39, 0.60, 0.01]])
        obs_distr = [distributions.Gaussian(np.array([3., 0.]),
                        np.array([[2., 1.], [1., 4.]])),
                     distributions.Gaussian(np.array([-2., 3.]),
                         np.array([[3., -1.], [-1., 2.]])),
                     distributions.Gaussian(np.array([-2., -3.]),
                         np.array([[3., -1.], [-1., 2.]]))]

        dur_distr = [distributions.NegativeBinomial(15, 0.3, D=100) for _ in range(K)]
        # dur_distr = [distributions.PoissonDuration(20, D=100),
        #              distributions.PoissonDuration(40, D=100),
        #              distributions.PoissonDuration(60, D=100)]
        seq, X = gen_data.gen_hsmm(pi, A, obs_distr, dur_distr, 1000)

        init_pi = np.ones(K) / K
        init_obs_distr = [distributions.Gaussian(np.array([1., 0.]), np.eye(2)),
                          distributions.Gaussian(np.array([0., 1.]), np.eye(2)),
                          distributions.Gaussian(np.array([-1., 0.]), np.eye(2))]
        init_dur_distr = [distributions.NegativeBinomial(15, 0.3, D=100) for _ in range(K)]
        # init_dur_distr = [distributions.PoissonDuration(50, D=100),
        #                   distributions.PoissonDuration(50, D=100),
        #                   distributions.PoissonDuration(50, D=100)]

        print 'HSMM - batch EM'
        tau, A_batch, obs_distr_batch, dur_distr_batch, pi_batch, ll_train, _ = \
                hsmm.em_hsmm(X, init_pi, init_obs_distr, init_dur_distr, fit_durations=True)
        em_hsmm_seq = np.argmax(tau, axis=1)

        step = lambda t: 1. / (t ** 0.6)
        print 'HSMM - online EM'
        online_hsmm_seq, A_online, obs_distr_online, dur_distr_online = \
                hsmm.online_em_hsmm(X, init_pi, init_obs_distr, dur_distr, t_min=80, step=step, fit_durations=False)
        _, lalphastar, _, lbetastar = hsmm.alpha_beta(X, pi_batch, A_online, obs_distr_online, dur_distr_online)
        online_hsmm_ll = hsmm.log_likelihood(lalphastar, lbetastar)
