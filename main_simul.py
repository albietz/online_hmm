import distributions
import gen_data
import hmm
import hsmm
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    K = 2
    pi = np.array([0.3, 0.7])
    A = np.array([[0.1, 0.9],
                  [0.2, 0.8]])
    obs_distr = [distributions.Gaussian(np.array([3., 0.]),
                    np.array([[2., 1.], [1., 4.]])),
                 distributions.Gaussian(np.array([-2., 3.]),
                     np.array([[3., -1.], [-1., 2.]]))]

    dur_distr = [distributions.NegativeBinomial(15, 0.3, D=200) for _ in range(K)]

    seq, X = gen_data.gen_hmm(pi, A, obs_distr, 5000)
    # seq, X = gen_data.gen_hsmm(pi, A, obs_distr, dur_distr, 10000)

    init_pi = np.ones(K) / K
    init_obs_distr = [distributions.Gaussian(np.array([1., 0.]), np.eye(2)),
        distributions.Gaussian(np.array([0., 1.]), np.eye(2))]

    # HMM
    print 'HMM - batch EM'
    tau, A_batch, obs_distr_batch, pi_batch, ll_train, _ = hmm.em_hmm(X, init_pi, init_obs_distr)
    em_hmm_seq = np.argmax(tau, axis=1)

    step = lambda t: 1. / (t ** 0.6)
    print 'HMM - online EM'
    online_hmm_seq, tau, A_online, obs_distr_online = hmm.online_em_hmm(X, init_pi, init_obs_distr, t_min=80, step=step)
    online_hmm_ll = hmm.log_likelihood(*hmm.alpha_beta(X, pi_batch, A_online, obs_distr_online))
