import distributions
import numpy as np

def gen_hmm(pi, A, obs_distr, T):
    K = len(obs_distr)
    seq = np.zeros(T, dtype=int)
    X = np.zeros((T,obs_distr[0].dim))

    seq[0] = np.argmax(np.random.multinomial(1, pi))

    for t in range(T-1):
        seq[t+1] = np.argmax(np.random.multinomial(1, A[seq[t]]))

    for k in range(K):
        X[seq == k] = obs_distr[k].sample(np.sum(seq == k))

    return seq, X

def gen_hsmm(pi, A, obs_distr, dur_distr, T):
    K = len(obs_distr)
    seq = np.zeros(T, dtype=int)
    X = np.zeros((T,obs_distr[0].dim))

    t = 0
    while t < T:
        if t == 0:
            seq[t] = np.argmax(np.random.multinomial(1, pi))
        else:
            seq[t] = np.argmax(np.random.multinomial(1, A[seq[t-1]]))

        d = dur_distr[seq[t]].sample()
        seq[t:t+d] = seq[t]
        t = t + d

    for k in range(K):
        X[seq == k] = obs_distr[k].sample(np.sum(seq == k))

    return seq, X

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

    # seq, X = gen_hmm(pi, A, obs_distr, 10000)
    seq, X = gen_hsmm(pi, A, obs_distr, dur_distr, 10000)
