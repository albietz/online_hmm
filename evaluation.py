import collections
import munkres
import numpy as np
from sklearn.metrics import confusion_matrix

def best_alignment(true_seq, seq, K):
    '''
    Find the best matching between the classes in the true and
    predicted sequence, and return the modified sequence.
    '''
    cm = confusion_matrix(seq, true_seq, np.arange(K))
    _, perm = zip(*munkres.Munkres().compute(-cm))
    perm = np.asarray(perm)
    return perm[seq]

def prf(true_seq, seq):
    '''
    Precision, Recall, F-measure
    true_seq and seq are binary numpy arrays of the true and predicted
    sequences, respectively.
    '''
    tp = float(np.sum(np.logical_and(seq == 1, true_seq == 1)))
    pos_pred = np.sum(seq == 1)
    pos_true = np.sum(true_seq == 1)
    p = tp / pos_pred if pos_pred > 0 else 0.
    r = tp / pos_true if pos_true > 0 else 0.
    f = 2 * p * r / (p + r) if p+r > 0 else 0.
    return p, r, f

def evaluate(true_seq, seq, K):
    seq = best_alignment(true_seq, seq, K)
    p_ev, r_ev, f_ev = prf(true_seq != 0, seq != 0) # 0=background

    p_cw, r_cw, f_cw = [], [], []
    for k in range(1, K):
        p, r, f = prf(true_seq == k, seq == k)
        p_cw.append(p)
        r_cw.append(r)
        f_cw.append(f)

    PRF = collections.namedtuple('PRF', 'p r f pcw rcw fcw')
    return PRF(p_ev, r_ev, f_ev, np.mean(p_cw), np.mean(r_cw), np.mean(f_cw))

