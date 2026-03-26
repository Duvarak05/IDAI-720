from collections import Counter
import numpy as np

def Reweighing(X, Y, A):
    # A is list of sensitive column names
    A_vals = list(X[A].itertuples(index=False, name=None))
    Y = np.array(Y)

    n = len(Y)

    # counts
    P_y = Counter(Y)
    P_a = Counter(A_vals)
    P_ay = Counter(zip(A_vals, Y))

    # convert to probabilities
    for k in P_y:
        P_y[k] /= n
    for k in P_a:
        P_a[k] /= n
    for k in P_ay:
        P_ay[k] /= n

    # compute weights
    sample_weight = np.zeros(n)
    for i in range(n):
        sample_weight[i] = (P_y[Y[i]] * P_a[A_vals[i]]) / P_ay[(A_vals[i], Y[i])]

    # Rescale
    sample_weight = sample_weight * len(Y) / sum(sample_weight)

    return sample_weight
    # Rescale the sum of sample weights to len(y) before returning it
    sample_weight = sample_weight * len(Y) / sum(sample_weight)
    return sample_weight
