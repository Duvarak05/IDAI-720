from collections import Counter
import numpy as np

def Reweighing(X, Y, A):
    Y = np.array(Y)

    # If A is column name(s), extract from X
    try:
        A_vals = list(X[A].itertuples(index=False, name=None))
    except:
        A_vals = np.array(A)
        if A_vals.ndim == 1:
            A_vals = [(a,) for a in A_vals]

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

    sample_weight = np.zeros(n)

    for i in range(n):
        sample_weight[i] = (P_y[Y[i]] * P_a[A_vals[i]]) / P_ay[(A_vals[i], Y[i])]

    # rescale
    sample_weight = sample_weight * len(Y) / sum(sample_weight)

    return sample_weight
