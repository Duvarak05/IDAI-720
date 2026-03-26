from collections import Counter
import numpy as np

def Reweighing(X, Y, A):
    # X: dataframe (not used here)
    # Y: binary labels (0/1)
    # A: sensitive attribute (binary)

    Y = np.array(Y)
    A = np.array(A)

    n = len(Y)

    # Probabilities
    P_y = Counter(Y)
    P_a = Counter(A)
    P_ay = Counter(zip(A, Y))

    # convert counts -> probabilities
    for k in P_y:
        P_y[k] /= n
    for k in P_a:
        P_a[k] /= n
    for k in P_ay:
        P_ay[k] /= n

    # compute weights
    sample_weight = np.zeros(n)

    for i in range(n):
        sample_weight[i] = (P_y[Y[i]] * P_a[A[i]]) / P_ay[(A[i], Y[i])]

    # Rescale weights to sum = len(Y)
    sample_weight = sample_weight * len(Y) / sum(sample_weight)

    return sample_weight

