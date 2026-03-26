from collections import Counter
import numpy as np

def Reweighing(X, Y, A):
    Y = np.array(Y)
    A = np.array(A)
    n = len(Y)

    # Use tolist() so Counter gets plain Python ints, not numpy scalars
    P_y  = Counter(Y.tolist())
    P_a  = Counter(A.tolist())
    P_ay = Counter(zip(A.tolist(), Y.tolist()))

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
        a_i = A[i].item()   # numpy scalar → Python int
        y_i = Y[i].item()
        sample_weight[i] = (P_y[y_i] * P_a[a_i]) / P_ay[(a_i, y_i)]

    # Rescale the sum of sample weights to len(Y)
    sample_weight = sample_weight * len(Y) / sum(sample_weight)
    return sample_weight
