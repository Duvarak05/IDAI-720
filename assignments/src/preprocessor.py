from collections import Counter
import numpy as np

def Reweighing(X, Y, A):
    n = len(Y)

    # X is a dict, A is a list of key names
    a_cols = [A] if isinstance(A, str) else list(A)

    # Build sensitive attribute tuples from dict values
    a_vals = [tuple(X[c][i] for c in a_cols) for i in range(n)]

    y_counts  = Counter(Y)
    a_counts  = Counter(a_vals)
    ay_counts = Counter(zip(a_vals, Y))

    p_y  = {y: c / n for y, c in y_counts.items()}
    p_a  = {a: c / n for a, c in a_counts.items()}
    p_ay = {ay: c / n for ay, c in ay_counts.items()}

    sample_weight = np.array([
        p_y[y] * p_a[a] / p_ay[(a, y)]
        for a, y in zip(a_vals, Y)
    ])

    return sample_weight * n / sample_weight.sum()
