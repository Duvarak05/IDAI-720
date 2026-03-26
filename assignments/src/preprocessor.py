from collections import Counter
import numpy as np

def Reweighing(X, Y, A):
    n = len(Y)

    # Support A as a string (single attr) or list of strings
    a_cols = [A] if isinstance(A, str) else list(A)

    # Build per-row sensitive attribute tuples
    a_vals = list(map(tuple, X[a_cols].values))

    # Marginal counts
    y_counts  = Counter(Y)
    a_counts  = Counter(a_vals)
    ay_counts = Counter(zip(a_vals, Y))

    # Probabilities
    p_y  = {y: c / n for y, c in y_counts.items()}
    p_a  = {a: c / n for a, c in a_counts.items()}
    p_ay = {ay: c / n for ay, c in ay_counts.items()}

    # sample_weight(a, y) = P(y) * P(a) / P(a, y)
    sample_weight = np.array([
        p_y[y] * p_a[a] / p_ay[(a, y)]
        for a, y in zip(a_vals, Y)
    ])

    # Rescale so weights sum to n
    sample_weight = sample_weight * n / sample_weight.sum()

    return sample_weight
