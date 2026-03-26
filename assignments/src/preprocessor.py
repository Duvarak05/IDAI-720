from collections import Counter
import numpy as np

def Reweighing(X, Y, A):
    n = len(Y)

    # Convert A to a flat list of column names regardless of input type
    a_cols = A if not isinstance(A, str) else [A]

    # Build sensitive attribute tuples directly from numpy to avoid pandas indexing issues
    a_vals = [tuple(row) for row in X.iloc[:, [X.columns.get_loc(c) for c in a_cols]].values]

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
