from collections import Counter
import numpy as np

def Reweighing(X, Y, A):
    n = len(Y)
    
    # Compute P(Y=y) for each y
    y_counts = Counter(Y)
    p_y = {y: count / n for y, count in y_counts.items()}
    
    # Compute joint (a, y) groups using the sensitive attribute columns
    a_vals = X[A].apply(tuple, axis=1)
    
    # Compute P(A=a) and P(A=a, Y=y)
    a_counts = Counter(a_vals)
    ay_counts = Counter(zip(a_vals, Y))
    
    p_a   = {a: count / n for a, count in a_counts.items()}
    p_ay  = {ay: count / n for ay, count in ay_counts.items()}
    
    # sample_weight(a, y) = P(Y=y) * P(A=a) / P(A=a, Y=y)
    sample_weight = np.array([
        p_y[y] * p_a[a] / p_ay[(a, y)]
        for a, y in zip(a_vals, Y)
    ])
    
    # Rescale so weights sum to n
    sample_weight = sample_weight * n / sample_weight.sum()
    
    return sample_weight
