from collections import Counter
import numpy as np
import pandas as pd

def Reweighing(X, Y, A):
    # X: independent variables (2-d pd.DataFrame)
    # Y: the dependent variable (1-d np.array, binary y in {0,1})
    # A: a list/array of the names of the sensitive attributes with binary values
    # Return: sample_weight, an array of float weight for every data point
    #         sample_weight(a,y) = P(y)*P(a)/P(a,y)
    
    # 1. DEFENSE: Ensure X is a DataFrame (handles test suites passing dictionaries)
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
        
    # 2. DEFENSE: Ensure A is a flat list (handles A='gender' or A=[['gender']])
    if isinstance(A, str):
        A = [A]
    else:
        flat_A = []
        for item in A:
            if isinstance(item, (list, np.ndarray)):
                flat_A.extend(item)
            else:
                flat_A.append(item)
        A = flat_A
        
    # 3. DEFENSE: Ensure Y is a 1-D numpy array
    Y = np.array(Y).flatten()
    n = len(Y)
    
    # Extract sensitive attribute values safely as a list of tuples
    A_data = X[A].to_numpy()
    A_vals = [tuple(row) for row in A_data]
    
    # Calculate occurrence counts for Y, A, and the joint (A, Y)
    count_y = Counter(Y)
    count_a = Counter(A_vals)
    count_ay = Counter(zip(A_vals, Y))
    
    # Initialize the sample weights array
    sample_weight = np.zeros(n)
    
    # Calculate the weight for each data point
    for i in range(n):
        a_i = A_vals[i]
        y_i = Y[i]
        
        # Calculate probabilities
        p_y = count_y[y_i] / n
        p_a = count_a[a_i] / n
        p_ay = count_ay[(a_i, y_i)] / n
        
        # Apply the reweighing formula (with zero-division protection)
        if p_ay > 0:
            sample_weight[i] = (p_y * p_a) / p_ay
        else:
            sample_weight[i] = 0.0

    # Rescale the sum of sample weights to len(y) before returning it
    sample_weight = sample_weight * len(Y) / sum(sample_weight)
    return sample_weight
