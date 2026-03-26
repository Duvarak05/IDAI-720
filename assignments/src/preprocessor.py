from collections import Counter
import numpy as np

def Reweighing(X, Y, A):
    # X: independent variables (2-d pd.DataFrame)
    # Y: the dependent variable (1-d np.array, binary y in {0,1})
    # A: a list/array of the names of the sensitive attributes with binary values
    # Return: sample_weight, an array of float weight for every data point
    #         sample_weight(a,y) = P(y)*P(a)/P(a,y)
    
    n = len(Y)
    
    # Extract sensitive attribute values for each row as a list of tuples 
    # (Tuples are hashable, making them compatible with Counter)
    A_vals = list(X[A].itertuples(index=False, name=None))
    
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
        
        # Apply the reweighing formula
        sample_weight[i] = (p_y * p_a) / p_ay

    # Rescale the sum of sample weights to len(y) before returning it
    sample_weight = sample_weight * len(Y) / sum(sample_weight)
    return sample_weight
