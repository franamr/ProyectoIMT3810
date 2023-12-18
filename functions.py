
import numpy as np


def aca(matrix, tolerance = 1e-8, max_iter = 40):
    '''
    Adaptive Cross Approximation
    
    Input 
    matrix: matrix, 
    tolerance: stoping tolerance, 1e-8 by default
    max_iter: maximum number of iterations, 40 by default
    
    Output: 
    A, B rectangular matrices, M = A.T*B
    '''
    n = np.shape(matrix)[0]
    rows = []
    cols = []

    for v in range(1, max_iter+1):
        # Compute the maximal entry
        i_max, j_max = np.unravel_index(np.abs(matrix).argmax(), matrix.shape)
        delta = matrix[i_max, j_max]

        if delta == 0:
            return np.array(rows),np.array(cols)
        
        else:
            a_row = matrix[:, j_max].copy()
            rows.append(a_row)
            b_col = matrix[i_max, :].copy()/delta
            cols.append(b_col)
            matrix -= np.outer(a_row, b_col)
            eps = (np.linalg.norm(a_row)*np.linalg.norm(b_col))
            if eps < tolerance:
                return np.array(rows),np.array(cols)

    return np.array(rows), np.array(cols)
  
  
def aca_matvec(matrix, vector, max_iter, tolerance = 1e-8):
    '''
    Matvec operation for matrix with Adaptive Cross Approximation
    
    Input
    matrix : Matrix 
    vector: vector
    max_iter: maximum number of iterations
    tolerance: ACA tolerance, 1e-8 by default
    
    Output:
    vector: matrix*vector
    '''
    rows, cols = aca(matrix, tolerance, max_iter)
    return rows.T@(cols@vector)


def hmatvec(matrix, vector, max_dimension):
    '''
    Matvec operation for matrix with hierarchical compression
    with weak admissibility condition.
    
    Input: 
    matrix: Matrix
    vector: vecotr
    max_dimension: dimension of dense sub-matrix block
    
    Output:
    vector: Matrix*vector
    '''
    n = np.shape(vector)[0]
    
    if n <= max_dimension:
        return np.dot(matrix, vector)
    else:
        matrix11 = matrix[:n//2, :n//2]
        matrix12 = matrix[:n//2, n//2:]
        matrix21 = matrix[n//2:, :n//2]
        matrix22 = matrix[n//2:, n//2:]
        vector1 = vector[:n//2]
        vector2 = vector[n//2:]
        result1 = hmatvec(matrix11, vector1, max_dimension) + \
            aca_matvec(matrix12, vector2, np.shape(matrix12)[0])
        result2 = hmatvec(matrix21, vector1, max_dimension) + \
            aca_matvec(matrix22, vector2, np.shape(matrix22)[0])

        return np.concatenate((result1,result2))