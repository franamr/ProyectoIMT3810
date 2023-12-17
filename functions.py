
import numpy as np

def ACA(M, tol, k):
    '''
    Adaptive Cross Approximation
    
    Input: M matrix, tol stoping tolerance, k max iterations
    Output: A, B rectangular matrices, M = A*B
    '''
    n = np.shape(M)[0]
    #R = np.zeros((t, s))
    rows = []
    cols = []

    for nu in range(1, k+1):
        # Compute the maximal entry in modulus
        i_max, j_max = np.unravel_index(np.abs(M).argmax(), M.shape)
        delta = M[i_max, j_max]

        if delta == 0:
            # The algorithm terminates with the exact rank nu-1 representation R_nu-1
            return np.array(rows),np.array(cols)
        else:
            # Compute the entries of the vectors a_nu, b_nu
            
            a_row = M[:, j_max].copy()
            rows.append(a_row)
            b_col = M[i_max, :].copy() / delta
            cols.append(b_col)
            # Subtract the rank one approximation
            M -= np.outer(a_row, b_col)
            eps = (np.linalg.norm(a_row)*np.linalg.norm(b_col))
            print(eps)
            if eps < tol:
                return np.array(rows),np.array(cols)

        # Output the rank k approximation
        #R += delta * np.outer(a_nu, b_nu)

    return np.array(rows),np.array(cols)



def aca_matvec(M,b,k,tol = 1e-8):
    '''
    Matvec operation for matriz with Adaptive Cross Approximation
    
    Input: 
    M : Matrix 
    b: vector
    
    Output:
    a = Mb
    '''
    rows, cols = ACA(M, tol,k)
    return cols.T@(rows@b)


def hmatvec(M,b, max_p):
    '''
    Matvec con hierarchical matrices
    
    Input: 
    M: Matrix
    b: vecotr
    
    Output:
    a = Mb
    '''
    n = np.shape(b)[0]
    a = np.zeros(n)
    
    if n <= max_p:
        return np.dot(M,b)
    else:
        
        M11 = M[:n//2, :n//2]
        M12 = M[:n//2, n//2:]
        M21 = M[n//2:, :n//2]
        M22 = M[n//2:, n//2:]
        b1 = b[:n//2]
        b2 = b[n//2:]
        print(np.shape(M12),np)
        #result1 = hmatvec(M11, b1, max_p) + np.dot(M12, b2)
        #result2 = hmatvec(M21, b1, max_p) + np.dot(M22, b2)
        result1 = hmatvec(M11, b1, max_p) + aca_matvec(M12, b2, M12//2)
        result2 = hmatvec(M21, b1, max_p) + aca_matvec(M22, b2, M22//2)

        return np.concatenate((result1, result2))