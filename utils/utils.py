def batch_mean_update(n1: int,
                      m1: float, 
                      n2: int, 
                      m2: float) -> float:
    '''
    computes the running mean of two batches

    args:
        n1: number of items in batch 1
        m1: mean of batch 1
        n2: number of items in batch 2
        m2: mean of batch 2
    
    returns:
        global mean
    '''
    return (n1 * m1 +  n2 * m2) / (n1 + n2)

def batch_var_update(n1: int,
                     m1: float,
                     v1: float,
                     n2: int,
                     m2: float,
                     v2: float,
                     ddof=0) -> float:
    '''
    computes the running variance of two batches

    args:
        n1: number of items in batch 1
        m1: mean of batch 1
        v1: variance of batch 1
        n2: number of items in batch 2
        m2: mean of batch 2
        v2: variance of batch 2
        ddof: delta degree of freedom
    
    returns:
        global variance
    '''
    return ((n1 - ddof) * v1 + (n2 - ddof) * v2) / (n1 + n2 - ddof) + \
           (n1 * n2 * (m1 - m2) **2) / (n1 + n2) * (n1 + n2 - ddof)