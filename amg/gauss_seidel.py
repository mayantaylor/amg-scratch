import numpy as np

def solve(A, b, x=None, tol=1e-10, max_iterations=1000):
    [L, U] = np.tril(A), np.triu(A, 1)
    n = len(b)
    
    if x is None:
        x = np.zeros(n)
    
    for k in range(max_iterations):
        x = gauss_seidel_step(A, b, x)
        
        if np.linalg.norm(A @ x - b, ord=np.inf) < tol:
            print(f"Converged in {k+1} iterations.")
            break
            
    return x

def gauss_seidel_step(A, b, x=None):
    n = len(b)
    if x is None:
        x = np.zeros(n)

    [L, U] = np.tril(A), np.triu(A, 1)
    temp = b - U @ x
    x = np.linalg.solve(L, temp)
    
    return x
