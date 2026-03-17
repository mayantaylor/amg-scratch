
import gauss_seidel as gs
import numpy as np
import matplotlib.pyplot as plt

def strong_dependents(S, i):
    return np.where(S[:, i])[0]

def strong_influencers(S, i):
    return np.where(S[i, :])[0]

def weak_influencers(S, i, A):
    n = A.shape[1]
    result = []
    for j in range(n):
        if j == i:
            continue
        if A[i, j] != 0 and S[i, j] == 0:
            result.append(j)
    return result
 
def strong_connections(A, theta=0.25):
    # TODO: should probably store S in a sparse format
    
    n = A.shape[0]
    S = np.zeros((n, n), dtype=bool)
    for i in range(n):
        # compute max over row (excluding A_ii), for i's strongest connection
        max_connection = np.max(np.abs(A[i, np.arange(n) != i]))
        
        # determine if every connection to i is strong or not
        for j in range(n):
            if A[i, j] < -theta * max_connection:
                S[i, j] = True
    return S

def plot_cf_split(C, vertices):
    plt.figure(figsize=(8, 6))

    for i in range(n):
        if C[i]:
            plt.scatter(vertices[i, 0], vertices[i, 1], color='red', label='Coarse' if 'Coarse' not in plt.gca().get_legend_handles_labels()[1] else "")
        else:
            plt.scatter(vertices[i, 0], vertices[i, 1], color='blue', label='Fine' if 'Fine' not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.title('Ruge-Stuben C/F Splitting')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid()
    plt.show()
    
# ruge stuben is approach 1 (build C and F sets). aggregation would be approach 2
def ruge_stuben(S):
    n = S.shape[0]
    decision = np.zeros(n, dtype=bool)
    coarse_nodes = np.zeros(n, dtype=bool)
    
    # row i of S is the strong influencers of node i
    # column j of S is the strong dependents of node i
    
    coarse_power = np.zeros(n)  # coarse power measures how many UNDECIDED nodes strongly depend on this node.
    
    for i in range(n):
        # compute number of nodes that strongly depend on i
        coarse_power[i] = np.sum(S[:, i])

    while (np.sum(~decision) > 0):
        # pick node with max coarse_power to make strong
        m = np.argmax(coarse_power)
        if decision[m]:
            break
        
        decision[m] = True
        coarse_power[m] = -1 # this node is now decided, so it shouldnt be selected anymore
        coarse_nodes[m] = True
                        
        # let all strong dependents of this node be weak
        for j in strong_dependents(S, m):
            if not decision[j]:
                decision[j] = True
                coarse_power[j] = -1 # this node is now decided, so it shouldnt be selected anymore
                coarse_nodes[j] = False
                
                for k in strong_dependents(S, j): # all dependents have more coarse power (because they depend on a weak node?)
                    if not decision[k]:
                        coarse_power[k] += 1
                        
                for k in strong_influencers(S, j): # all influencers have less coarse power (because a dependent is no longer undecided)
                    if not decision[k]:
                        coarse_power[k] -= 1
        
    return coarse_nodes

def build_interpolation_matrix(A, C, S):
    # P is tall and skinny. takes coarse grid to full grid
    coarse_indices = np.where(C)[0]
    fine_indices = np.where(~C)[0]
    
    fine_to_coarse = np.cumsum(C) - 1
        
    P = np.zeros((len(C), len(coarse_indices)))
    
    for idx, i in enumerate(coarse_indices):
        P[i, idx] = 1
        
    for i in fine_indices:    
        strong_C_of_i = set(k for k in strong_influencers(S, i) if C[k])
    
        for k in strong_C_of_i: # only sum over strong influencers of i
            
            # numerator
            # a[i,k] + sum over j strong FINE influencers of i of (a[i,j] * a[j, k] / m in C_i and C_j of a[j, m])
            numerator = A[i, k]
            for j in strong_influencers(S, i):
                if not C[j]: # only sum over strong influencers that are fine nodes
                    num = A[i, j] * A[j, k]
                    
                    denom = 0
                    for m in strong_influencers(S, j):
                        if m in strong_C_of_i: # only sum over strong influencers of j that are also strong influencers of i and are fine nodes
                            denom += A[j, m]
                    
                    if denom != 0:
                        numerator += num / denom

            # denominator
            # a[i, i] + sum over j weak influencers of a[i, j]
            # where weak influencers are... neighbors - strong neighbors
            denominator = A[i, i]
            for j in weak_influencers(S, i, A):
                if C[j]: # only sum over weak influencers that are coarse nodes
                    denominator += A[i, j]  
                                
            P[i, fine_to_coarse[k]] = - numerator / denominator
        
    # compute row sums and rescale so that row sums of P are 1 (so that constant vectors are preserved)
    row_sums = P.sum(axis=1)
    for i in range(P.shape[0]):
        if row_sums[i] != 0:
            P[i, :] /= row_sums[i]
            
    
    return P

def setup_phase(A, theta):
    S = strong_connections(A, theta)        
    C = ruge_stuben(S)

    #plot_cf_split(C, vertices)

    P = build_interpolation_matrix(A, C, S) # P
    R = P.T
    
    return [P, R]

def v_cycle(A, b, x_0, theta=.25, depth=1, relax_iterations=1):
    
    if depth == 0:
        return np.linalg.solve(A, b)

    # pre-relaxation
    x_0 = gs.solve(A, b, x_0, max_iterations=relax_iterations)
    
    # setup
    [P, R] = setup_phase(A, theta)
    
    # fine to coarse
    n = A.shape[0]
    r = b - A @ x_0
    
    coarse_r = R @ r
    initial_guess = np.zeros(coarse_r.shape)
    coarse_A = R @ A @ P
    
    # recursive call
    error = v_cycle(coarse_A, coarse_r, initial_guess, theta, depth - 1)
    
    # coarse to fine
    fine_error = P @ error
    
    # post relaxation
    x = x_0 + fine_error
    x = gs.solve(A, b, x, max_iterations=relax_iterations)
    
    return x
