import amg
import gauss_seidel as gs
import numpy as np
import matplotlib.pyplot as plt

from pyamg.gallery import load_example

        
ex = load_example('recirc_flow')

A = ex['A']
vertices = ex['vertices']

n = A.shape[0]
print(vertices.shape)
print(A.shape)

A = np.array(A.todense()) # convert to dense for simplicity
b = np.ones(n)

x_0 = np.ones(n)
x_true = np.linalg.solve(A, b)
orig_error = np.abs(x_0 - x_true)

# params
d = 4
relax = 1
iters = 20

for i in range(iters):
    x_0 = amg.v_cycle(A, b, x_0, depth=d, relax_iterations=relax)
    
    
error = np.abs(x_0 - x_true)

# plot error
plt.figure(figsize=(12, 5))
plt.plot(orig_error, label='Original error')
plt.plot(error, label='Error after AMG V-cycle')
plt.legend()
title_string = "AMG V-cycle error with depth " + str(d) + " and relax iters " + str(relax)
plt.title(title_string)
plt.xlabel('Node index')
plt.ylabel('Error') 
plt.show()
