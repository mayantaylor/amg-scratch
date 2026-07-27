### generated matrices descriptor and reproducers

| matrix class | dimension | density | reproducer |
| A | 65536 | .001 | python make_matrix.py A.npz 65536 65536 diagonal_spectrum --nnz 4294967 --generate-spectrum5 --seed 1 |
| spectrum | 65536 | .0001 | python make_matrix.py A.npz 65536 65536 diagonal_spectrum --nnz 4294967 --generate-spectrum5 --seed 1 |
| 1percent | 65536 | .01 | python generate.py 1percent.npz 65536 65536 diagonal_spectrum --nnz 42949673 --generate-spectrum5 --seed 1 | 
| rand | variable | .01 | python generate.py matrices/rand.131072.01.npz 131072 131072 random --density .01 --seed 1

diag matrices generated with spread = 0, seed 1