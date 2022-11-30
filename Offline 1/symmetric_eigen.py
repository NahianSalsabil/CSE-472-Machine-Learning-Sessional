import numpy as np

n = input("dimension: ")
n = int(n)
A = np.random.randint(-100, 100, size=(n,n))
A_symm = (A + A.T)/2
while np.linalg.det(A) == 0:
    A = np.random.randint(-100, 100, size=(n,n))
A_symm = (A + A.T)/2

print(f'A_symm: ', A_symm)

eigen_values, eigen_vectors = np.linalg.eig(A_symm)   # Lambda, V

print(f'eigen_values: ', eigen_values)
print(f'eigen_vectors: ', eigen_vectors)

a = np.zeros((n, n), int);
diag_Lambda = np.diag(eigen_values)
print(f'diag_Lambda: ', diag_Lambda)

v_inv = np.linalg.inv(eigen_vectors)
print(f'v_inv: ', v_inv)

A_ = eigen_vectors.dot(diag_Lambda.dot(v_inv))
print(f'A_: ', A_)

if np.allclose(A_symm, A_):
    print("Reconstruction Successful!!!")