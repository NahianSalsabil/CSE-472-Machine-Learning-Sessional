import numpy as np

n = input("dimension: ")
n = int(n)
A = np.random.randint(100, size=(n,n))
while np.linalg.det(A) == 0:
    A = np.random.randint(100, size=(n,n))

print(f'A: ', A)

eigen_values, eigen_vectors = np.linalg.eig(A)   # Lambda, V

print(f'eigen_values: ', eigen_values)
print(f'eigen_vectors: ', eigen_vectors)

a = np.zeros((n, n), int);
diag_Lambda = np.diag(eigen_values)
print(f'diag_Lambda: ', diag_Lambda)

v_inv = np.linalg.inv(eigen_vectors)
print(f'v_inv: ', v_inv)

A_ = eigen_vectors.dot(diag_Lambda.dot(v_inv))
print(f'A_: ', A_)

if np.allclose(A, A_):
    print("Reconstruction Successful!!!")