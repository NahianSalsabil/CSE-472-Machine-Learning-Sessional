import numpy as np

n = input("n: ")
m = input("m: ")
n = int(n)
m = int(m)
A = np.random.randint(100, size=(n,m))

# print(f'A: ', A)

U, D, V_T = np.linalg.svd(A)

# print(f'U: ', U)
# print(f'D: ', D)
# print(f'V_T: ', V_T)

A_plus = np.linalg.pinv(A)
# print(f'A_plus: ', A_plus)

V = V_T.T
U_T = U.T
# print(f'V: ', V, f'\nU_T: ', U_T)
D_diag = np.diag(D)

# print(f'D_diag: \n', D_diag)

## Reciprocal
for i in range(0, len(D_diag[0])):
    for j in range(0, len(D_diag)):
        if i == j:
            D_diag[i][j] = 1/D_diag[i][j]

## Padding zero to obtain n x m matrix
if n < m:  # add column
    zero = np.zeros((n,1))
    i = n
    while i != m:
        D_diag = np.append(D_diag, zero, axis=1)
        i += 1
    # print(f'D_diag: \n', D_diag)

elif n > m:   # add row
    zero = np.zeros((m,1))
    i = m
    while i != n:
        D_diag = np.append(D_diag, zero.T, axis = 0)
        i += 1
    # print(f'D_diag: \n', D_diag)

D_diag_T = D_diag.T
# print(f'D_diag_T: \n', D_diag_T)

A_plus_ = V.dot(D_diag_T).dot(U_T)
# print(f'A_plus_: ', A_plus_)

if np.allclose(A_plus, A_plus_):
    print("Reconstruction Successful!!!")

    