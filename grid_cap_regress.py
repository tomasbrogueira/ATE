import numpy as np
import matplotlib.pyplot as plt

np.random.seed(98)
m = 500
j = 1j

# injeções nodais iniciais
S = -np.array([0.224, 0.708, 1.572, 0.072]) * np.exp(j * 0.3176)
I = np.conj(S).reshape(-1,1)  # (4,1)

# admitâncias
y12 = 1 - j*10
y13 = 2 * y12
y23 = 3 - j*20
y34 = y23
y45 = 2 * y12

Y_full = np.array([
    [y12+y13,  -y12,     -y13,      0,     0],
    [   -y12, y12+y23,   -y23,      0,     0],
    [   -y13,   -y23, y13+y23+y34, -y34,   0],
    [      0,      0,      -y34, y34+y45, -y45]
], dtype=complex)
Y = Y_full[:4,:4]  # retirar coluna do slack

e4 = np.random.randn(4, m) * 0.25
e1 = np.random.randn(m) * 0.5
i1w = [I[0,0].real]

i12, i13, i23 = [], [], []

# simulação do circuito
for t in range(m-1):
    next_I = 0.65 * I[:, t:t+1] + e4[:, t:t+1]
    i1w.append(0.75 * i1w[-1] + e1[t])
    next_I[0,0] = -i1w[-1] + j * next_I[0,0].imag

    I = np.hstack((I, next_I))
    
    v = 1 + np.linalg.inv(Y) @ I[:, t+1]

    I12 = y12 * (v[0] - v[1])
    I13 = y13 * (v[0] - v[2])
    I23 = y23 * (v[1] - v[2])
    i12.append(np.abs(I12) * np.sign(I12.real))
    i13.append(np.abs(I13) * np.sign(I13.real))
    i23.append(np.abs(I23) * np.sign(I23.real))

Ii = np.abs(I) * np.sign(I.real)  # (4,m)

# plots
plt.figure(figsize=(10,9))

plt.subplot(3,1,1)
plt.plot(I.real.T)
plt.title('Injeções nodais (Parte Real)')
plt.xlabel('Time stamp')
plt.ylabel('I [pu]')
plt.legend([f'Nó {i+1}' for i in range(I.shape[0])])

plt.subplot(3,1,2)
plt.plot(I.imag.T, '--')
plt.title('Injeções nodais (Parte Imaginária)')
plt.xlabel('Time stamp')
plt.ylabel('I [pu]')
plt.legend([f'Nó {i+1}' for i in range(I.shape[0])])

plt.subplot(3,1,3)
lines = np.vstack([i12, i13, i23]).T
plt.plot(lines)
plt.title('Correntes nos ramos')
plt.xlabel('Time stamp')
plt.ylabel('I_branch [pu]')
plt.legend(['12','13','23'])

plt.tight_layout()
plt.show()

# detetar violações e OLS
rate_12, rate_13, rate_23 = 0.5 , 1.0, 0.25
ij = lines
X, y = [], []
for t in range(len(ij)):
    pos_vio = max(ij[t,0]-rate_12, ij[t,1]-rate_13, ij[t,2]-rate_23)
    if pos_vio > 0:
        X.append(Ii[:,t])
        y.append(pos_vio)
X = np.column_stack([X, np.ones(len(X))])
y = np.array(y).reshape(-1,1)

beta = np.linalg.inv(X.T @ X) @ X.T @ y
res = y - X @ beta

print("Coeficientes estimados:", beta.flatten())
print("Erro médio (norm/k):", np.linalg.norm(res)/len(y))
