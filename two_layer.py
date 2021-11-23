# Source: https://colab.research.google.com/drive/1laqL8AuCpfMq044FTbRM7QlmLhMcgtH4?authuser=2#scrollTo=a4ud48OJPfeU

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_A(W, X, l):

    B = W @ X @ X.T @ W.T + l * np.eye(W.shape[0])
    A = W.T @ np.linalg.inv(B) @ W @ X

    return A

def EM2_EtrM_EA2_given_X(X, l, n, d, p, n_sample = 20):

    EM2 = 0
    EtrM = 0
    EA2 = 0
    for i in range(n_sample):
        W = np.random.normal(0, 1/d, size=(p, d))
        A = compute_A(W, X, l)
        M = A @ X.T
        EM2 = EM2 + np.linalg.norm(M)**2
        EtrM = EtrM + np.trace(M)
        EA2 = EA2 + np.linalg.norm(A)**2
    EA2 = EA2/n_sample
    EtrM = EtrM/n_sample
    EM2 = EM2/n_sample
    return EM2, EtrM, EA2

def generate_X(n, d):

    X = np.zeros((d, n))
    for i in range(n):
        X[:, i] = np.random.multivariate_normal(np.zeros(d), np.eye(d)/d)
    return X

def EM2_EtrM_EA2(l, n, d, p, n_sample = 25):

    EM2 = 0
    EtrM = 0
    EA2 = 0
    for i in range(n_sample):
        X = generate_X(n, d)
        EM2_tmp, EtrM_tmp, EA2_tmp = EM2_EtrM_EA2_given_X(X, l, n, d, p)
        EM2 = EM2 + EM2_tmp
        EtrM = EtrM + EtrM_tmp
        EA2 = EA2 + EA2_tmp

    EM2 = EM2/n_sample
    EtrM = EtrM/n_sample
    EA2 = EA2/n_sample

    return EM2, EtrM, EA2

# from google.colab import drive
# drive.mount('/content/gdrive')

l, n, d = 0.01, 800, 30

Risk_clean = []
EA2 = []

# list_p = [2, 5, 10, 50, 100, 200, 300, 400, 600, 800, 1000, 1200, 1500, 2000]
list_p = [i*5 for i in range(3, 10)] + [i*5 for i in range(11, 20)]

for p in list_p:

    EM2_tmp, EtrM_tmp, EA2_tmp = EM2_EtrM_EA2(l, n, d, p)
    EA2.append(EA2_tmp)
    Risk_clean.append((EM2_tmp -2*EtrM_tmp + d)/d)
    print('p={}: Risk={}, EA2={}'.format(p, (EM2_tmp -2*EtrM_tmp + d)/d, EA2_tmp))

# np.savetxt('/content/gdrive/My Drive/bias-variance-trade-off/results/numerical_two_layer/Risk_clean_2.csv', Risk_clean, delimiter=",")
# np.savetxt('/content/gdrive/My Drive/bias-variance-trade-off/results/numerical_two_layer/EA2_2.csv', EA2, delimiter=",")

EA2_1 = np.loadtxt('/content/gdrive/My Drive/bias-variance-trade-off/results/numerical_two_layer/EA2.csv', delimiter=",")
EA2_2 = np.loadtxt('/content/gdrive/My Drive/bias-variance-trade-off/results/numerical_two_layer/EA2_2.csv', delimiter=",")
Risk_clean_1 = np.loadtxt('/content/gdrive/My Drive/bias-variance-trade-off/results/numerical_two_layer/Risk_clean.csv', delimiter=",")
Risk_clean_2 = np.loadtxt('/content/gdrive/My Drive/bias-variance-trade-off/results/numerical_two_layer/Risk_clean_2.csv', delimiter=",")

list_p_1 = [2, 5, 10, 50, 100, 200, 300, 400, 600, 800, 1000, 1200, 1500, 2000]
list_p_2 = [i*5 for i in range(3, 10)] + [i*5 for i in range(11, 20)]

list_sigma = [0, 0.01, 0.02, 0.05, 1, 2, 2.5, 5]
# list_sigma = [5]
p_start = 0

EA2 = np.concatenate((EA2_1, EA2_2))
Risk_clean = np.concatenate((Risk_clean_1, Risk_clean_2))
list_p = list_p_1 + list_p_2

for sigma in list_sigma:
    Risk_tmp = []
    for i in range(p_start, len(list_p)):
        Risk_tmp.append(Risk_clean[i] + (sigma**2)*EA2[i]/d)
    plt.figure()
    plt.scatter(list_p[p_start:], Risk_tmp, label = '{}'.format(sigma))

    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.show()

plt.figure()
plt.scatter(list_p, Risk_clean)
plt.yscale('log')
plt.show()

plt.figure()
plt.scatter(list_p, EA2)
plt.yscale('log')
plt.show()






