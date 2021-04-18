import numpy as np
import copy
from math import *
import time
import matplotlib.pyplot as plt

#------------------------1. Décomposition de Cholesky------------------------

#                           --- QUESTION 1 ---

#Géneré M inversible
# MT = np.transpose(M)
# A = MTM

## Verification Symétrique / Définie / Positive
#def Vérification

#A = np.array([[2, 5, 6],[4, 11, 9],[-2, -8, 7]])
#B = np.array([[7], [12], [3]])


def Cholesky(A):
    n,n = np.shape(A)
    L = np.zeros((n,n))

    for i in range(0, n):
        S = 0

        for j in range(0, i):
            S += L[i, j] ** 2
        L[i,i] = sqrt(A[i, i] - S)

        for k in range(i + 1, n):
            S = 0

            for j in range(0, i):
                S += L[k, j] * L[i, j]

            L[k, i] = (A[k, i] - S) / L[i, i]
        
    return L




#----2. Résolution de systèmes à l’aide de la décomposition de Cholesky-----

#                           --- QUESTION 1 ---

def ResolCholesky(A,B):
    n, n = np.shape(A)
    B = B.reshape(n,1)
    L = Cholesky(A)
    Lt = np.transpose(L)
    x = np.zeros(n)
    y = []

    #Pour: Ly = b
    for i in range(0,n):
        y.append(B[i])

        for k in range(0,i):
            y[i] = y[i] - L[i, k] * y[k]
        y[i] = y[i] / L[i,i]

    #Pour: LTx = y
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(Lt[i, i + 1:], x[i + 1:])) / Lt[i, i]

    return x


#----------------------3. Expérimentation des méthodes----------------------

#                           --- QUESTION 1 ---

def Mat_Positive(n):
    for i in range(n):
        A = np.random.rand(n, n)
        At = np.transpose(A)
        M = np.dot(A,At)
        return M


#                           --- GAUSS ---

def ReductionGauss(Au):
    n, m = np.shape(Au)
    for i in range(0, n-1):
        if Au[i,i] == 0 :
            Au[i,:] = Au[i+1]

        else :
            for j in range(i+1,n):
                g = Au[j, i] / Au[i, i]
                Au[j,:] = Au[j,:] - g * Au[i,:]

    return Au

def ResolutionSystTriSup(Tu):
    n, m = np.shape(Tu)
    x = np.zeros(n)
    x[n-1] = Tu[n-1,m-1] / Tu[n-1,n-1]

    for i in range(n-2, -1, -1):
        x[i] = Tu[i,m-1]
        for j in range(i+1, n):
            x[i]= x[i] - Tu[i,j] * x[j]
        x[i] =  x[i] / Tu[i,i]

    return x

def Gauss(A, B):
    n, m = np.shape(A)
    B = B.reshape(n,1)
    Ared = np.column_stack((A, B))

    return ResolutionSystTriSup(ReductionGauss(Ared))

#                           --- LU ---

def DecompositionLU(A):
    n, n = np.shape(A)
    L = np.eye(n)
    U = np.copy(A)
    for i in range(0, n - 1):

        for j in range(i + 1, n):
            g = U[j, i] / U[i, i]
            L[j, i] = g
            U[j,:] = U[j,:] - g * U[i,:]

    return L, U


def ResolutionLU(L, U, B):
    n, n = np.shape(L)
    x = np.zeros(n)
    y = []

    for i in range(0, n):
        y.append(B[i])

        for k in range(0, i):
            y[i] = y[i] - L[i, k] * y[k]
        y[i] = y[i] / L[i, i]

    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x

def LU(A, B):
    n, n = np.shape(A)
    B = B.reshape(n, 1)
    L, U = DecompositionLU(A)
    return ResolutionLU(L, U, B)


#                       --- GRAPHIQUES ---

t_Gauss = []
t_LU = []
t_Cho = []
t_Cho_npl = []

E_Gauss = []
E_LU = []
E_Cho = []
E_Cho_npl = []

for n in range(1,502,50):
    A = Mat_Positive(n)
    B = np.random.rand(1, n)
    Bb = B.reshape(n, 1)
    C = np.copy(A)

    #Temps / Erreur -> Gauss
    start_Gauss = time.time()
    n_Gauss = Gauss(A, B)
    stop_Gauss = time.time()
    t_Gauss.append(stop_Gauss - start_Gauss)
    E_Gauss.append(np.linalg.norm(np.dot(A,n_Gauss)-np.ravel(B)))

    #Temps / Erreur -> LU
    start_LU = time.time()
    n_LU = LU(A,B)
    stop_LU = time.time()
    t_LU.append(stop_LU - start_LU)
    E_LU.append(np.linalg.norm(np.dot(A,n_LU)-np.ravel(B)))
    
    #Temps / Erreur -> Cholesky
    start_Cho = time.time()
    n_Cho = ResolCholesky(A,B)
    stop_Cho = time.time()
    t_Cho.append(stop_Cho - start_Cho)
    E_Cho.append(np.linalg.norm(np.dot(A,n_Cho)-np.ravel(B)))

    #Temps / Erreur -> Cholesky via linalg.cholesky
    start_Cho_npl = time.time()
    n_Cho_npl = np.linalg.cholesky(A)
    stop_Cho_npl = time.time()
    t_Cho_npl.append(stop_Cho_npl - start_Cho_npl)
#    E_Cho_npl.append(np.linalg.norm(np.dot(A,n_Cho_npl)))

size_n = [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

#                           --- GRAPH n/t ---

plt.style.use('dark_background')
plt.plot(size_n, t_Gauss, color='red', label = 'Gauss')
plt.plot(size_n, t_LU, color='lime', label = 'LU')
plt.plot(size_n, t_Cho, color='blue', label = 'Cholesky')
plt.plot(size_n, t_Cho_npl, color='cyan', label = 'linalg.cholesky')
plt.xlabel('Taille de la matrice (n)')
plt.ylabel('Temps (s)')
plt.legend(loc='upper left')
plt.title('Temps de calcul CPU en fonction du nombre de matrice n')
plt.show()

#                           --- GRAPH n/E ---

plt.style.use('dark_background')
plt.plot(size_n, E_Gauss, color='red', label = 'Gauss')
plt.plot(size_n, E_LU, color='lime', label = 'LU')
plt.plot(size_n, E_Cho, color='blue', label = 'Cholesky')
#plt.plot(size_n, E_Cho_npl, color='cyan', label = 'linalg.cholesky')
plt.xlabel('Taille de la matrice (n)')
plt.ylabel('||AX - B||')
plt.legend(loc='upper left')
plt.title("Estimation de l'erreur commise")
plt.show()