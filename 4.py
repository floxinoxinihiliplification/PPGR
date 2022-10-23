import numpy as np
from numpy import linalg as LA
import math as M 

n = 5

def ParametriKamere(T):
    #Koordinate centra kamere se odrede kao TC = 0
    det1 = np.array([
        [T[0][1], T[0][2], T[0][3]],
        [T[1][1], T[1][2], T[1][3]],
        [T[2][1], T[2][2], T[2][3]]
    ])
    c1 = LA.det(det1)
    det2 = np.array([
        [T[0][0], T[0][2], T[0][3]],
        [T[1][0], T[1][2], T[1][3]],
        [T[2][0], T[2][2], T[2][3]]
    ])
    c2 = -LA.det(det2)
    det3 = np.array([
        [T[0][0], T[0][1], T[0][3]],
        [T[1][0], T[1][1], T[1][3]],
        [T[2][0], T[2][1], T[2][3]]
    ])
    c3 = LA.det(det3)
    det4 = np.array([
        [T[0][0], T[0][1], T[0][2]],
        [T[1][0], T[1][1], T[1][2]],
        [T[2][0], T[2][1], T[2][2]]
    ])
    c4 = -LA.det(det4)

    [c1, c2, c3] = [c1, c2, c3] / c4
    
    C = np.array([np.round(c1, 6), np.round(c2, 6), np.round(c3, 6)])
    
    T0 = T[:, :3]
    
    if LA.det(T0) < 0:
        T = -T
        T0 = T[:, :3]
    
    [Q, R] = LA.qr(LA.inv(T0))
    if R[0, 0] < 0:
        R[0, :] = -R[0, :]
        Q[:, 0] = -Q[:, 0]
        
    if R[1, 1] < 0:
        R[1, :] = -R[1, :]
        Q[:, 1] = -Q[:, 1]
    
    if R[2, 2] < 0:
        R[2, :] = -R[2, :]
        Q[:, 2] = -Q[:, 2]
    
    K = np.round(LA.inv(R), 6)
    
    if K[2][2] != 1:
        K = K / K[2][2]
        
    A = np.round(LA.inv(Q), 6)
    
    return K, A, C

def matrix(x, xp):
    A = np.array([  [0, 0, 0, 0, -xp[2]*x[0], -xp[2]*x[1], -xp[2]*x[2], -xp[2]*x[3], xp[1]*x[0], xp[1]*x[1], xp[1]*x[2], xp[1]*x[3]],
                    [xp[2]*x[0], xp[2]*x[1], xp[2]*x[2], xp[2]*x[3], 0, 0, 0, 0, -xp[0]*x[0], -xp[0]*x[1], -xp[0]*x[2], -xp[0]*x[3]]])
    return A

def CameraDLP(originali, projekcije):

    M = np.zeros((12, 12))
    i = 0
    for (x, xp) in zip(originali, projekcije):
        M[i:i+2] = matrix(x, xp)
        i = i+2

    _, _, Vt = LA.svd(M, full_matrices=True)

    V = np.zeros((3,4))

    V[0] = Vt[11][0:4]
    V[1] = Vt[11][4:8]
    V[2] = Vt[11][8:]

    V = V / V[0][0]
    V = np.round(V, 6)

    return V

if __name__ == "__main__":
    
    T = np.array([  [5,     -1-(2*n),   3,      18-(3*n)],
                    [0,     -1,         5,      21],
                    [0,     -1,         0,      1]])
    #T = np.array([  [1, 5, 7, -2],
    #                [3, 0, 2, 3],
    #                [4, -1, 0, 1]])

    K, A, C = ParametriKamere(T)
    print("K:\n", K, "\n")
    print("A:\n", A, "\n")
    print("C:\n", C, "\n")

    M1 = np.array([460, 280, 250, 1])
    M2 = np.array([50, 380, 350, 1])
    M3 = np.array([470, 500, 100, 1])
    M4 = np.array([380, 630, 50 * n, 1])
    M5 = np.array([30 * n, 290, 0, 1])
    M6 = np.array([580, 0, 130, 1])
       
    M1p = np.array([288, 251, 1])
    M2p = np.array([79, 510, 1])
    M3p = np.array([470, 440, 1])
    M4p = np.array([520, 590, 1])
    M5p = np.array([365, 388, 1])
    M6p = np.array([365, 20, 1])
    #M1 = np.array([30,250,195,1])
    #M2 = np.array([30,250,0,1])
    #M3 = np.array([175,145,85,1])
    #M4 = np.array([125,20,60,1])
    #M5 = np.array([290,125,60,1])
    #M6 = np.array([290,20,0,1])
#
    #M1p = np.array([679,86,1])
    #M2p = np.array([612,343,1])
    #M3p = np.array([461,283,1])
    #M4p = np.array([321,254,1])
    #M5p = np.array([325,419,1])
    #M6p = np.array([179,432,1])
    originali = np.array([M1, M2, M3, M4, M5, M6])
    projekcije = np.array([M1p, M2p, M3p, M4p, M5p, M6p])

    V = CameraDLP(originali, projekcije)
    np.set_printoptions(suppress=True)
    print("V:\n", V)