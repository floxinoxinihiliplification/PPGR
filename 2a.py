import numpy as np 
from numpy import linalg as LA
import math as m
import matplotlib.pyplot as plt
import cv2

def naiveAlg(x1, x2, x3, x4, x1p, x2p, x3p, x4p):

    A0 = np.array([1,0,0])
    B0 = np.array([0,1,0])
    C0 = np.array([0,0,1])
    D0 = np.array([1,1,1])

    D  = np.array([[x1[0], x2[0], x3[0]], [x1[1], x2[1], x3[1]], [x1[2], x2[2], x3[2]]])
    D1 = np.array([[x4[0], x2[0], x3[0]], [x4[1], x2[1], x3[1]], [x4[2], x2[2], x3[2]]])
    D2 = np.array([[x1[0], x4[0], x3[0]], [x1[1], x4[1], x3[1]], [x1[2], x4[2], x3[2]]])
    D3 = np.array([[x1[0], x2[0], x4[0]], [x1[1], x2[1], x4[1]], [x1[2], x2[2], x4[2]]])

    det = round(LA.det(D))
    a = round(LA.det(D1))/det
    b = round(LA.det(D2))/det
    c = round(LA.det(D3))/det

    P1 = np.zeros((3,3))
    P1[:,0] = a*x1.T
    P1[:,1] = b*x2.T
    P1[:,2] = c*x3.T

    Dp  = np.array([[x1p[0], x2p[0], x3p[0]], [x1p[1], x2p[1], x3p[1]], [x1p[2], x2p[2], x3p[2]]])
    D1p = np.array([[x4p[0], x2p[0], x3p[0]], [x4p[1], x2p[1], x3p[1]], [x4p[2], x2p[2], x3p[2]]])
    D2p = np.array([[x1p[0], x4p[0], x3p[0]], [x1p[1], x4p[1], x3p[1]], [x1p[2], x4p[2], x3p[2]]])
    D3p = np.array([[x1p[0], x2p[0], x4p[0]], [x1p[1], x2p[1], x4p[1]], [x1p[2], x2p[2], x4p[2]]])

    detp = round(LA.det(Dp))
    ap = round(LA.det(D1p))/detp
    bp = round(LA.det(D2p))/detp
    cp = round(LA.det(D3p))/detp

    P2 = np.zeros((3,3))
    P2[:,0] = ap*x1p.T
    P2[:,1] = bp*x2p.T
    P2[:,2] = cp*x3p.T

    P1_inv = LA.inv(P1)

    P = np.dot(P2, P1_inv)
    if P[0][0] != 0:
        P = P / P[0][0]
    P = P.round(5)

    return P  #/7 za njihov primer

def DLTmatrix(x, xp):
    A = np.array([[0, 0, 0, -xp[2]*x[0], -xp[2]*x[1], -xp[2]*x[2], xp[1]*x[0], xp[1]*x[1], xp[1]*x[2]], [xp[2]*x[0], xp[2]*x[1], xp[2]*x[2], 0, 0, 0, -xp[0]*x[0], -xp[0]*x[1], -xp[0]*x[2]]])
    return A

def DLT4(x1, x2, x3, x4, x1p, x2p, x3p, x4p):
    A1 = DLTmatrix(x1, x1p)
    A2 = DLTmatrix(x2, x2p)
    A3 = DLTmatrix(x3, x3p)
    A4 = DLTmatrix(x4, x4p)

    A = np.zeros((8, 9))
    A[0:2] = A1
    A[2:4] = A2
    A[4:6] = A3
    A[6:] = A4

    U, D, Vt = LA.svd(A, full_matrices=True)

    P = np.zeros((3,3))
    P[0] = Vt[8][0:3]
    P[1] = Vt[8][3:6]
    P[2] = Vt[8][6:]

    if P[0][0] != 0:
        P = P / P[0][0]
    P = P.round(5)

    return P

def DLT(x1, x2, x3, x4, x5, x1p, x2p, x3p, x4p, x5p):
    A1 = DLTmatrix(x1, x1p)
    A2 = DLTmatrix(x2, x2p)
    A3 = DLTmatrix(x3, x3p)
    A4 = DLTmatrix(x4, x4p)
    A5 = DLTmatrix(x5, x5p)

    A = np.zeros((10, 9))
    A[0:2] = A1
    A[2:4] = A2
    A[4:6] = A3
    A[6:8] = A4
    A[8:] = A5

    U, D, Vt = LA.svd(A, full_matrices=True)

    P = np.zeros((3,3))
    P[0] = Vt[8][0:3]
    P[1] = Vt[8][3:6]
    P[2] = Vt[8][6:]

    if P[0][0] != 0:
        P = P / P[0][0]
    P = P.round(5)

    return P

def afinize(x):
    new = np.zeros(2)
    new[0] = x[0]/x[2]
    new[1] = x[1]/x[2]

    return new

def f(x):
    x[0] = x[0]/x[2]
    x[1] = x[1]/x[2]
    x[2] = 1

    return x

def homogenize(x):
    new = np.zeros(3)
    new[0] = x[0]
    new[1] = x[1]
    new[2] = 1

    return new

def normalize(x1, x2, x3, x4, x5):

    x1a = afinize(x1)
    x2a = afinize(x2)
    x3a = afinize(x3)
    x4a = afinize(x4)
    x5a = afinize(x5)

    c = (x1a + x2a + x3a + x4a + x5a)/5

    G = np.eye(3)
    G[0][2] = -c[0]
    G[1][2] = -c[1]

    y1 = x1a-c
    y2 = x2a-c
    y3 = x3a-c
    y4 = x4a-c
    y5 = x5a-c

    d = (LA.norm(y1) + LA.norm(y2) + LA.norm(y3) + LA.norm(y4) + LA.norm(y5))/5
    l = m.sqrt(2)/d

    S = np.eye(3)
    S[0][0] = l
    S[1][1] = l

    T = np.dot(S, G)

    return T, np.dot(T, x1), np.dot(T, x2), np.dot(T, x3), np.dot(T, x4), np.dot(T, x5)

def modifiedDLT(x1, x2, x3, x4, x5, x1p, x2p, x3p, x4p, x5p):
    T, x1m, x2m, x3m, x4m, x5m = normalize(x1, x2, x3, x4, x5)
    Tp, x1pm, x2pm, x3pm, x4pm, x5pm = normalize(x1p, x2p, x3p, x4p, x5p)

    Pp = DLT(x1m, x2m, x3m, x4m, x5m, x1pm, x2pm, x3pm, x4pm, x5pm)

    Tp_inv = LA.inv(Tp)

    P = np.dot(np.dot(Tp_inv, Pp), T)

    if P[0][0] != 0:
        P = P / P[0][0]
    P = P.round(5)

    return P #/1.5 za njihov primer   

def modifiedDLT4(x1, x2, x3, x4, x5, x1p, x2p, x3p, x4p, x5p):
    T, x1m, x2m, x3m, x4m, x5m = normalize(x1, x2, x3, x4, x5)
    Tp, x1pm, x2pm, x3pm, x4pm, x5pm = normalize(x1p, x2p, x3p, x4p, x5p)

    Pp = DLT4(x1m, x2m, x3m, x4m, x1pm, x2pm, x3pm, x4pm)

    Tp_inv = LA.inv(Tp)

    P = np.dot(np.dot(Tp_inv, Pp), T)

    if P[0][0] != 0:
        P = P / P[0][0]
    P = P.round(5)

    return P #/1.5 za njihov primer  

def projection(x1, x2, x3, x4, x5, x1p, x2p, x3p, x4p, x5p, P):
    x1new = np.dot(P, x1p)
    x2new = np.dot(P, x2p)
    x3new = np.dot(P, x3p)
    x4new = np.dot(P, x4p)
    x5new = np.dot(P, x5p)

    return x1new, x2new, x3new, x4new, x5new

def graph5(x1, x2, x3, x4, x5, x1p, x2p, x3p, x4p, x5p):
    xs = [x1[0], x2[0], x3[0], x4[0], x5[0], x1[0]]
    ys = [x1[1], x2[1], x3[1], x4[1], x5[1], x1[1]]

    xps = [x1p[0], x2p[0], x3p[0], x4p[0], x5p[0], x1p[0]]
    yps = [x1p[1], x2p[1], x3p[1], x4p[1], x5p[1], x1p[1]]

    P1 = DLT(x1, x2, x3, x4, x5, x1p, x2p, x3p, x4p, x5p)
    x1n, x2n, x3n, x4n, x5n = projection(x1, x2, x3, x4, x5, x1p, x2p, x3p, x4p, x5p, P1)
    P2 = modifiedDLT(x1, x2, x3, x4, x5, x1p, x2p, x3p, x4p, x5p)
    x1nm, x2nm, x3nm, x4nm, x5nm = projection(x1, x2, x3, x4, x5, x1p, x2p, x3p, x4p, x5p, P2)
    P = naiveAlg(x1, x2, x3, x4, x1p, x2p, x3p, x4p)
    x1nn, x2nn, x3nn, x4nn, x5nn = projection(x1, x2, x3, x4, x5, x1p, x2p, x3p, x4p, x5p, P)

    xns = [x1n[0], x2n[0], x3n[0], x4n[0], x5n[0], x1n[0]]
    yns = [x1n[1], x2n[1], x3n[1], x4n[1], x5n[1], x1n[1]]

    xnms = [x1nm[0], x2nm[0], x3nm[0], x4nm[0], x5nm[0], x1nm[0]]
    ynms = [x1nm[1], x2nm[1], x3nm[1], x4nm[1], x5nm[1], x1nm[1]]

    xnns = [x1nn[0], x2nn[0], x3nn[0], x4nn[0], x5nn[0], x1nn[0]]
    ynns = [x1nn[1], x2nn[1], x3nn[1], x4nn[1], x5nn[1], x1nn[1]]

    plt.plot(xs, ys)
    plt.plot(xps, yps)
    plt.plot(xnns, ynns)
    plt.plot(xns, yns)
    plt.plot(xnms, ynms)
    plt.legend(['original', 'zadata slika','dobijena slika naivni', 'dobijena slika DLT', 'dobijena slika modifikovani DLT'])
    plt.show()

def distortion():
    img = cv2.imread("img.jpeg", cv2.IMREAD_COLOR)

    plt.imshow(img)
    plt.show()
    pts_src = np.array([[200, 91],[853, 103],[956, 587],[136, 621]], dtype=float)

    img_copy = np.copy(img)

    img_copy = cv2.cvtColor(img_copy,cv2.COLOR_BGR2RGB)

    A = np.array([200, 91, 1])
    B = np.array([853, 103, 1])
    C = np.array([965, 587, 1])
    D = np.array([136, 621, 1])

    maxWidth = 1080
    maxHeight = 788

    Ap = np.array([0,0,1])
    Bp = np.array([maxWidth, 0, 1])
    Cp = np.array([maxWidth, maxHeight,1])
    Dp = np.array([0, maxHeight,1])
   
    M = naiveAlg(A, B, C, D, Ap, Bp, Cp, Dp)
    out = cv2.warpPerspective(img,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)
    plt.imshow(out)
    plt.show()

if __name__ == '__main__':

    #x1 = np.array([1, 1, 1])                                                                   #KOMENTAR: Ovo je zadati test primer, ali sa skaliranim koord.
    #x2 = np.array([5, 2, 1])                                                                   # Primetimo da je odstupanje dosta vece na ovom primeru,
    #x3 = np.array([6, 4, 1])                                                                   # odnosno da do njega dolazi vec na prvoj decimali u slucaju
    #x4 = np.array([-1, 7, 1])                                                                  # poredjenja naivnog i DLT algoritma, tj. drugoj u slucaju
    #x1p = np.array([0, 0, 1])                                                                  # poredjenja obicnog i modifikovanog DLT algoritma
    #x2p = np.array([10, 0, 1])
    #x3p = np.array([10, 5, 1])
    #x4p = np.array([0, 5, 1])
    #x5 = np.array([3, 1, 1])
    #x5p = np.array([3, -1, 1])

    #x1 = np.array([-6, -1, 1])                                                                 #KOMENTAR: Sve matrice su skalirane tako da je u koordinata 2,2
    #x2 = np.array([6, -1, 1])                                                                  # jednaka jedini, odnosno podeljene su sa vrednoscu na koordinati
    #x3 = np.array([1, 1, 1])                                                                   # 2,2 kako bi se lakse uporedile. Primecujemo da se naivni i DLT
    #x4 = np.array([-1, 1, 1])                                                                  # razlikuju tek na drugoj, a negde cak i trecoj ili cetvtoj koord.
    #x5 = np.array([-6, 1, 4])                                                                  # Takodje primecujemo da je odstupanje izmedju obicnog i modifikovanog
    #x1p = np.array([-6, -3, 3])                                                                # algoritma zanemarljivo mala, obzirom da do odstupanja dolazi
    #x2p = np.array([6, -3, 3])                                                                 # tek na petoj decimali.
    #x3p = np.array([6, 3, 3])
    #x4p = np.array([-6, 3, 3])
    #x5p = np.array([-3.5, -1.8, 3.2])

    #x1 = np.array([2,1,1])
    #x2 = np.array([1,2,1])
    #x3 = np.array([3,4,1])
    #x4 = np.array([-1,-3,1])
    #x5 = np.array([-2,5,1])
    #x1p = np.array([0,1,1])
    #x2p = np.array([5,0,1])
    #x3p = np.array([2,-5,1])
    #x4p = np.array([-1,-1,1])
    #x5p = np.array([4,1,2])
    #umesto da koristimo x5p iz zakomentarisanog dela nize, zaokruzili smo ga
    #P1  = naiveAlg(x1, x2, x3, x4, x1p, x2p, x3p, x4p)
    #x5p = np.dot(P1, x5)
    #print(x5p)

    x1 = np.array([0,-3,1])
    x2 = np.array([0,-1,1])
    x3 = np.array([4,-1,1])
    x4 = np.array([-7,-4,1])
    x5 = np.array([0,5,1])
    x1p = np.array([3,-1,1])
    x2p = np.array([4,4,1])
    x3p = np.array([9,1,1])
    x4p = np.array([5,-2,1])
    x5p = np.array([7,2,2])

    #P2 = DLT4(x1, x2, x3, x4, x1p, x2p, x3p, x4p)
    #P3 = modifiedDLT4(x1, x2, x3, x4, x5, x1p, x2p, x3p, x4p, x5p)
    #P2 = DLT(x1, x2, x3, x4, x5, x1p, x2p, x3p, x4p, x5p)
    P3 = modifiedDLT(x1, x2, x3, x4, x5, x1p, x2p, x3p, x4p, x5p)
    #print("Naivni:\n", P1)
    #print("DLT:\n", P2)
    print("Modifikovani DLT:\n", P3)

    #graph5(x1, x2, x3, x4, x5, x1p, x2p, x3p, x4p, x5p)
    distortion()