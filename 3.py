import numpy as np
from numpy import linalg as LA
import math as M

def Euler2A(phi, theta, psi):
    #A = Rz(psi)*Ry(theta)*Rx(phi)
    Rx =  np.array([[1,0,0],
                    [0,M.cos(phi),-M.sin(phi)],
                    [0,M.sin(phi),M.cos(phi)]])
    Ry =  np.array([[M.cos(theta),0,M.sin(theta)],
                    [0,1,0],
                    [-M.sin(theta),0,M.cos(theta)]])
    Rz =  np.array([[M.cos(psi),-M.sin(psi),0],
                    [M.sin(psi),M.cos(psi),0],
                    [0,0,1]])

    return Rz.dot(Ry).dot(Rx)
def A2Euler(A):
    #if LA.det(A) != 1:
    #    return
    #E = np.eye(3)
    #if A.T != LA.inv(A):
    #    return
    if A[2][0] < 1:
        if A[2][0] > -1:                        #jedinstveno resenje
            psi = M.atan2(A[1][0],A[0][0])
            theta = M.asin(-A[2][0])
            phi = M.atan2(A[2][1],A[2][2])
        else:                                   #nije jedinstveno resenje, slucaj Ox3 = -Oz (Gimbal lock)
            psi = M.atan2(-A[0][1],A[1][1])
            theta = M.pi/2
            phi = 0
    else:                                       #nije jedinstveno resenje, slucaj Ox3 = Oz (Gimbal lock)
        psi = M.atan2(-A[0][1],A[1][1])
        theta = -M.pi/2
        phi = 0
    
    return phi, theta, psi
def normalize(p):
    p = np.array(p)
    return p/LA.norm(p)
def perpendicular(p) :
    u = np.empty_like(p)
    u[0] = -p[1]
    u[1] = p[0]
    return u
def A2AxisAngle(A):
    #if np.all(A == np.eye(3)):
    #    return
    #if LA.det(A) != 1:
    #    return
    #if np.any(A.T != LA.inv(A)):
    #    return 
  
    AE = A - np.eye(3)
    first = AE[0]
    second = AE[1]
    third = AE[2]
    
    p = np.cross(first, second)
    if not np.any(p):
        p = np.cross(first, third)
        if not np.any(p):
            p = np.cross(second, third)
    p = normalize(p)

    u = first
    if not np.any(u):
        u = second
        if not np.any(u):
            u = third
    u = normalize(first)
    
    up = A.dot(u)
    up = normalize(up)
    
    phi = round(M.acos(u.dot(up)),4)
    mp = LA.det(np.array([u, up, p]))
    if mp < 0:
        p = -p

    return (p.round(6), phi)
def Rodrigez(p, phi):
    p = normalize(p)
    px =  np.array([[0, -p[2], p[1]],
                    [p[2], 0, -p[0]],
                    [-p[1], p[0],0]])
    p = np.reshape(p, (3,1))
    R1 = p.dot(p.T)
    Ep = np.eye(3) - R1

    return R1 + M.cos(phi)*Ep + M.sin(phi)*px
def AxisAngle2Q(p, phi):
    w = round(M.cos(phi/2),6)
    p = normalize(p)

    x = round(M.sin(phi/2)*p[0],6)
    y = round(M.sin(phi/2)*p[1],6)
    z = round(M.sin(phi/2)*p[2],6)

    return x,y,z,w

def Q2AxisAngle(q):
    if q[0] == 0 and q[1] == 0 and q[2] == 0 and q[3] == 0: #ne sme biti nula
        return
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]

    p = np.array([x,y,z])
    phi = round(2 * M.acos(w),4)

    if LA.norm(p) != 1:
        phi = round(2*M.asin(LA.norm(p)),4)
        p = normalize(p)
        
    return p.round(6), phi

if __name__ == '__main__':
    
    A = Euler2A(M.atan(3/5), M.asin(1/4), -M.atan(4/7))
    #A = Euler2A(-M.atan(1/4), -M.asin(8/9), M.atan(4))
    #print("Euler2A:\n", A)

    p,alpha = A2AxisAngle(A)
    #print("A2AxisAngle:\n", p, alpha)

    A1 = Rodrigez(p, alpha)
    #print("Rodrigez:\n", A1)

    phi, theta, psi = A2Euler(A)
    #print(f"A2Euler:\n phi = {M.tan(phi)},  theta = {M.sin(theta)}, psi = {M.tan(psi)}")

    x,y,z,w = AxisAngle2Q(p, alpha)
    #print(f"AxisAngle2Q:\n q = {x}i + {y}j + {z}k + {w}")

    q = np.array([x,y,z,w])

    p1, phi1 = Q2AxisAngle(q)
    #print(f"Q2AxisAngle:\n p = {p1}, phi = {phi1}")