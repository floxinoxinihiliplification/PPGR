import numpy as np
import math as M
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib import animation as A
import mpl_toolkits.mplot3d.axes3d as Axes3D

def normalize(p):
    p = np.array(p)
    if LA.norm(p) != 0:
        return p/LA.norm(p)
    else:
        return p

def Lerp(q1, q2, tm, t):
    q = (1-t/tm)*q1 + t/tm*q2
    return q

def SLerp(q1, q2, tm, t):
    #if t < 0:
    #    return
    #if t > tm:
    #    return
    q1 = normalize(q1)
    q2 = normalize(q2)

    cos = q1.dot(q2)
    if cos < 0:
        q1 = -q1
        cos = -cos
    if cos > 0.95:
        return Lerp(q1,q2,tm, t)
    phi = M.acos(cos)

    q = (M.sin(phi*(1-t/tm))/M.sin(phi))*q1 + (M.sin(phi*t/tm)/M.sin(phi))*q2
    q = normalize(q)
    return q

def invQ(q):
    inv = np.array([-q[0],-q[1],-q[2],q[3]])
    return inv #/ (LA.norm(q) ** 2)

def multiplyQ(q1, q2):
    x1,y1,z1,w1 = q1
    x2,y2,z2,w2 = q2
    return np.array([-x1*x2 - y1*y2 - z1*z2 + w1*w2,
                     x1*w2 + y1*z2 - z1*y2 + w1*x2,
                    -x1*z2 + y1*w2 + z1*x2 + w1*y2,
                     x1*y2 - y1*x2 + z1*w2 + w1*z2])
    

# Rotacija kvaterniona preko q * p * q^-1. p mora da bude kvaternion pa zato dodajemo 0 na kraj (w=0)
def transform(p, q):
    p = np.array([p[0], p[1], p[2], 0.0])
    return multiplyQ(multiplyQ(q, p), invQ(q))[:-1]

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
def AxisAngle2Q(p, phi):
    w = round(M.cos(phi/2),6)
    p = normalize(p)

    x = round(M.sin(phi/2)*p[0],6)
    y = round(M.sin(phi/2)*p[1],6)
    z = round(M.sin(phi/2)*p[2],6)

    return x,y,z,w
def AxisAngle(A):
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


if __name__ == '__main__':

    tm = 100
    
    # Pocetna pozicija
    pStart = np.array([7, 5, 6])

    # Rotacija
    AStart = Euler2A(M.pi / 6, 0, 0)
    # Pocetni kvaternion
    u, angle = AxisAngle(AStart)
    angle = 0
    qStart = AxisAngle2Q(u, angle)
    
    # Krajnja pozicija
    pEnd = np.array([2, 1, 4])

    # Rotacija
    AEnd = Euler2A(-M.pi / 3, 0, -M.pi / 9)
    # Krajnji kvaternion
    u, angle = AxisAngle(AEnd)
    qEnd = AxisAngle2Q(u, angle) 


    fig = plt.figure()
    ax = Axes3D.Axes3D(fig)
    ax.set_xlim3d([0.0, 10.0])
    ax.set_xlabel('X osa')

    ax.set_ylim3d([0.0, 10.0])
    ax.set_ylabel('Y osa')

    ax.set_zlim3d([0.0, 10.0])
    ax.set_zlabel('Z osa')

    ax.view_init(10, -5)
    colors = ['r', 'g', 'b']
    axis = np.array(sum([ax.plot([], [], [], c=c) for c in colors], []))

    # Ovo su pocetne i krajnje tacke duzi od kojih krecemo
    startpoints = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    endpoints = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]])
    
    # Iscrtavanje pocetne i krajnje pozicije
    # Inicijalno sve u koord. pocetku, pa kada se primeni transformacija treba da se translira
    for i in range(3):
        start = transform(startpoints[i], qStart)
        end = transform(endpoints[i], qStart)
        start += pStart
        end += pStart
	    
        # Iscrtavamo duz na grafiku
        ax.plot([start[0], end[0]], [start[1], end[1]], zs=[start[2], end[2]], color=colors[i])

        start = transform(startpoints[i], qEnd)
        end = transform(endpoints[i], qEnd)
        start += pEnd
        end += pEnd
        ax.plot([start[0], end[0]], [start[1], end[1]], zs=[start[2], end[2]], color=colors[i])

    # Init funkcija za animaciju
    def init():
        for a in axis:
            a.set_data(np.array([]), np.array([]))
            a.set_3d_properties(np.array([]))

        return axis

    def animate(frame):
        q = SLerp(np.array(qStart), np.array(qEnd), tm, frame)
	    
        # Korak koji se dodaje tackama vektora da bi se konstantno translirale ka krajnjim tackama a ne samo rotirale u koord. pocetku
        korak = frame * (pEnd - pStart) / tm
	    
        for a, start, end in zip(axis, startpoints, endpoints):
            start = transform(np.array(start), np.array(q))
            end = transform(np.array(end), np.array(q))
            start += pStart + korak
            end += pStart + korak

            a.set_data(np.array([start[0], end[0]]), np.array([start[1], end[1]]))
            a.set_3d_properties(np.array([start[2], end[2]]))
        
        fig.canvas.draw()
        return axis

    anim = A.FuncAnimation(fig, animate, frames=tm, init_func=init, interval=5, repeat=True, repeat_delay=20)
    anim.save('slerp.gif')
    plt.show()