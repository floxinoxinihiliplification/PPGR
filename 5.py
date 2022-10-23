import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

def matrix(x,y):
    M = np.array([x[0]*y[0],x[1]*y[0],x[2]*y[0],x[0]*y[1],x[1]*y[1],x[2]*y[1],x[0]*y[2],x[1]*y[2],x[2]*y[2]])
    return M

if __name__ == "__main__":

    #x1 = np.array([958, 38, 1])
    #y1 = np.array([933, 33, 1])
    #x2 = np.array([1117, 111, 1])
    #y2 = np.array([1027, 132, 1])
    #x3 = np.array([874, 285, 1])
    #y3 = np.array([692, 223, 1])
    #x4 = np.array([707, 218, 1])
    #y4 = np.array([595, 123, 1])
    #x9 = np.array([292, 569, 1])
    #y9 = np.array([272, 360, 1])
    #x10 = np.array([770, 969, 1])
    #y10 = np.array([432, 814, 1])
    #x11 = np.array([770, 1465, 1])
    #y11 = np.array([414, 1284, 1])
    #x12 = np.array([317, 1057, 1])
    #y12 = np.array([258, 818, 1])

    x1 = np.array([816, 112, 1])    
    x2 = np.array([951, 161, 1])
    x3 = np.array([989, 125, 1])
    x4 = np.array([854, 78, 1])
    x5 = np.array([792, 306, 1]) #nevidljivo y
    x6 = np.array([912, 357, 1])
    x7 = np.array([949, 320, 1])
    x9 = np.array([323, 346, 1])
    x10 = np.array([453, 370, 1])
    x11 = np.array([507, 272, 1])
    x12 = np.array([385, 248, 1])
    x13 = np.array([366, 560, 1]) #nevidljivo y
    x14 = np.array([477, 584, 1])
    x15 = np.array([526, 489, 1])
    x17 = np.array([136, 553, 1]) #nevidljivo y
    x18 = np.array([430, 762, 1])    
    x19 = np.array([814, 381, 1])
    x20 = np.array([548, 253, 1])
    x21 = np.array([174, 655, 1]) #nevidljivo y
    x22 = np.array([446, 860, 1])
    x23 = np.array([807, 492, 1])
    y1 = np.array([913, 447, 1])
    y2 = np.array([814, 559, 1])
    y3 = np.array([918, 610, 1])
    y4 = np.array([1012, 490, 1])
    y6 = np.array([773, 770, 1])
    y7 = np.array([863, 824, 1])
    y8 = np.array([956, 701, 1]) #nevidljivo x
    y9 = np.array([299, 73, 1])
    y10 = np.array([252, 121, 1])
    y11 = np.array([371, 135, 1])
    y12 = np.array([413, 90, 1])
    y14 = np.array([287, 327, 1])
    y15 = np.array([394, 342, 1])
    y16 = np.array([435, 288, 1]) #nevidljivo x
    y18 = np.array([138, 321, 1])
    y19 = np.array([532, 529, 1])
    y20 = np.array([741, 350, 1])
    y22 = np.array([163, 428, 1])
    y23 = np.array([531, 643, 1])
    y24 = np.array([735, 356, 1]) #nevidljivo x


    xx = np.array([x1, x2, x11, x12, x15, x19, x22, x23])
    yy = np.array([y1, y2, y11, y12, y15, y19, y22, y23])    

    M = np.zeros((8,9))
    for i in range(8):
        M[i] = matrix(xx[i],yy[i])
    
    _, _, Vt = LA.svd(M, full_matrices=True)

    F = np.zeros((3,3))

    F[0] = Vt[8][0:3]
    F[1] = Vt[8][3:6]
    F[2] = Vt[8][6:]

    U,D,Vt = LA.svd(F, full_matrices=True)
    e1 = Vt[2]
    e1 = e1/e1[2]

    _,D,V = LA.svd(np.transpose(F))
    e2 = V[2]
    e2 = e2/e2[2]

    new_D = np.eye(3,3)
    new_D = new_D*D
    new_D[2][2] = 0
    new_F = np.dot(np.dot(U, new_D),Vt)

    #x6 = np.array([1094, 536, 1])
    #y6 = np.array([980, 535, 1])
    #x7 = np.array([862, 729, 1])
    #y7 = np.array([652, 638, 1])
    #x8 = np.array([710, 648, 1])
    #y8 = np.array([567, 532, 1])
    #x14 = np.array([1487, 598, 1])
    #y14 = np.array([1303, 700, 1])
    #x15 = np.array([1462, 1079, 1])
    #y15 = np.array([1257, 1165, 1])
    #y13 = np.array([1077, 269, 1])

    #NEVIDLJIVE TACKE trazimo x8, x16, x24, y5, y13, y17, y21

    x8 = np.cross(np.cross(np.cross(np.cross(x1, x5), np.cross(x7, x3)), x4), np.cross(np.cross(np.cross(x4, x1), np.cross(x2, x3)), x5))
    x8 = x8 / x8[2]
    x8.round()
    x16 = np.cross(np.cross(np.cross(np.cross(x9, x13), np.cross(x15, x11)), x12), np.cross(np.cross(np.cross(x12, x9), np.cross(x10, x11)), x13))
    x16 = x16 / x16[2]
    x16.round()
    x24 = np.cross(np.cross(np.cross(np.cross(x17, x21), np.cross(x23, x19)), x20), np.cross(np.cross(np.cross(x20, x17), np.cross(x18, x19)), x21))
    x24 = x24 / x24[2]
    x24.round()
    y5 = np.cross(np.cross(np.cross(np.cross(y4, y8), np.cross(y6, y2)), y1), np.cross(np.cross(np.cross(y4, y1), np.cross(y2, y3)), y8))
    y5 = y5 / y5[2]
    y5.round()
    y13 = np.cross(np.cross(np.cross(np.cross(y12, y16), np.cross(y14, y10)), y9), np.cross(np.cross(np.cross(y12, y9), np.cross(y10, y11)), y16))
    y13 = y13 / y13[2]
    y13.round()
    y17 = np.cross(np.cross(np.cross(np.cross(y19, y18), np.cross(y23, y22)), y24), np.cross(np.cross(np.cross(y20, y19), np.cross(y23, y24)), y22))
    y17 = y17 / y17[2]
    y17.round()
    y21 = np.cross(np.cross(np.cross(np.cross(y20, y24), np.cross(y22, y18)), y17), np.cross(np.cross(np.cross(y20, y17), np.cross(y18, y19)), y24))
    y21 = y21 / y21[2]
    y21.round()




    xxx = np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24])
    yyy = np.array([y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24])    

    #print(xxx)
    #print(yyy)

    eye = np.eye(3)
    column = np.matrix([0, 0, 0])
    T1 = np.concatenate((eye, np.transpose(column)), axis = 1)

    E2 =  np.array([[0, -e2[2], e2[1]],
                    [e2[2], 0, -e2[0]],
                    [-e2[1], e2[0],0]])

    T2 = np.dot(E2, new_F)
    e22 = np.matrix(e2)
    T2 = np.concatenate((T2, np.transpose(e22)), axis = 1)

    rekonstruisane = []
    for i in range (24):
        x = xxx[i]
        y = yyy[i]
        mat = np.concatenate((np.array(x[1]*T1[2]-x[2]*T1[1]),np.array(-x[0]*T1[2]+x[2]*T1[0]),np.array(y[1]*T2[2]-y[2]*T2[1]),np.array(-y[0]*T2[2]+y[2]*T2[0])))

        #print(np.shape(mat))
        _,_,V1 = LA.svd(mat,full_matrices=True)
        pom = V1[3]
        pom = pom/pom[3]
        rekonstruisane.append(pom)

    np.set_printoptions(suppress=True)

    for i in range (24):
        print(f"\n{i+1}. rekonstrukcija:\n{rekonstruisane[i]}")

    rekonstruisane_faktor = np.zeros((24, 3))
    for i in range(24):
        rekonstruisane_faktor[i][0] = rekonstruisane[i][0]
        rekonstruisane_faktor[i][1] = rekonstruisane[i][1]
        rekonstruisane_faktor[i][2] = rekonstruisane[i][2]*1000

    iviceKeks = np.array([[1, 2], [2, 3], [3, 4], [4, 1],
                          [5, 6], [6, 7], [7, 8], [8, 5],
                          [1, 5], [2, 6], [3, 7], [4, 8]])

    iviceCaj = np.array([[9, 10], [10, 11], [11, 12], [12,  9], 
                            [13, 14], [14, 15], [15, 16], [16, 13],
                            [9, 13], [10, 14], [11, 15], [12, 16]])

    iviceModem = np.array([[17, 18], [18, 19], [19, 20], [20, 17], 
                            [21, 22], [22, 23], [23, 24], [24, 21],
                            [17, 21], [18, 22], [19, 23], [20, 24]])

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    for ivica in iviceModem:
        ax.plot3D([rekonstruisane_faktor[ivica[0] - 1][0], rekonstruisane_faktor[ivica[1] - 1][0]], [rekonstruisane_faktor[ivica[0] - 1][1], rekonstruisane_faktor[ivica[1] - 1][1]], [rekonstruisane_faktor[ivica[0] - 1][2], rekonstruisane_faktor[ivica[1] - 1][2]], 'red')

    for ivica in iviceCaj:
        ax.plot3D([rekonstruisane_faktor[ivica[0] - 1][0], rekonstruisane_faktor[ivica[1] - 1][0]], [rekonstruisane_faktor[ivica[0] - 1][1], rekonstruisane_faktor[ivica[1] - 1][1]], [rekonstruisane_faktor[ivica[0] - 1][2], rekonstruisane_faktor[ivica[1] - 1][2]], 'blue')

    for ivica in iviceKeks:
        ax.plot3D([rekonstruisane_faktor[ivica[0] - 1][0], rekonstruisane_faktor[ivica[1] - 1][0]], [rekonstruisane_faktor[ivica[0] - 1][1], rekonstruisane_faktor[ivica[1] - 1][1]], [rekonstruisane_faktor[ivica[0] - 1][2], rekonstruisane_faktor[ivica[1] - 1][2]], 'green')

    plt.show()
