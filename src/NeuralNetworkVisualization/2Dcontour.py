import src.dataGeneration.buildData as bd

import config
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tr
from mpl_toolkits.mplot3d import axes3d, Axes3D
import math
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import sys
import time



dim = config.dim
theta = 0

def getRotationMatrix(phi, dim):
    ret = np.identity(dim)
    ret[0:0]=math.cos(phi)
    ret[0,1]=-math.sin(phi)
    ret[1,0]=math.sin(phi)
    ret[1,1]=math.cos(phi)
    return ret

def rotated2(d1, d2, phi):
    R=getRotationMatrix(phi,len(d1))
    d1=np.reshape(d1,(len(d1),1))
    temp = np.identity(len(d1)) - np.dot(d1, np.transpose(d1))/(np.linalg.norm(d1))
    d2 = np.dot(temp, np.dot(R, np.reshape(d2,(len(d1),1))))
    return d2

def projection(x, minima, d1, d2):
    """Project a point x onto space determined by d1, d2"""
    x = np.reshape(x, minima.shape)
    minima = np.asarray(minima)
    b = x-minima
    A = np.ones((len(d1),2))
    A[:,0]=np.squeeze(d1)
    A[:,1]=np.squeeze(d2)
    #A = np.concatenate([d1,d2],axis=1)
    At = np.transpose(A)
    AA = np.dot(At, A)
    AA_inverse = np.linalg.inv(AA)
    ret = np.dot(AA_inverse, np.dot(At, b))
    return ret

def convertOthrographic(d1, d2):
    """adjust d2 to make d1 othrographic to d2
        usage: d2= convertOthrographic(d1,d2)
    """
    d1 = np.transpose(np.asarray(d1))
    d2 = np.transpose(np.asarray(d2))
    temp = np.inner(d1, d1)

    d2 = d1 - (np.inner(d1, d1) / np.inner(d2, d1)) * d2

    return d2

lim1=20#limit the range of values
N=20000 #number of points in the train set


#[X,fnn,min_pos]=bd.nonconvex_matrix_absolute(lim1,N,dim=dim)
(A, Y)=bd.readA_y(dim)
minima1 = np.matmul(np.linalg.inv(A),Y)
minima2 = np.matmul(np.linalg.inv(A),-Y)



#find another minima point
echo = 100000
lr = 0.01
start_point = 15* np.random.rand(dim,1)-2 + minima1
trace = bd.gradientDescentAbsMatrix(opt_echos= echo, start_point = start_point, lr = lr)


min_pos=minima1

print("minima1", np.transpose(minima1))
print("minima2", np.transpose(minima2))
print("gd result", trace[-1,0:-1])

d1 = (minima2 - minima1)
#d1 = np.asmatrix(np.random.rand(dim, 1)-0.5)
d2 = np.random.rand(dim)-0.5
d2 = convertOthrographic(d1, d2)



rotate1 = np.random.rand(dim)-0.5
rotate1 = convertOthrographic(d1, rotate1)

rotate2 = np.random.rand(dim)-0.5
rotate2 = convertOthrographic(d1, rotate2)



for f in range(40):

    print("theta",theta)
    theta+=np.pi/20
    path = np.zeros([echo, 2])
    d2_temp = rotated2(d1, d2, theta)
    for i in range(echo):
        x_path = np.transpose(trace[i,0:-1])
        path[i,:] = np.transpose(projection(x_path, min_pos, d1, d2_temp))

    #print(path)

    x = range(-30,100)
    y = range(-10,10)
    Map = np.asarray([[i/50,j/500]for i in x for j in y])
    fnn=[]
    for pos in Map:
        i=pos[0]
        j=pos[1]
        (A, y_single) = bd.readA_y(dim)
        Y = np.ones((dim, 1)) * y_single

        X_temp = min_pos + i*d1 + j*d2_temp
        Temp = Y - abs(np.matmul(A, X_temp))
        fnn_temp = np.transpose(np.linalg.norm(Temp)**2)
        fnn_temp = math.log(fnn_temp+1)
        fnn.append(fnn_temp)

    Map = np.asarray(Map)
    print(Map.shape)


    fig = plt.figure(1)
    ax = Axes3D(fig)
    surf = ax.plot_trisurf(np.asarray(Map[:,0]),np.asarray(Map[:,1]),np.squeeze(fnn),cmap='coolwarm',linewidth=0.2, antialiased=False)
    plt.colorbar(surf)



    fig = plt.figure(2)
    triang=tr.Triangulation(np.asarray(Map[:,0]),np.asarray(Map[:,1]))
    surf = plt.tricontourf(triang,np.squeeze(fnn),np.arange(0, 15, 1))#draw contour colors
    plt.colorbar()#draw colorbar
    surf2 = plt.tricontour(triang,np.squeeze(fnn),np.arange(0, 15, 1))#draw contour lines
    line = plt.plot(np.asarray(path[:,0]),np.asarray(path[:,1]),c='r')
    plt.plot(np.asarray(path[-1,0]),np.asarray(path[-1,1]),marker='x')
    #plt.set_xlim([min(min(Z1)),max(max(Z1))])
    #plt.set_ylim([min(min(Z2)),max(max(Z2))])
    plt.title("rotate "+"%.2f" % theta)#set title


    #plt.show()
    #time.sleep(5)
    fig.savefig(str(f)+".png")
    fig.clear()


