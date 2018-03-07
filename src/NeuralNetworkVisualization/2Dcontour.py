import src.dataGeneration.buildData as bd

import config
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tr
from mpl_toolkits.mplot3d import axes3d, Axes3D

dim=config.dim


lim1=20#limit the range of values
N=20000 #number of points in the train set


[X,fnn,min_pos]=bd.norm2Linear(lim1,N,dim=dim)
mean = np.zeros(dim)
cov = np.identity(dim)

#Map=np.zeros((N,2))

[d1,d2,d] = np.random.multivariate_normal(mean, cov, 3)

x = range(-100,100)
y = range(-100,100)
Map = []
fnn=[]
for i in x:
    for j in y:
        (A, y_single) = bd.readA_y(dim)
        Y = np.ones((dim, 1)) * y_single
        Map.append([i,j])
        X_temp = min_pos + i*d1 + j*d2
        Temp = Y - np.dot(A, X_temp)
        fnn_temp = np.transpose(np.linalg.norm(Temp)**2)

        fnn.append(fnn_temp)
Map = np.asarray(Map)
    #X_train = np.squeeze(np.transpose(Temp))

"""
for i in range(1,N):
    A = np.array([d1, d2, d])
    A2 = np.transpose(A)
    AA = np.dot(A, A2)
    offset = np.transpose(np.array([X[i, :]])) - min_pos
    b = np.dot(A, offset)
    AA_inv = np.linalg.inv(AA)
    ret = np.dot(AA_inv, b)
    Map[i,0]=ret[0,0]
    Map[i,1]=ret[1,0]
"""
fig = plt.figure(1)
ax = Axes3D(fig)
surf = ax.plot_trisurf(np.asarray(Map[:,0]),np.asarray(Map[:,1]),np.squeeze(fnn),cmap='coolwarm',linewidth=0.2, antialiased=False)
plt.colorbar(surf)
plt.show()

