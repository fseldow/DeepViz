import src.dataGeneration.buildData as bd

import config
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tr
from mpl_toolkits.mplot3d import axes3d, Axes3D
import math

dim=config.dim
def projection(x, minima, d1, d2):
    x = np.reshape(x, minima.shape)
    minima = np.asarray(minima)
    b = np.reshape(x-minima, (dim,1))
    A = np.concatenate([d1,d2],axis=1)
    At = np.transpose(A)
    AA = np.matmul(At, A)
    AA_inverse = np.linalg.inv(AA)
    ret = np.matmul(AA_inverse, np.dot(At, b))
    return ret

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
d2 = np.asmatrix(np.random.rand(dim, 1)-0.5)

d1 = np.transpose(np.asarray(d1))
d2 = np.transpose(np.asarray(d2))


d2 = d1 - (np.inner(d1, d1)[0,0] / np.inner(d2, d1)[0,0]) * d2
print("d1.d2",np.inner(d1,d2))

d1 = np.transpose(np.asarray(d1))
d2 = np.transpose(np.asarray(d2))


path = np.zeros([echo, 2])
for i in range(echo):
    x_path = np.transpose(trace[i,0:-1])
    path[i,:] = np.transpose(projection(x_path, min_pos, d1, d2))

print(path)

"""
mean = np.zeros(dim)
cov = np.identity(dim)

#Map=np.zeros((N,2))

[d1,d2,d] = np.random.multivariate_normal(mean, cov, 3)
"""
x = range(-30,100)
y = range(-10,10)
Map = np.asarray([[i/50,j/5]for i in x for j in y])
fnn=[]
for pos in Map:
    i=pos[0]
    j=pos[1]
    (A, y_single) = bd.readA_y(dim)
    Y = np.ones((dim, 1)) * y_single

    X_temp = min_pos + i*d1 + j*d2
    Temp = Y - abs(np.matmul(A, X_temp))
    fnn_temp = np.transpose(np.linalg.norm(Temp)**2)
    fnn_temp = math.log(fnn_temp+1)
    fnn.append(fnn_temp)

Map = np.asarray(Map)
print(Map.shape)

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


plt.figure(2)
triang=tr.Triangulation(np.asarray(Map[:,0]),np.asarray(Map[:,1]))
plt.tricontourf(triang,np.squeeze(fnn))#draw contour colors
plt.colorbar()#draw colorbar
plt.tricontour(triang,np.squeeze(fnn))#draw contour lines
plt.plot(np.asarray(path[:,0]),np.asarray(path[:,1]),c='r')
plt.plot(np.asarray(path[-1,0]),np.asarray(path[-1,1]),marker='x')
#plt.set_xlim([min(min(Z1)),max(max(Z1))])
#plt.set_ylim([min(min(Z2)),max(max(Z2))])
plt.title("2d")#set title

plt.show()
