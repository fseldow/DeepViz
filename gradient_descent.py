import numpy as np
from buildData import addInput
from buildData import readA_y
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.tri as tr
from keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #disable tensorflow warnings
from buildData import gradientDescent
from buildData import gradientDescentWithConstraint
#import Train_model
import config


dim=config.dim
step=0.1

opt_echos=10000
N_test=10000

(A,y) = readA_y(dim)
#trace=np.empty((opt_echos,3))
x_list=[]
x_pre=1000*np.random.rand(dim,1)

encoder=load_model('my_encoder_3.h5')
decoder=load_model('my_decoder_3.h5')
model=load_model('my_model_3.h5')
real_decoder=load_model('real_decoder_3.h5')



#try to get starting point I want
z_start=[[-30],[-30]]
x_pre=np.transpose(real_decoder.predict(addInput(z_start)))
print('starting point',x_pre)


trace=gradientDescent(opt_echos,A,y,x_pre,encoder,model,step)


x_trace=np.asarray(trace[:,0])
y_trace=np.asarray(trace[:,1])
z_trace=np.log10(np.asarray(trace[:,2])+1)

step=0.01
trace2=gradientDescentWithConstraint(opt_echos,A,y,x_pre,encoder,model,step)


x_trace2=np.asarray(trace2[:,0])
y_trace2=np.asarray(trace2[:,1])
z_trace2=np.log10(np.asarray(trace2[:,2])+1)

print('normal',trace)
print('add constraints',trace2)
###########################################################################
A_inv=np.linalg.inv(A)
center_3d=np.empty((dim,1))
center_3d=addInput(center_3d)
low_center=encoder.predict(center_3d)
#build data in Z domain
z1_center=low_center[0,0]
z2_center=low_center[0,1]

z1_scale=3*(max(x_trace)-min(x_trace)+1)
z2_scale=3*(max(y_trace)-min(y_trace)+1)

Z1=z1_scale*np.random.rand(1,N_test)-(z1_scale/2-z1_center)
Z2=z2_scale*np.random.rand(1,N_test)-(z2_scale/2-z2_center)
Z=np.concatenate((Z1,Z2))
Z=np.squeeze(np.transpose(Z))
Y_test=np.log10(decoder.predict(Z)+1)
#############################################################################
#plot

fig = plt.figure(1)
ax = fig.gca(projection='3d')
surf = ax.plot_trisurf(np.asarray(Z[:,0]),np.asarray(Z[:,1]),np.squeeze(Y_test),cmap='coolwarm',linewidth=0.2, antialiased=False)
ax.plot(x_trace,y_trace,z_trace,c='r')
ax.scatter(x_trace,y_trace,z_trace,c='g')
#plt.colorbar(surf)

fig = plt.figure(2)
ax = fig.gca(projection='3d')
surf = ax.plot_trisurf(np.asarray(Z[:,0]),np.asarray(Z[:,1]),np.squeeze(Y_test),cmap='coolwarm',linewidth=0.2, antialiased=False)
ax.plot(x_trace2,y_trace2,z_trace2,c='r')
ax.scatter(x_trace2,y_trace2,z_trace2,c='g')
#plt.colorbar(surf)


ax.set_xlim([min(min(Z1)),max(max(Z1))])
ax.set_ylim([min(min(Z2)),max(max(Z2))])

plt.figure(3)
triang=tr.Triangulation(np.asarray(Z[:,0]),np.asarray(Z[:,1]))
plt.tricontourf(triang,np.squeeze(Y_test))#draw contour colors
plt.colorbar()#draw colorbar
plt.tricontour(triang,np.squeeze(Y_test))#draw contour lines
plt.plot(x_trace,y_trace,c='r')
plt.scatter(x_trace,y_trace,c='g')
#plt.set_xlim([min(min(Z1)),max(max(Z1))])
#plt.set_ylim([min(min(Z2)),max(max(Z2))])
plt.title("2d")#set title

plt.figure(4)
triang=tr.Triangulation(np.asarray(Z[:,0]),np.asarray(Z[:,1]))
plt.tricontourf(triang,np.squeeze(Y_test))#draw contour colors
plt.colorbar()#draw colorbar
plt.tricontour(triang,np.squeeze(Y_test))#draw contour lines
plt.plot(x_trace2,y_trace2,c='r')
plt.scatter(x_trace2,y_trace2,c='g')
#plt.set_xlim([min(min(Z1)),max(max(Z1))])
#plt.set_ylim([min(min(Z2)),max(max(Z2))])
plt.title("constraints")#set title

plt.show()