import numpy as np
import os
import config

def generate_a(dim=3):
    """get random vector a and variable y for selected dimension"""
    y=2
    a=np.random.rand(dim,1)
    return (a,y)

def generateA(dim=3):
    """get random Matrix A and vector y for selected dimension"""
    try:
        A=np.random.rand(dim,dim)-0.5
        A_inv=np.linalg.inv(A)
        #Y=generateY(dim)
        min=10*np.random.rand(dim,1)-5
        Y=abs(np.dot(A,min))
        #Y=generateY(dim)

        return (A,Y)
        pos=np.where(min<0)
        if len(pos[0])==0 or len(pos[0])==dim:
            (A, Y) = generateA(dim)
    except:
        (A,Y)=generateA(dim)
    return (A,Y)

def generateY(dim=3):
    Y=30*np.random.rand(dim,1)
    return Y

############################################

print("###########################################################")
print("start generate parameters")

max_dim=50
project_dir = config.project_dir
filePath = config.dim_dir

fileName='A_Y_para_'
fileName_vector='ay_para_'

for i in range(max_dim+1):
    if i<2:
        continue;
    (A,Y)=generateA(i)
    (a,y)=generate_a(i)
    #Y=generateY(i)
    file=filePath+fileName+str(i)+'.npz'
    np.savez(file, A, Y)

    file2 = filePath + fileName_vector + str(i) + '.npz'
    np.savez(file2, a, y)
print("complete!")
'''
outfile='./dim_parameter/A_Y_para_4.npz'
npzfile = np.load(outfile)
print(npzfile.files)
print(npzfile['arr_0'])
print(npzfile['arr_1'])
'''