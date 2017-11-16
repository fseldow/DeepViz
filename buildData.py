import numpy as np
import config
MAX_INT=10
def readA_y(dim):
    outfile = './dim_parameter/A_Y_para_'+str(dim)+'.npz'
    npzfile = np.load(outfile)
    A=npzfile['arr_0']
    y=npzfile['arr_1']
    return (A,y)


def constructData(lim1,N,dim=3):
    a = -lim1
    b = lim1
    (A,y_single) = readA_y(dim)
    A_inv=np.linalg.inv(A)

    Y = np.ones((dim, N)) * y_single

    X_mean=np.dot(A_inv,np.asarray(y_single))

    print('should min',X_mean)

    X = lim1 * np.random.rand(dim, N) - lim1 / 2 + X_mean
    X_train = np.squeeze(np.transpose(X))

    Temp =Y- np.dot(A, X)
    fnn = np.transpose(np.linalg.norm(Temp,axis=0)**2)
    fnn=np.squeeze(fnn)

    X_train = np.squeeze(np.transpose(Temp))
    return [X_train,fnn]

def constructDataWithConstraints(lim1,N,dim=3):

    a = -lim1
    b = lim1
    (A,y_single) = readA_y(dim)
    A_inv=np.linalg.inv(A)

    Y = np.ones((dim, N)) * y_single


    X_mean=np.dot(A_inv,np.asarray(y_single))
    print('should min', X_mean)

    X1 = lim1 * np.random.rand(dim, int(N/2))#+ X_mean
    X2 = lim1 * np.random.rand(dim, int(N/2)) -lim1/2+ X_mean
    X=np.concatenate((X1,X2),axis=1)

    X_train = np.squeeze(np.transpose(X))

    Temp =Y- np.dot(A, X)
    fnn = np.linalg.norm(Temp,axis=0)**2
    exceed_pos=np.where(X < 0)
    max_result=max(fnn)*MAX_INT
    print('inf',max_result)
    list=[]
    [list.append(i) for i in exceed_pos[1] if not i in list]
    for item in list:
        fnn[item]=max_result


    #fnn=np.squeeze(fnn)
    print('minima_point',getMinima(X_train,fnn))
    #X_train = np.squeeze(np.transpose(Temp))
    return [X_train,fnn]

def constructDataMAE(lim1,N):
    a = -lim1
    b = lim1
    A = np.array([[4, 9, 2], [8, 1, 6], [3, 5, 7]])
    A_inv=np.linalg.inv(A)
    y1 = np.ones((1, N)) * 8
    y2 = np.ones((1, N)) * 2
    y3 = np.ones((1, N)) * 5
    y = np.concatenate((y1, y2, y3))

    X_mean=np.dot(A_inv,np.asarray(y[:,0]))

    #X_mean=np.dot(np.inv(A),y)
    X1 = np.random.normal(X_mean[0], 10, (1,N))
    X2 = np.random.normal(X_mean[1], 10, (1,N))
    X3 = np.random.normal(X_mean[2], 10, (1,N))

    # concetenate values into a matrix
    X = np.concatenate((X1, X2, X3))

    Temp = np.dot(A, X)
    x1 = np.asarray(Temp[0, :])
    x2 = np.asarray(Temp[1, :])
    x3 = np.asarray(Temp[2, :])

    y1 = np.asarray(y[0, :])
    y2 = np.asarray(y[1, :])
    y3 = np.asarray(y[2, :])
    fnn = np.transpose(abs(y1 - x1)  + abs(y2 - x2)  + abs(y3 - x3) )
    x1 = np.asarray(X[0, :])
    x2 = np.asarray(X[1, :])
    x3 = np.asarray(X[2, :])
    #add = np.vstack([x1 * x1, x2 * x2, x3 * x3, x1 * x2, x1 * x3, x2 * x3])
    #X = np.vstack((X, add))
    X_train = np.squeeze(np.transpose(X))
    return [X_train,fnn]


def addInput(center_3d):
    #add = np.vstack([center1 * center1, center2 * center2, center3 * center3, center1 * center2, center1 * center3, center2 * center3])
    #center_3d = np.vstack((center_3d, add))
    (A,y)=readA_y(dim=config.dim)
    #center_3d=y-np.dot(A,center_3d)
    center_3d = np.expand_dims(np.squeeze(np.transpose(center_3d)), axis=0)
    return center_3d




def gradientDescent(opt_echos,A,y,start_point,encoder,model,lr):
    trace = np.empty((opt_echos, 3))
    x_pre=start_point
    for i in range(opt_echos):

        # x_list.append(x_pre)
        x_input = addInput(x_pre)
        z_pos = encoder.predict(x_input)
        y_output = model.predict(x_input)
        print(i, 'normal_path', np.transpose(x_pre))
        trace[i, :] = (np.concatenate((z_pos, y_output), axis=1))

        #descent
        temp = np.dot(A, x_pre) - y
        # MSE
        temp2 = x_pre - 2 * lr * np.dot(np.transpose(A), temp)
        #if np.linalg.norm(temp2-x_pre)<lr*0.01:
        #    trace=np.delete(trace,range(i+1,opt_echos),0)
         #   break
        #else:
        x_pre=temp2
        # MAE
        # x_pre = x_pre - step*np.sign(temp)
        #constraints
        #x_pre[np.where(x_pre<0)]=0
    return trace

def gradientDescentWithConstraint(opt_echos,A,y,start_point,encoder,model,lr):
    trace = np.empty((opt_echos, 3))
    x_pre=start_point
    for i in range(opt_echos):
        # x_list.append(x_pre)

        x_input = addInput(x_pre)
        z_pos = encoder.predict(x_input)
        y_output = model.predict(x_input)
        print(i, 'constraint_path', np.transpose(x_pre))
        trace[i, :] = (np.concatenate((z_pos, y_output), axis=1))

        #descent
        temp = np.dot(A, x_pre) - y
        # MSE
        temp2=x_pre
        x_pre = x_pre - 2 * lr * np.dot(np.transpose(A), temp)
        #constraints
        x_pre[np.where(x_pre<0)]=0
        #if np.linalg.norm(temp2-x_pre)<lr*0.01:
         #   trace = np.delete(trace, range(i + 1, opt_echos), 0)
         #   break
    return trace
def getKey(a):
    return a[1]
def getMinima(X_train,fnn):
    min_pos=np.where(fnn==np.min(fnn))
    ret=X_train[min_pos[0],:]
    fnn_cp=[]
    for i in range(len(fnn)):
        fnn_cp.append([i,fnn[i]])
    fnn_cp.sort(key=getKey)
    for i in range(10):
        print(str(i)+'th',X_train[fnn_cp[i][0],:])
    return ret

'''
x_trace=np.squeeze(trace[:,0])
y_trace=np.squeeze(trace[:,1])
z_trace=np.squeeze(trace[:,2])
'''
'''
(A,y)=readA_y(config.dim)
print(A,y)
'''