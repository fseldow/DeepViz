import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import sys

# USAGE:
# python3 min_line_test.py [n] [-f] [-td]
# [n] is the id of the function to plot, 1 or 2
# -f to use a fixed seed (set at the top of each function)
# -td to plot a 3D surface. otherwise, plot a 2D contour


N = 50
M = 5
delta = 0.025
RANGE = 2
alpha = np.arange(-RANGE-1, RANGE+0.5, delta)
beta = np.arange(-RANGE-1, RANGE+0.5, delta)

ALPHA, BETA = np.meshgrid(alpha, beta)

def plot_f1(fix):
   if fix:
      np.random.seed(11)
      #A = np.asarray([[0,6],[9,3]])
      #y = np.asarray([3,4])
      #x_min = np.matmul(np.linalg.inv(A),y)
   #else:
   # generate random A and y
   while True:
      #A = np.random.rand(DIM,DIM)
      A = np.random.randint(5, size=(M,M))
      y = np.random.randint(5, size=(M,))
      #y = np.random.rand(DIM)
      # make sure A is invertible
      try:
         x_min = np.matmul(np.linalg.inv(A),y)
         break
      except:
         pass
   #print(A)
   #print(y)
   #print(np.shape(x_min))
   #print("x_min = ", x_min)
   # generate random 2D vectors u and v and normalize
   u = np.random.rand(M)
   v = np.random.rand(M)
   a = np.dot(u,v)/np.power(np.linalg.norm(u),2)
   v = v - a*u
   #u = np.asarray([1,0])
   #v = np.asarray([0,1])

   #print('original vector')
   #print(v)
   #print('normalized vector')
   #print(np.linalg.norm(v))
   v = v*(np.linalg.norm(x_min)/np.linalg.norm(v))
   u = u*(np.linalg.norm(x_min)/np.linalg.norm(u))
   
   #print(np.linalg.norm(x_min))
   #print(v)
   f = np.zeros(np.shape(ALPHA))

   for i in range(len(alpha)):
      for j in range(len(beta)):
         z = x_min + alpha[i]*v + beta[j]*u
         f[i,j] = f1_out(z,y,A) 

   return f
   
def plot_f2(fix):
   # generate random A and x
   if fix:
      np.random.seed(5)
      #A = np.asarray([[6,2],[5,9]])
      #x_min = np.asarray([9,0])

      #A = np.asarray([[0,4],[4,6]])
      #x_min = np.asarray([9,9])
   #else:
   A = np.random.randint(5, size=(N,M))
   x_min = np.random.randint(5, size=(M,))
   
   y = abs(np.matmul(A,x_min))   

   #print(A)
   #print(x_min)
   #print(np.shape(x_min))
   #print("x_min = ", x_min)
   # generate random 2D vectors u and v and normalize
   
   
   #print(np.linalg.norm(x_min))
   #print(v)

   
         
   # compute gradient descent in alpha-beta space
   print("starting grad descent")
   #x_init = np.random.randint(5, size=(M,))*2
   x_init = -x_min + np.random.rand(5,)*3
   #compute grad dec in x space
   xseq = f2_grad_descent(x_init,y,A,x_min)
   print("done grad descent")

   # get u based on local minima and orthogonalize v
   #u = np.random.rand(M)
   u = xseq[-1] - x_min
   v = np.random.rand(M)
   a = np.dot(u,v)/np.power(np.linalg.norm(u),2)
   v = v - a*u

   v = v*(np.linalg.norm(x_min)/np.linalg.norm(v))
   u = u*(np.linalg.norm(x_min)/np.linalg.norm(u))
   
   #compute function in alpha-beta space
   f = np.zeros(np.shape(ALPHA))

   for i in range(len(alpha)):
      for j in range(len(beta)):
         z = x_min + alpha[i]*v + beta[j]*u
         f[i,j] = f2_out(z,y,A) 

   #convert vector x to alpha-beta space
   g = []
   for i in range(len(xseq)):
      a,b = xy_to_alpha_beta(xseq[i],x_min,u,v)
      g.append(np.asarray((a,b)))

   
   return f,g


# f = || y-Ax ||^2
def f1_out(x,y,A):
   return np.power(np.linalg.norm(y-np.matmul(A,x)),2)

# f = || y-|Ax| ||^2
def f2_out(x,y,A):
   return np.power(np.linalg.norm(y-abs(np.matmul(A,x))),2)

# takes params y, A, and x_min and initial x and computes gradient descent
# will recompute if minimum found is x_min
# returns sequence x of gradient descent values
def f2_grad_descent(x_init,y,A,x_min):

   #df = 0
   x = x_init
   done = False
   tau = 0.00001
   #xseq = np.asarray(np.asarray([x_init]))
   xseq = [x_init]
   while True:
      while not done:
         df = 0
         for i in range(N):
            t1 = -2*(y[i]-abs(np.dot(A[i,:],x)))
            t2 = (np.dot(A[i,:],x)/abs(np.dot(A[i,:],x)))*A[i,:]
            df = df + t1*t2
            
         x_new = x-tau*df
         #xseq = np.append(xseq,np.asarray([x_new]))
         xseq.append(x_new)
         if(abs(np.linalg.norm(x-x_new))<0.00001):
            done = True
         #print(abs(np.linalg.norm(x-x_new)))
         print(f2_out(x_new,y,A))
         x = x_new
      #break if we're confident this is a different minimum
      #if np.linalg.norm(xseq[-1]-x_min) > 0.01:
      break
      #print("repeating")
   print(x_min)
   print(xseq[-1])

   return xseq

# solves least squares to get mapping in alpha-beta space of x
# returns alpha,beta
def xy_to_alpha_beta(x, x_min, u, v):
   # solve min(alpha,beta): ||(x-x_min) - [u v]*[alpha beta]'||^2
   up = np.asarray([u])
   vp = np.asarray([v])
   #A = np.concatenate((up,vp),axis=1)
   A = np.column_stack((u,v))
   b = x - x_min

   var1 = np.matmul(np.transpose(A),A)
   var2 = np.matmul(np.transpose(A),b)
   var3 = np.matmul(np.linalg.inv(var1),var2)
   
   # return alpha,beta
   return var3[0],var3[1]

if __name__ == "__main__":
   func2plot = None
   fixed = False
   threed = False
   # get command line arguments
   try:
      func2plot = sys.argv[1]
   except:
      pass
   try:
      if sys.argv[2] == "-f":
         fixed = True
      elif sys.argv[3] == "-f":
         fixed = True
   except:
      pass
   try:
      if sys.argv[2] == "-td":
         threed = True
      elif sys.argv[3] == "-td":
         threed = True
   except:
      pass


   # plot with new seed everytime enter is pressed
   while True:
      if func2plot == "2":
         print("Plotting function 2")
         #f = plot_f1()
         if fixed:
            fix = True
         else:
            fix = False
         f,g = plot_f2(fix)
      else:
         print("Plotting function 1")
         if fixed:
            fix = True
         else:
            fix = False
         f = plot_f1(fix)

      #plotting
      fig = plt.figure()
      #CS = plt.contour(ALPHA, BETA, f)
      #plt.clabel(CS, inline=1, fontsize=10)

      # 3D plot
      if threed:
         ax = fig.gca(projection='3d')
         surf = ax.plot_surface(ALPHA, BETA, f, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
         #plt.zlabel('f')
      
      # contour                 
      else:
         CS = plt.contourf(ALPHA, BETA, f, 100, origin='lower')
         cbar = plt.colorbar(CS)
         #grad descent for function 2
         if func2plot == "2":
            print(np.shape(g))
            plt_a = np.empty((0,))
            plt_b = np.empty((0,))
            for i in range(0,len(g),int(np.floor(len(g)/10))):
               #plt.plot(g[i][0],g[i][1],markerfacecolor='r', markeredgecolor='r', marker='o', markersize=2, alpha=0.6)
               plt_a = np.append(plt_a,g[i][0])
               plt_b = np.append(plt_b,g[i][1])
            plt.plot(g[len(g)-1][0],g[len(g)-1][1],markerfacecolor='k', markeredgecolor='k', marker='x', markersize=10, alpha=0.6)
            plt.plot(plt_a,plt_b,'-r', markeredgecolor='r',markersize=2)
            #print(g[-1])
      plt.xlabel(r'$\alpha$')
      plt.ylabel(r'$\beta$')
      plt.title(r'$f(x_{min}+\alpha v + \beta u)$')
      
      
      plt.show(block=False)
      wait = input("PRESS ENTER TO CONTINUE.")
      if input()=='q':
         break

      plt.close()