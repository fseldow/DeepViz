import numpy as np

def savePlane(path, alpha, beta, loss, name = ""):
    surface = np.reshape(loss, (len(beta), len(alpha)))
    if(len(name)==0):
        np.savetxt(path + "surface.csv", surface, delimiter=",", fmt='%10.5f')
    else:
        np.savetxt(path + name, surface, delimiter=",", fmt='%10.5f')
    axisInf = []
    xInf = [min(alpha), max(alpha), len(alpha), (max(alpha) - min(alpha)) / (len(alpha) - 1)]
    yInf = [min(beta), max(beta), len(beta), (max(beta) - min(beta)) / (len(beta) - 1)]
    axisInf.append(xInf)
    axisInf.append(yInf)
    np.savetxt(path + "AxisSetting.csv", np.asarray(axisInf), delimiter=',', fmt='%10.5f')