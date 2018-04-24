import os

#declare some global parameters
dim = 5
MAX_INT_TIME = 10
Epoch_visual = 100

#def targetFunctionDict(targetFunction):
project_dir = os.path.dirname(os.path.abspath(__file__))+'/'


data_dir    = project_dir+'data/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

module_dir  = data_dir+'module/'
if not os.path.exists(module_dir):
    os.makedirs(module_dir)

dim_dir     = data_dir+'dim_parameter/'
if not os.path.exists(dim_dir):
    os.makedirs(dim_dir)

