
# % reset -f
from __future__ import absolute_import, print_function
import time
import numpy as np
# import pyopencl as cl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import signal
from matplotlib import rcParams
from scipy import io
import scipy.sparse as sp
import scipy.io as sio
import os


Network_path = './Generated delta Lattice Embedded Networks L=500 dg=12'
if not os.path.exists(Network_path):
    raise('There is not the path of data')

laplacian_name_npz = '/laplacian_matrix_deltaLEN_L=500_dg=12_number=1.npz'

adjacent_matrix = sp.load_npz(Network_path + laplacian_name_npz)

NetSize = adjacent_matrix.shape[0]

Data_path = './The same initial conditions for spatial configurations/'
if not os.path.exists(Data_path):
    os.makedirs(Data_path)

beta = 0.01; nu = 0.2; gamma = 1 / 7;
Ds = 16.0; Di = 1.0;

Omega = 13.36;

for Omega in np.asarray([13.36,22,26]):
    Omega = round(Omega, 6)
    # Initial values#############################################################################################
    # The array is allocated on the GPU and the initial values are copied onto it
    I0 = (beta * (nu * Omega - 1) + np.sqrt((beta + beta * nu * Omega) ** 2 - 4 * beta * nu * gamma)) / (2 * beta * nu)

    I = I0 + (np.random.rand(np.array(NetSize, dtype=np.int32)) - 0.5) * 0.1
    # (np.sqrt(a)-b + np.sqrt(a/c)-b)/2 + (np.random.rand(n*n)-0.5)
    I_host = I.astype(np.float64)

    S0 = Omega - I0
    # input()
    S = S0 + (np.random.rand(np.array(NetSize, dtype=np.int32)) - 0.5) * 0.1
    S_host = S.astype(np.float64)  # 转换数据类型

    print('Omega = ', Omega)
    print('         NetSize:', NetSize)
    print('shape of S_host:', S_host.shape)
    print('shape of I_host:', I_host.shape)

    print('mean S0=' + "{:.6f}".format(round(np.mean(S0), 6))
          + ',  mean I0=' + "{:.6f}".format(round(np.mean(I0), 6))
          + ',  mean S0+I0=' + "{:.6f}".format(round(np.mean(S0+I0), 6)))
    # save initial conditions

    if np.array(np.where(S_host < 0)).size > 0:
        print(np.array(np.where(S_host < 0)).size)
        raise Exception("Appearing negative values in S_host.")

    if np.array(np.where(I_host < 0)).size > 0:
        print(np.array(np.where(I_host < 0)).size)
        raise Exception("Appearing negative values in I_host.")

    data_IC = {'S_host': S_host, 'I_host': I_host, }
    mat_name = 'initialconditionsOmega=' + "{:.6f}".format(Omega).replace('.', 'dot') + '.mat'
    sio.savemat(Data_path + mat_name, data_IC)