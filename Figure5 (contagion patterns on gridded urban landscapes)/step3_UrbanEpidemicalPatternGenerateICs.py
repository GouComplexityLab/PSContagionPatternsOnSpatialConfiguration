
# % reset -f
from __future__ import absolute_import, print_function
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import os


target_covariance = 0.6
NetworkonUrbanLandscape_path = './Network on Gridded Urban Landscape Beijing Powerlaw12 cov=' + str(target_covariance)
if not os.path.exists(NetworkonUrbanLandscape_path):
    raise('There is not the path of data')

laplacian_name_npz = '/laplacian_matrix_Beijing.npz'

adjacent_matrix = sp.load_npz(NetworkonUrbanLandscape_path + laplacian_name_npz)

NetSize = adjacent_matrix.shape[0]

Data_path = './The same initial conditions for gridded urban landscape/'
if not os.path.exists(Data_path):
    os.makedirs(Data_path)

beta = 0.01; nu = 0.2; gamma = 1 / 7;
Ds = 16.0; Di = 1.0;


for N in np.asarray([13.36]):
    N = round(N, 6)
    # print('N = ', N)
    # Initial values#############################################################################################
    # The array is allocated on the GPU and the initial values are copied onto it
    I0 = (beta * (nu * N - 1) + np.sqrt((beta + beta * nu * N) ** 2 - 4 * beta * nu * gamma)) / (2 * beta * nu)
    I = I0 + (np.random.rand(np.array(NetSize, dtype=np.int32)) - 0.5) * 0.1
    I_host = I.astype(np.float64)

    S0 = N - I0
    S = S0 + (np.random.rand(np.array(NetSize, dtype=np.int32)) - 0.5) * 0.1
    S_host = S.astype(np.float64)  # 转换数据类型

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
    # "{:.3f}".format(N)
    mat_name = 'initialconditionsN=' + "{:.6f}".format(N).replace('.', 'dot') + '.mat'
    sio.savemat(Data_path + mat_name, data_IC)