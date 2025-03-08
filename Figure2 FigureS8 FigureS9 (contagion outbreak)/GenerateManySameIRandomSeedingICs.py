
# % reset -f
from __future__ import absolute_import, print_function
import time
import numpy as np  # 支持大规模的数值计算，提供了高性能的数组对象和数学函数库
from scipy import signal
from scipy import io
import scipy.sparse as sp
import scipy.io as sio
import os


import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


L = 500
Network_path = './Generated delta Lattice Embedded Networks L=' + str(L) + ' dg=12'
if not os.path.exists(Network_path):
    raise('There is not the path of data')

laplacian_name_npz = '/laplacian_matrix_deltaLEN_L=' + str(L) + '_dg=12_number=1.npz'

adjacent_matrix = sp.load_npz(Network_path + laplacian_name_npz)

NetSize = adjacent_matrix.shape[0]

Data_path = './Many identical random seeding initial conditions L=' + str(L) + '/'
if not os.path.exists(Data_path):
    os.makedirs(Data_path)

beta = 0.01; nu = 0.2; gamma = 1 / 7;  # beijing
Ds = 16.0; Di = 1.0;

Omega = 13.36;
SeedingI = 3;

colors = [(0., 0.40784314, 0.21568627, 1.0), (0.83929258, 0.18454441, 0.15286428, 1.0)]
custom_cmap = LinearSegmentedColormap.from_list('black_to_green', colors, N=256)


p = 0.01
ICn = 1

p_values = np.asarray([0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.011,0.012,0.013,0.014,0.015,0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20,0.21,0.22,0.23,0.24,0.25,0.3])
print('p_values:', p_values)
print('p_values.shape:', p_values.shape)

p_values1 = np.arange(0.001,0.016,0.001)
p_values1 = np.concatenate((p_values1, np.asarray([0.1, 0.2, 0.3])))
print('p_values1:', p_values1)
print('p_values1.shape:', p_values1.shape)

p_values2 = np.arange(0.1,0.26,0.01)
p_values2 = np.concatenate((p_values2, np.asarray([0.3])))
print('p_values2:', p_values2)
print('p_values2.shape:', p_values2.shape)

p_values = np.concatenate((np.arange(0.001,0.016,0.001), np.arange(0.1,0.26,0.01), np.asarray([0.3])))
print('p_values:', p_values)
print('p_values.shape:', p_values.shape)

for ICn in np.asarray([1]):
    for p in p_values:
        Omega = round(Omega, 6)
        p = np.round(p, 3)

        random_numbers = np.random.uniform(0, 1, size=NetSize)

        less_than_p = np.where(random_numbers < p)[0]
        greater_equal_p = np.where(random_numbers >= p)[0]

        S = Omega * np.ones(NetSize)
        I = 0     * np.ones(NetSize)

        S[less_than_p] = S[less_than_p] - SeedingI
        I[less_than_p] = I[less_than_p] + SeedingI

        I_host = I.astype(np.float64)
        S_host = S.astype(np.float64)

        # save initial conditions
        if np.array(np.where(S_host < 0)).size > 0:
            print(np.array(np.where(S_host < 0)).size)
            raise Exception("Appearing negative values in S_host.")

        if np.array(np.where(I_host < 0)).size > 0:
            print(np.array(np.where(I_host < 0)).size)
            raise Exception("Appearing negative values in I_host.")

        print('min and max of S:', np.min(S), np.max(S))
        print('min and max of I:', np.min(I), np.max(I))


        data_IC = {'S_host': S_host, 'I_host': I_host, 'random_numbers': random_numbers}
        mat_name = ('RandomSeedingICn=' + str(ICn) + 'Omega=' + "{:.3f}".format(Omega).replace('.', 'dot') + 'SeedingI=' + "{:.3f}".format(SeedingI).replace('.', 'dot') + 'p=' + "{:.3f}".format(p).replace('.', 'dot') + '.mat')
        sio.savemat(Data_path + mat_name, data_IC)

        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        im0 = ax[0].imshow(S_host.reshape(L, L), cmap=custom_cmap, extent=[0, L, 0, L]);
        ax[0].set_title('$S$ density')
        ax[0].set_axis_off()
        fig.colorbar(im0, ax=ax[0], orientation='vertical', fraction=0.046, pad=0.04)

        im1 = ax[1].imshow(I_host.reshape(L, L), cmap=custom_cmap, extent=[0, L, 0, L]);  # clim=(0,11) 这里设定你想控制的范围就行的
        ax[1].set_title('$I$ density')
        ax[1].set_axis_off()
        fig.colorbar(im1, ax=ax[1], orientation='vertical', fraction=0.046, pad=0.04)

        fig_name = ('RandomSeedingICn=' + str(ICn) + 'Omega=' + "{:.3f}".format(Omega).replace('.', 'dot') + 'SeedingI=' + "{:.3f}".format(SeedingI).replace('.', 'dot') + 'p=' + "{:.3f}".format(p).replace('.', 'dot') + '.png')
        plt.savefig(Data_path + '/' + fig_name, bbox_inches='tight', pad_inches=0.5)
        plt.close()
        # plt.show()