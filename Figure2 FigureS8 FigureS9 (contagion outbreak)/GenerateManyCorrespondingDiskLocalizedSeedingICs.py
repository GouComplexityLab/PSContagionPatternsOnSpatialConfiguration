
# % reset -f
# from __future__ import absolute_import, print_function
import numpy as np  # 支持大规模的数值计算，提供了高性能的数组对象和数学函数库
import scipy.sparse as sp
import scipy.io as scio
import os

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sklearn.neighbors import NearestNeighbors


L = 500

X = np.linspace(0, L, L + 1);
X_step = (X[1] - X[0]);

X = X[0:-1] + X_step * 1 / 2;
# X = X[0:-1] + X_step * 1;

[X, Y] = np.meshgrid(X, X);

x = X.flatten(); y = Y.flatten();



# 生成点集
points = np.column_stack((x, y))
print('shape of points:', points.shape)
# print('points:\n', points)

# Augment the points for periodic boundary conditions
augmented_x = []
augmented_x.append(x)

augmented_y = []
augmented_y.append(y)

for dx in [-L, 0, L]:
    for dy in [-L, 0, L]:
        if dx ==0 and dy ==0:
            pass
        else:
            augmented_x.append(x + dx)
            augmented_y.append(y + dy)

augmented_x = np.hstack(augmented_x)
augmented_y = np.hstack(augmented_y)

augmented_points = np.column_stack((augmented_x, augmented_y))
print('shape of augmented_points:', augmented_points.shape)

nbrs_euclidean = NearestNeighbors(n_neighbors=200000, algorithm='kd_tree').fit(augmented_points)

''' generate corresponding disk-shaped localized seeding '''

Network_path = './Generated delta Lattice Embedded Networks L=' + str(L) + ' dg=12'
if not os.path.exists(Network_path):
    raise('There is not the path of data')

laplacian_name_npz = '/laplacian_matrix_deltaLEN_L=' + str(L) + '_dg=12_number=1.npz'

adjacent_matrix = sp.load_npz(Network_path + laplacian_name_npz)

NetSize = adjacent_matrix.shape[0]
nodeindex = np.array(range(1, NetSize+1))

LoadRandomSeedingData_path = './Many identical random seeding initial conditions L=' + str(L) + '/'
if not os.path.exists(LoadRandomSeedingData_path):
    raise Exception("The expected Random Seeding Data Path does not exist. Please execute ...")

SaveLocalizedSeedingData_path = './Many corresponding disk localized seeding initial conditions L=' + str(L) + '/'
if not os.path.exists(SaveLocalizedSeedingData_path):
    os.makedirs(SaveLocalizedSeedingData_path)

beta = 0.01; nu = 0.2; gamma = 1 / 7;  # beijing
Ds = 16.0; Di = 1.0;

Omega = 13.36;
SeedingI = 3;

orig_map = plt.colormaps.get_cmap('RdYlGn')
custom_cmap = orig_map.reversed()

colors = [(0., 0.40784314, 0.21568627, 1.0), (0.83929258, 0.18454441, 0.15286428, 1.0)]
custom_cmap = LinearSegmentedColormap.from_list('black_to_green', colors, N=256)

p = 0.01;
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

p_values = p_values1

print('p_values:', p_values)
print('p_values.shape:', p_values.shape)


for ICn in np.asarray([1]):

    for p in p_values:
        Omega = round(Omega, 6)
        p = np.round(p, 3)
        loadmat_name = ('RandomSeedingICn=' + str(ICn) + 'Omega=' + "{:.3f}".format(Omega).replace('.', 'dot') + 'SeedingI=' + "{:.3f}".format(SeedingI).replace('.', 'dot') + 'p=' + "{:.3f}".format(p).replace('.', 'dot') + '.mat')
        S_host = scio.loadmat(LoadRandomSeedingData_path + loadmat_name)['S_host'][0]
        I_host = scio.loadmat(LoadRandomSeedingData_path + loadmat_name)['I_host'][0]
        random_numbers = scio.loadmat(LoadRandomSeedingData_path + loadmat_name)['random_numbers'][0]

        less_than_p = np.where(I_host > 0)[0]
        greater_equal_p = np.where(I_host == 0)[0]
        print("===seeding position:", less_than_p.shape)
        print("===not seeding position:", greater_equal_p.shape)


        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        im0 = ax[0].imshow(S_host.reshape(L, L), cmap=custom_cmap, extent=[0, L, 0, L]);
        ax[0].set_title('$S$ density')
        ax[0].set_axis_off()
        fig.colorbar(im0, ax=ax[0], orientation='vertical', fraction=0.046, pad=0.04)

        im1 = ax[1].imshow(I_host.reshape(L, L), cmap=custom_cmap, extent=[0, L, 0, L]);
        ax[1].set_title('$I$ density')
        ax[1].set_axis_off()
        fig.colorbar(im1, ax=ax[1], orientation='vertical', fraction=0.046, pad=0.04)

        fig_name = ('RandomSeedingICn=' + str(ICn) + 'Omega=' + "{:.3f}".format(Omega).replace('.', 'dot') + 'SeedingI=' + "{:.3f}".format(SeedingI).replace('.', 'dot') + 'p=' + "{:.3f}".format(p).replace('.', 'dot') + '.png')
        plt.savefig(SaveLocalizedSeedingData_path + '/' + fig_name, bbox_inches='tight', pad_inches=0.5)
        plt.close()
        # plt.show()

        nodenumber_seedingI = less_than_p.shape[0]
        sampled_nodes = np.random.choice(nodeindex, size=1, replace=False)
        target_sampled_point = np.array(points[sampled_nodes])
        print('target_sampled_point:', target_sampled_point)

        distances, indices = nbrs_euclidean.kneighbors(target_sampled_point)
        distances = np.squeeze(distances)
        indices = np.squeeze(indices)
        indices = np.mod(indices, L ** 2)

        # Shuffle indices for points with the same distance
        unique_distances, inverse_indices = np.unique(distances, return_inverse=True)
        shuffled_indices = []
        shuffled_same_distance_indices_extend = []
        # for d in range(5):
        for d in range(len(unique_distances)):
            same_distance_indices = np.where(inverse_indices == d)[0]
            shuffled_same_distance_indices = np.random.permutation(same_distance_indices)
            shuffled_same_distance_indices_extend.extend(shuffled_same_distance_indices)
            shuffled_indices.extend(indices[shuffled_same_distance_indices])

        shuffled_distances = distances[shuffled_same_distance_indices_extend]
        searching_node_indices = shuffled_indices[0:nodenumber_seedingI]

        searching_nodes_x = x[searching_node_indices]
        searching_nodes_y = y[searching_node_indices]

        S = Omega * np.ones(NetSize)
        I = 0 * np.ones(NetSize)

        print("S.shape:", S.shape)
        print("I.shape:", I.shape)

        S[searching_node_indices] = S[searching_node_indices] - SeedingI
        I[searching_node_indices] = I[searching_node_indices] + SeedingI

        I_localizedseeding = I.astype(np.float64)
        S_localizedseeding = S.astype(np.float64)  # 转换数据类型

        less_than_p = np.where(I_localizedseeding > 0)[0]
        greater_equal_p = np.where(I_localizedseeding == 0)[0]
        print("===localized seeding position:", less_than_p.shape)
        print("===not localized seeding position:", greater_equal_p.shape)


        print('shape of S_localizedseeding:', S_localizedseeding.shape)
        print('shape of I_localizedseeding:', I_localizedseeding.shape)

        print('mean S0=' + "{:.6f}".format(round(np.mean(S), 6))
              + ',  mean I0=' + "{:.6f}".format(round(np.mean(I), 6))
              + ',  mean S0+I0=' + "{:.6f}".format(round(np.mean(S + I), 6)))

        data_IC = {'S_host': S_localizedseeding, 'I_host': I_localizedseeding, 'sampled_nodes': sampled_nodes}
        mat_name = ('CorrespondingLocalizedSeedingICn=' + str(ICn) + 'Omega=' + "{:.3f}".format(Omega).replace('.', 'dot') + 'SeedingI=' + "{:.3f}".format(SeedingI).replace('.', 'dot') + 'p=' + "{:.3f}".format(p).replace('.', 'dot') + '.mat')

        scio.savemat(SaveLocalizedSeedingData_path + mat_name, data_IC)

        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        im0 = ax[0].imshow(S_localizedseeding.reshape(L, L), cmap=custom_cmap, extent=[0, L, 0, L]);
        ax[0].set_title('$S$ density')
        ax[0].set_axis_off()
        fig.colorbar(im0, ax=ax[0], orientation='vertical', fraction=0.046, pad=0.04)

        im1 = ax[1].imshow(I_localizedseeding.reshape(L, L), cmap=custom_cmap, extent=[0, L, 0, L]);
        ax[1].set_title('$I$ density')
        ax[1].set_axis_off()
        fig.colorbar(im1, ax=ax[1], orientation='vertical', fraction=0.046, pad=0.04)

        fig_name = ('CorrespondingLocalizedSeedingICn=' + str(ICn) + 'Omega=' + "{:.3f}".format(Omega).replace('.', 'dot') + 'SeedingI=' + "{:.3f}".format(SeedingI).replace('.', 'dot') + 'p=' + "{:.3f}".format(p).replace('.', 'dot') + '.png')
        plt.savefig(SaveLocalizedSeedingData_path + '/' + fig_name, bbox_inches='tight', pad_inches=0.5)
        plt.close()
        # plt.show()

        # fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        # im0 = ax[0].imshow(I_host.reshape(L, L), cmap=custom_cmap, extent=[0, L, 0, L]);
        # ax[0].set_title('$I$ random seeding')
        # ax[0].set_axis_off()
        # fig.colorbar(im0, ax=ax[0], orientation='vertical', fraction=0.046, pad=0.04)
        #
        # im1 = ax[1].imshow(I_localizedseeding.reshape(L, L), cmap=custom_cmap, extent=[0, L, 0, L]);
        # ax[1].set_title('$I$ localized seeding')
        # ax[1].set_axis_off()
        # fig.colorbar(im1, ax=ax[1], orientation='vertical', fraction=0.046, pad=0.04)
        #
        # fig_name = ('ComparedSeedingICn=' + str(ICn) + 'Omega=' + "{:.3f}".format(Omega).replace('.','dot') + 'SeedingI=' + "{:.3f}".format(SeedingI).replace('.', 'dot') + 'p=' + "{:.3f}".format(p).replace('.', 'dot') + '.png')
        # plt.savefig(SaveLocalizedSeedingData_path + '/' + fig_name, bbox_inches='tight', pad_inches=0.5)
        # plt.close()
        # # plt.show()