# -*- coding: utf-8 -*-
"""
Created on Thursday, 30 May 2019

"""
import os
import time
from datetime import datetime
import numpy as np
import scipy.io as scio
import scipy.sparse as sp
# import cupy as cp

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

if __name__ == '__main__':

    ####################################################################################################################
    L = 500
    Network_path = './Generated delta Lattice Embedded Networks L=' + str(L) + ' dg=12'
    if not os.path.exists(Network_path):
        raise ('There is not the path of data')
    laplacian_name_npz = '/laplacian_matrix_deltaLEN_L=' + str(L) + '_dg=12_number=1.npz'
    laplacian_matrix = sp.load_npz(Network_path + laplacian_name_npz)

    node_number = np.array(laplacian_matrix.shape[0], dtype=np.int_)
    ####################################################################################################################
    # set the parameters
    beta = 0.01;
    nu = 0.2;
    gamma = 1/7;
    Omega = 13.36;
    SeedingI = 3;


    Ds = 16.0;
    Di = 1.0;


    dX = 1  # Spatial step size
    dY = 1  # Spatial step size

    DS = Ds / dX / dY;  # scaled diffusion rate of A, corresponding PDE model
    DI = Di / dX / dY;  # scaled diffusion rate of B, corresponding PDE model

    p = 0.1;
    ICn = 1;

    p_values = np.arange(0.1, 0.26, 0.01)
    p_values = np.concatenate((p_values, np.asarray([0.3])))

    for ICn in np.asarray([1]):
        for p in p_values:
            p = np.round(p, 3)
            ####################################################################################################################
            # prepare the initial conditions path with executing setting_initial_conditions.py
            ICData_path = './Many identical random seeding initial conditions L=' + str(L) + '/'
            isExists = os.path.exists(ICData_path)
            if not isExists:
                raise Exception("The expected Intial Conditions Path does not exist. Please execute setting_initial_conditions.py")

            mat_name = ('RandomSeedingICn=' + str(ICn) + 'Omega=' + "{:.3f}".format(Omega).replace('.', 'dot') + 'SeedingI=' + "{:.3f}".format(SeedingI).replace('.', 'dot') + 'p=' + "{:.3f}".format(p).replace('.', 'dot') + '.mat')
            # mat_name = 'initialconditions'+ '.mat'
            # load initial conditions
            S0 = scio.loadmat(ICData_path + mat_name)['S_host'][0]
            I0 = scio.loadmat(ICData_path + mat_name)['I_host'][0]
            print('mean I0:', np.mean(I0), ', mean S0:', np.mean(S0))
            print('shape of S_host:', S0.shape)
            print('shape of I_host:', I0.shape)

            nodes_seedingI = np.where(I0 > 0)[0]
            print("nodes_seedingI:", nodes_seedingI.shape)
            ####################################################################################################################
            T = 1000
            delta_T = 0.001
            interval_time = 10
            step = np.fix(interval_time/delta_T)

            # # test
            # delta_T = 0.001
            # T = delta_T * 5
            # interval_time = delta_T
            # step = np.fix(interval_time/delta_T)

            Data_path = './Step1 Cupy Delta12 Data ICn={} N={} Omega={} beta={} nu={} p={:.3f}/'.format(ICn, node_number, Omega, beta, nu, p)
            if not os.path.exists(Data_path):
                print('Data_path:', Data_path)
                raise Exception("The expected Data Path does not exist. Please execute ...")

            # FigurePath = './Step1 Cupy Delta12 Fig N={} Omega={} beta={} nu={} p={}/'.format(node_number, Omega, beta, nu, p)
            SeriesPath = Data_path.replace("Data", "Figure")
            SeriesPath = SeriesPath.replace("Step1", "Series dataStep1/Series Step1")

            print('SeriesPath:', SeriesPath)
            if not os.path.exists(SeriesPath):
                os.makedirs(SeriesPath)

            orig_map = plt.colormaps.get_cmap('RdYlGn')
            custom_cmap = orig_map.reversed()

            colors = [(0., 0.40784314, 0.21568627, 1.0), (0.83929258, 0.18454441, 0.15286428, 1.0)]
            custom_cmap = LinearSegmentedColormap.from_list('black_to_green', colors, N=256)

            fig, ax = plt.subplots(1, 2, figsize=(16, 6))
            im0 = ax[0].imshow(S0.reshape(L, L), cmap=custom_cmap, extent=[0, L, 0, L]);  # clim=(0,11)
            ax[0].set_title('$S$ density')
            # ax[0].set_xlim(116.2, 116.6)
            # ax[0].set_ylim(39.85, 40.05)
            ax[0].set_axis_off()
            fig.colorbar(im0, ax=ax[0], orientation='vertical', fraction=0.046, pad=0.04)

            im1 = ax[1].imshow(I0.reshape(L, L), cmap=custom_cmap, extent=[0, L, 0, L]);  # clim=(0,11)
            ax[1].set_title('$I$ density')
            # ax[1].set_xlim(116.2, 116.6)
            # ax[1].set_ylim(39.85, 40.05)
            ax[1].set_axis_off()
            fig.colorbar(im1, ax=ax[1], orientation='vertical', fraction=0.046, pad=0.04)
            text = fig.suptitle('initial state', x=0.5, y=0.05, fontsize=16);
            # plt.savefig(SeriesPath + 'Urban Susceptible ' + str(iter + 1) + '.png', bbox_inches='tight', pad_inches=0)
            print(SeriesPath + 'initial state.png')
            plt.savefig(SeriesPath + 'initial state.png', bbox_inches='tight', pad_inches=0.5)
            plt.close()
            # plt.show()

            '''plot loop'''
            time_start = time.time()
            starting_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            total_iteration = int(T / delta_T)

            series_sim_time = []
            series_averagedS = []
            series_averagedI = []
            series_S0_at_nodes_seedingI = []
            series_I0_at_nodes_seedingI = []

            averagedS = np.mean(S0)
            averagedI = np.mean(I0)

            S0_at_nodes_seedingI = S0[nodes_seedingI]
            I0_at_nodes_seedingI = I0[nodes_seedingI]

            sim_time = 0.0
            series_sim_time.append([sim_time])
            series_averagedS.append([averagedS])
            series_averagedI.append([averagedI])
            series_S0_at_nodes_seedingI.append([S0_at_nodes_seedingI])
            series_I0_at_nodes_seedingI.append([I0_at_nodes_seedingI])


            for iter in range(total_iteration):
                if (iter + 1) % step == 0:

                    sim_time = (iter + 1) * delta_T
                    # print('current iteration', iter + 1, '/', total_iteration, 'sim_time', "{:.4f}".format(sim_time))
                    savemat_name = 'SpatialConfigurationPattern T=' + "{:.4f}".format(sim_time) + '.mat'
                    S0 = scio.loadmat(Data_path + savemat_name)['S0'][0]
                    I0 = scio.loadmat(Data_path + savemat_name)['I0'][0]
                    # sim_time_ = scio.loadmat(Data_path + savemat_name)['sim_time'][0][0]
                    # print('sim_time:', sim_time, 'sim_time_:', sim_time_)
                    print('sim_time:' + "{:.4f}".format(sim_time), 'min and max of I0:', np.min(I0), np.max(I0), np.mean(S0 + I0))

                    averagedS = np.mean(S0)
                    averagedI = np.mean(I0)

                    S0_at_nodes_seedingI = S0[nodes_seedingI]
                    I0_at_nodes_seedingI = I0[nodes_seedingI]

                    series_sim_time.append([sim_time])
                    series_averagedS.append([averagedS])
                    series_averagedI.append([averagedI])
                    series_S0_at_nodes_seedingI.append([S0_at_nodes_seedingI])
                    series_I0_at_nodes_seedingI.append([I0_at_nodes_seedingI])

            array_series_sim_time = np.asarray(series_sim_time, dtype=np.float_)
            print('array_series_sim_time.shape', array_series_sim_time.shape)

            array_series_averagedS = np.squeeze(np.asarray(series_averagedS, dtype=np.float_))
            array_series_averagedI = np.squeeze(np.asarray(series_averagedI, dtype=np.float_))

            array_series_S0_at_nodes_seedingI = np.squeeze(np.asarray(series_S0_at_nodes_seedingI, dtype=np.float_))
            array_series_I0_at_nodes_seedingI = np.squeeze(np.asarray(series_I0_at_nodes_seedingI, dtype=np.float_))

            saveseriesmat_name = 'AveSeries.mat'
            scio.savemat(SeriesPath + saveseriesmat_name, {'nodes_seedingI': nodes_seedingI,
                                                           'array_series_sim_time': array_series_sim_time,
                                                           'array_series_averagedS': array_series_averagedS,
                                                           'array_series_averagedI': array_series_averagedI})

            time_end = time.time()
            elapsed_time = time_end - time_start
            print("Simulation took      : %1.1f (s)" % (elapsed_time))

            ending_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("starting datetime: ", starting_datetime, "\nending datetime: ", ending_datetime)