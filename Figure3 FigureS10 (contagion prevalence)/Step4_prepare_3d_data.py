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

from skimage import measure

def labelconnectivity_withperiodicalboundary(data, threshold):
    # labeled_connectivity = labelconnectivity_withperiodicalboundary(data, threshold)
    binary_data = (data > threshold).astype(int)

    labeled_connectivity = measure.label(binary_data, connectivity=1, background=0)

    rows, cols, heights = labeled_connectivity.shape


    for i in range(rows):
        for j in range(cols):
            for k in range(heights):

                if i == 0 or i == rows - 1 or j == 0 or k == 0 or j == cols - 1 or k == heights - 1:

                    current_label = labeled_connectivity[i, j, k]

                    if current_label != 0:

                        neighbors = [(i - 1, j, k), (i, j - 1, k), (i + 1, j, k), (i, j + 1, k)]
                        for ni, nj, nk in neighbors:

                            ni = ni % rows
                            nj = nj % cols
                            nk = nk % heights

                            neighbor_label = labeled_connectivity[ni, nj, nk]


                            if neighbor_label != current_label and neighbor_label != 0:
                                min_label = min(current_label, neighbor_label)
                                labeled_connectivity[labeled_connectivity == current_label] = min_label
                                labeled_connectivity[labeled_connectivity == neighbor_label] = min_label

    return labeled_connectivity

if __name__ == '__main__':

    ####################################################################################################################

    L = 500
    X = np.linspace(0, L, L + 1);
    X_step = (X[1] - X[0]);

    X = X[0:-1] + X_step * 1 / 2;
    # X = X[0:-1] + X_step * 1;

    [X, Y] = np.meshgrid(X, X);
    X = np.flipud(X)
    Y = np.flipud(Y)

    x = X.flatten();
    y = Y.flatten();

    node_number = np.array(L**2, dtype=np.int_)
    ####################################################################################################################
    # set the parameters
    beta = 0.01;
    nu = 0.2;
    gamma = 1/7;
    Omega = 13.36;

    Ds = 16.0;
    Di = 1.0;


    dX = 1  # Spatial step size
    dY = 1  # Spatial step size

    DS = Ds / dX / dY;  # scaled diffusion rate of A, corresponding PDE model
    DI = Di / dX / dY;  # scaled diffusion rate of B, corresponding PDE model

    ####################################################################################################################
    T = 4000
    delta_T = 0.001
    interval_time = 10
    step = np.fix(interval_time/delta_T)

    # # test
    # delta_T = 0.001
    # T = delta_T * 10
    # interval_time = delta_T
    # step = np.fix(interval_time/delta_T)

    total_iteration = int(T / delta_T)

    for Omega in np.asarray([13.36]):

        Data_path1 = './Step1 Cupy Delta4 Data N={} Omega={} beta={} nu={}/'.format(node_number, Omega, beta, nu)
        Data_path2 = './Step1 Cupy Delta12 Data N={} Omega={} beta={} nu={}/'.format(node_number, Omega, beta, nu)
        Data_path3 = './Step1 Cupy Poisson12 Data N={} Omega={} beta={} nu={}/'.format(node_number, Omega, beta, nu)
        Data_path4 = './Step1 Cupy Powerlaw12 Data N={} Omega={} beta={} nu={}/'.format(node_number, Omega, beta, nu)

        Data_paths = [Data_path1, Data_path2, Data_path3, Data_path4]
        for Data_path in Data_paths:
            print(Data_path)
            if not os.path.exists(Data_path):
                raise Exception("The expected Data Path does not exist. Please execute ...")

            SeriesDataPath = Data_path.replace("Data", "Figure")
            SeriesDataPath = SeriesDataPath.replace("Step1", "New Step1")
            SeriesDataPath = SeriesDataPath.replace("New Step1", "series data growth analysis/New Step1")

            if not os.path.exists(SeriesDataPath):
                os.makedirs(SeriesDataPath)


            orig_map = plt.colormaps.get_cmap('RdYlGn')  # viridis  YlGn, summer
            custom_cmap = orig_map.reversed()

            colors = [(0.        ,0.40784314,0.21568627,1.0), (0.83929258,0.18454441,0.15286428,1.0)]
            custom_cmap = LinearSegmentedColormap.from_list('black_to_green', colors, N=256)



            ''' step 1 save 3d data '''
            time_start = time.time()
            starting_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            I3d = []
            time_list = []

            for iter in range(total_iteration):
                if (iter + 1) % step == 0:

                    sim_time = (iter + 1) * delta_T
                    savemat_name = 'SpatialConfigurationPattern T=' + "{:.4f}".format(sim_time) + '.mat'
                    S0 = scio.loadmat(Data_path + savemat_name)['S0'][0]
                    I0 = scio.loadmat(Data_path + savemat_name)['I0'][0]

                    reshapedS = S0.reshape(L, L)
                    reshapedI = I0.reshape(L, L)
                    print('sim_time:' + "{:.4f}".format(sim_time), 'min and max of I0:', np.min(I0), np.max(I0), np.mean(S0 + I0))

                    I3d.append(reshapedI)
                    time_list.append(sim_time)

            I3d = np.array(I3d, dtype=np.float_)
            print(I3d.shape)
            I3d = np.transpose(I3d, (1, 2, 0))
            print(I3d.shape)

            savemat_name = 'GrowthRateAnalysisDataI3d.mat'
            scio.savemat(SeriesDataPath+savemat_name, {'I3d': I3d, 'Time': time_list})

            time_end = time.time()
            elapsed_time = time_end - time_start
            print("Simulation took      : %1.1f (s)" % (elapsed_time))

            threshold = 20
            labeled_connectivityI = labelconnectivity_withperiodicalboundary(I3d, threshold)
            savemat_name = 'GrowthRateAnalysisDataIConnectivity3d_threshold=' + "{:.2f}".format(threshold) + '.mat'
            scio.savemat(SeriesDataPath+savemat_name, {'labeled_connectivityI': labeled_connectivityI, 'Time': time_list})
            print('Finish.')

            ''' load data '''

            time_end = time.time()
            elapsed_time = time_end - time_start
            print("Simulation took      : %1.1f (s)" % (elapsed_time))

            ''' finish '''
            ending_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("starting datetime: ", starting_datetime, "\nending datetime: ", ending_datetime)