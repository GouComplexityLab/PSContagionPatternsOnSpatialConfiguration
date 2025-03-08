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
# matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

if __name__ == '__main__':

    ####################################################################################################################
    print("读取网络的矩阵")
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
    # # Wrap around values outside [0, L)
    # x = np.mod(x, L); y = np.mod(y, L);


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
    ####################################################################################################################
    T = 4000
    delta_T = 0.001
    interval_time = 500
    step = np.fix(interval_time/delta_T)

    # # test
    # delta_T = 0.001
    # T = delta_T * 10
    # interval_time = delta_T
    # step = np.fix(interval_time/delta_T)

    total_iteration = int(T / delta_T)

    # for Omega in np.asarray([22.0, 26.0]):
    for Omega in np.asarray([13.36, 22.0, 26.0]):

        Data_path1 = './Step1 Cupy Delta4 Data N={} Omega={} beta={} nu={}/'.format(node_number, Omega, beta, nu)
        Data_path2 = './Step1 Cupy Delta12 Data N={} Omega={} beta={} nu={}/'.format(node_number, Omega, beta, nu)
        Data_path3 = './Step1 Cupy Poisson12 Data N={} Omega={} beta={} nu={}/'.format(node_number, Omega, beta, nu)
        Data_path4 = './Step1 Cupy Powerlaw12 Data N={} Omega={} beta={} nu={}/'.format(node_number, Omega, beta, nu)

        Data_paths = [Data_path1, Data_path2, Data_path3, Data_path4]

        for Data_path in Data_paths:
            print(Data_path)
            if not os.path.exists(Data_path):
                raise Exception("The expected Data Path does not exist. Please execute ...")

            FigurePath = Data_path.replace("Data", "Figure")
            FigurePath = FigurePath.replace("Step1", "New Step1")
            print('FigurePath:', FigurePath)
            if not os.path.exists(FigurePath):
                os.makedirs(FigurePath)

            # savemat_name = 'InitialState.mat'
            # scio.savemat(FigurePath + savemat_name, {'S0': S0, 'I0': I0})

            '''plot loop'''
            time_start = time.time()
            starting_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            total_iteration = int(T / delta_T)


            for iter in range(total_iteration):
                if (iter + 1) % step == 0:

                    sim_time = (iter + 1) * delta_T

                    savemat_name = 'SpatialConfigurationPattern T=' + "{:.4f}".format(sim_time) + '.mat'
                    S0 = scio.loadmat(Data_path + savemat_name)['S0'][0]
                    I0 = scio.loadmat(Data_path + savemat_name)['I0'][0]

                    print('sim_time:' + "{:.4f}".format(sim_time), 'min and max of I0:', np.min(I0), np.max(I0), np.mean(S0 + I0))

                    savemat_name = 'SpatialConfigurationPattern T=' + "{:.4f}".format(sim_time) + '.mat'
                    scio.savemat(FigurePath + savemat_name, {'S0': S0, 'I0': I0, 'sim_time': sim_time})




            time_end = time.time()
            elapsed_time = time_end - time_start
            print("Simulation took      : %1.1f (s)" % (elapsed_time))

            ending_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("starting datetime: ", starting_datetime, "\nending datetime: ", ending_datetime)