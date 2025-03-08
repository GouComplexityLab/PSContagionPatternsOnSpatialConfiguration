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

    L = 500
    node_number = np.array(L**2, dtype=np.int_)
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

    ####################################################################################################################
    ####################################################################################################################
    T = 1000
    delta_T = 0.0001
    interval_time = 200
    step = np.fix(interval_time / delta_T)

    # # test
    # delta_T = 0.0001
    # T = delta_T * 5
    # interval_time = delta_T
    # step = np.fix(interval_time/delta_T)

    total_iteration = int(T / delta_T)

    p = 0.1;
    ICn = 1;

    p_values = np.asarray([0.008, 0.009, 0.3])

    print('p_values:', p_values)
    print('p_values.shape:', p_values.shape)

    for ICn in np.asarray([1]):
        for p in p_values:
            p = np.round(p, 3)

            Data_path = './Step2 Cupy Powerlaw12 Data ICn={} N={} Omega={} beta={} nu={} p={:.3f}/'.format(ICn, node_number, Omega, beta, nu, p)

            print(Data_path)
            if not os.path.exists(Data_path):
                raise Exception("The expected Data Path does not exist. Please execute ...")

            FigurePath = Data_path.replace("Data", "Figure")
            FigurePath = FigurePath.replace("./", "./PreparedData ")
            print('FigurePath:', FigurePath)
            if not os.path.exists(FigurePath):
                os.makedirs(FigurePath)

            time_start = time.time()
            starting_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            total_iteration = int(T / delta_T)

            for iter in range(total_iteration):
                if (iter + 1) % step == 0:

                    sim_time = (iter + 1) * delta_T
                    # print('current iteration', iter + 1, '/', total_iteration, 'sim_time', "{:.4f}".format(sim_time))
                    loadmat_name = 'SpatialConfigurationPattern T=' + "{:.4f}".format(sim_time) + '.mat'
                    S0 = scio.loadmat(Data_path + loadmat_name)['S0'][0]
                    I0 = scio.loadmat(Data_path + loadmat_name)['I0'][0]
                    # sim_time_ = scio.loadmat(Data_path + savemat_name)['sim_time'][0][0]
                    # print('sim_time:', sim_time, 'sim_time_:', sim_time_)
                    print('sim_time:' + "{:.4f}".format(sim_time), 'min and max of I0:', np.min(I0), np.max(I0), np.mean(S0 + I0))

                    savemat_name = 'SpatialConfigurationPattern T=' + "{:.4f}".format(sim_time) + '.mat'
                    scio.savemat(FigurePath + savemat_name, {'S0': S0, 'I0': I0, 'sim_time': sim_time})

            time_end = time.time()
            elapsed_time = time_end - time_start
            print("Simulation took      : %1.1f (s)" % (elapsed_time))

            ending_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("starting datetime: ", starting_datetime, "\nending datetime: ", ending_datetime)