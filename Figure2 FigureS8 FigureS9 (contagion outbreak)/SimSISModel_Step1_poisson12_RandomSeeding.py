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
import cupy as cp

if __name__ == '__main__':

    ####################################################################################################################
    L = 500
    Network_path = './Generated mypoisson Lattice Embedded Networks L=500 lam=12'
    if not os.path.exists(Network_path):
        raise ('There is not the path of data')
    laplacian_name_npz = '/laplacian_matrix_mypoissonLEN L=500 lam=12_number=1.npz'
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

    p_values2 = np.arange(0.1, 0.26, 0.01)
    p_values = np.concatenate((p_values2, np.asarray([0.3])))
    print('p_values:', p_values)
    print('p_values.shape:', p_values.shape)

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

            Data_path = './Step1 Cupy Poisson12 Data ICn={} N={} Omega={} beta={} nu={} p={:.3f}/'.format(ICn, node_number, Omega, beta, nu, p)
            if not os.path.exists(Data_path):
                os.makedirs(Data_path)

            '''simulation loop'''

            beta_gpu = cp.array(beta)
            nu_gpu = cp.array(nu)
            gamma_gpu = cp.array(gamma)
            Omega_gpu = cp.array(Omega)

            DS_gpu = cp.array(DS)
            DI_gpu = cp.array(DI)

            S0_gpu = cp.array(S0)  # S
            I0_gpu = cp.array(I0)  # I

            delta_T_gpu = cp.array(delta_T, dtype=cp.float32)

            laplacian_matrix_gpu = cp.sparse.csr_matrix(laplacian_matrix)


            time_start = time.time()
            starting_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            total_iteration = int(T / delta_T)

            for iter in range(total_iteration):
                # print(iter + 1)
                # ==============================================================================================================
                # F = - np.multiply(np.multiply(beta * (1 + nu * I0), I0), S0) + gamma * I0
                # S0 = S0 + (F + DS * laplacian_matrix.dot(S0)) * delta_T
                # I0 = I0 + (-F + DI * laplacian_matrix.dot(I0)) * delta_T

                F_gpu = - cp.multiply(cp.multiply(beta_gpu * (1 + nu_gpu * I0_gpu), I0_gpu), S0_gpu) + gamma_gpu * I0_gpu
                S0_gpu = S0_gpu + ( F_gpu + DS_gpu * laplacian_matrix_gpu.dot(S0_gpu)) * delta_T_gpu
                I0_gpu = I0_gpu + (-F_gpu + DI_gpu * laplacian_matrix_gpu.dot(I0_gpu)) * delta_T_gpu

                if (iter + 1) % step == 0:

                    # Get the data from the GPU
                    S0 = cp.asnumpy(S0_gpu)
                    I0 = cp.asnumpy(I0_gpu)

                    if np.array(np.where(S0 < 0)).size > 0 or np.array(np.where(I0 < 0)).size > 0:
                        print(np.array(np.where(S0 < 0)).size)
                        print(np.array(np.where(I0 < 0)).size)
                        raise Exception("Appearing negative values.")
                    if np.array(np.argwhere(np.isnan(S0))).size > 0 or np.array(np.argwhere(np.isnan(I0))).size > 0:
                        print(np.array(np.argwhere(np.isnan(S0))).size)
                        print(np.array(np.argwhere(np.isnan(I0))).size)
                        raise Exception("Appearing nan values.")
                    if np.array(np.argwhere(np.isinf(S0))).size > 0 or np.array(np.argwhere(np.isinf(I0))).size > 0:
                        print(np.array(np.argwhere(np.isinf(S0))).size)
                        print(np.array(np.argwhere(np.isinf(I0))).size)
                        raise Exception("Appearing inf values.")

                    # ==============================================================================================================


                    sim_time = (iter + 1) * delta_T
                    print('ICn', ICn, 'p', p, 'current iteration', iter + 1, '/', total_iteration, 'sim_time', "{:.4f}".format(sim_time))
                    savemat_name = 'SpatialConfigurationPattern T=' + "{:.4f}".format(sim_time) + '.mat'
                    scio.savemat(Data_path+savemat_name, {'S0': S0, 'I0': I0, 'sim_time': sim_time})

            time_end = time.time()
            elapsed_time = time_end - time_start
            print("Simulation took      : %1.1f (s)" % (elapsed_time))

            ending_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("starting datetime: ", starting_datetime, "\nending datetime: ", ending_datetime)