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

import networkx as nx
import shutil

# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

if __name__ == '__main__':

    ####################################################################################################################

    NetworkDelta4_path = './Generated delta Lattice Embedded Networks L=500 dg=4'
    if not os.path.exists(NetworkDelta4_path):
        raise ('There is not the path of data')
    adjacent_name_npz = '/adjacent_matrix_deltaLEN_L=500_dg=4_number=1.npz'
    adjacent_matrix = sp.load_npz(NetworkDelta4_path + adjacent_name_npz)


    L = 500
    Network_path = './Generated powerlaw Lattice Embedded Networks L=500 mu=3.6231'
    if not os.path.exists(Network_path):
        raise ('There is not the path of data')
    laplacian_name_npz = '/laplacian_matrix_powerlawLEN L=500 mu=3.6231_number=1.npz'
    laplacian_matrix = sp.load_npz(Network_path + laplacian_name_npz)

    node_number = np.array(laplacian_matrix.shape[0], dtype=np.int_)
    ####################################################################################################################
    # set the parameters
    obeta = 0.01; # original beta
    nu = 0.2;
    beta = obeta/4;
    gamma = 1/7;
    Omega = 13.36;

    Ds = 16.0;
    Di = 1.0;

    dX = 1  # Spatial step size
    dY = 1  # Spatial step size

    DS = Ds / dX / dY;  # scaled diffusion rate of A, corresponding PDE model
    DI = Di / dX / dY;  # scaled diffusion rate of B, corresponding PDE model



    ####################################################################################################################


    ControlData_path = './Step1 Cupy Powerlaw12 Data N={} Omega={} beta={} nu={}/'.format(node_number, Omega, obeta, nu)
    print(ControlData_path)
    if not os.path.exists(ControlData_path):
        raise Exception("The expected Data Path does not exist. Please execute ...")


    Controlsim_time = 2000.0
    # print('current iteration', iter + 1, '/', total_iteration, 'sim_time', "{:.4f}".format(sim_time))
    savemat_name = 'SpatialConfigurationPattern T=' + "{:.4f}".format(Controlsim_time) + '.mat'
    uncontroledS0 = scio.loadmat(ControlData_path + savemat_name)['S0'][0]
    uncontroledI0 = scio.loadmat(ControlData_path + savemat_name)['I0'][0]
    # sim_time_ = scio.loadmat(Data_path + savemat_name)['sim_time'][0][0]
    # print('sim_time:', sim_time, 'sim_time_:', sim_time_)
    print('Controlsim_time:' + "{:.4f}".format(Controlsim_time), 'mean uncontroledI0:', np.mean(uncontroledI0),
          ', mean uncontroledS0:', np.mean(uncontroledS0), 'mean uncontroledS0+uncontroledI0:', np.mean(uncontroledS0+uncontroledI0), '\nmin uncontroledI0:', np.min(uncontroledI0),
          ', max uncontroledI0:', np.max(uncontroledI0))
    print('shape of uncontroledS0:', uncontroledS0.shape)
    print('shape of uncontroledI0:', uncontroledI0.shape)

    ''' control setting '''
    controledS0 = uncontroledS0.copy()
    controledI0 = uncontroledI0.copy()

    ''' save and plot control into '''

    Data_path = './Step3 Cupy Powerlaw12 Data DecreaseBeta KeepDs/'
    print(Data_path)
    if not os.path.exists(Data_path):
        os.makedirs(Data_path)
    savemat_name = 'UncontroledSpatialConfigurationPattern T=' + "{:.4f}".format(Controlsim_time) + '.mat'
    scio.savemat(Data_path + savemat_name,
                 {'S0': uncontroledS0, 'I0': uncontroledI0, 'Controlsim_time': Controlsim_time})

    FigurePath = './Step3 Cupy Powerlaw12 Figure DecreaseBeta KeepDs/'
    if not os.path.exists(FigurePath):
        os.makedirs(FigurePath)

    orig_map = plt.colormaps.get_cmap('RdYlGn')  # viridis  YlGn, summer
    custom_cmap = orig_map.reversed()  #

    colors = [(0., 0.40784314, 0.21568627, 1.0), (0.83929258, 0.18454441, 0.15286428, 1.0)]  #
    custom_cmap = LinearSegmentedColormap.from_list('black_to_green', colors, N=256)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].imshow(uncontroledS0.reshape(L, L), cmap=custom_cmap, extent=[0, L, 0, L]);  # clim=(0,11)
    ax[0].set_title('$S$ density')
    # ax[0].set_xlim(116.2, 116.6)
    # ax[0].set_ylim(39.85, 40.05)
    ax[0].set_axis_off()

    ax[1].imshow(uncontroledI0.reshape(L, L), cmap=custom_cmap, extent=[0, L, 0, L]);  # clim=(0,11)
    ax[1].set_title('$I$ density')
    # ax[1].set_xlim(116.2, 116.6)
    # ax[1].set_ylim(39.85, 40.05)
    ax[1].set_axis_off()
    text = fig.suptitle('Control implement at time = ' + "{:.0f}".format(Controlsim_time), x=0.5, y=0.05, fontsize=16);
    # plt.savefig(FigurePath + 'Urban Susceptible ' + str(iter + 1) + '.png', bbox_inches='tight', pad_inches=0)
    plt.savefig(FigurePath + 'Control previous Urban Infectious ' + "{:.0f}".format(Controlsim_time) + '.png',
                bbox_inches='tight',
                pad_inches=0.5)
    plt.close()
    # plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].imshow(controledS0.reshape(L, L), cmap=custom_cmap, extent=[0, L, 0, L]);  # clim=(0,11)
    ax[0].set_title('$I$ density uncontroled')
    # ax[0].set_xlim(116.2, 116.6)
    # ax[0].set_ylim(39.85, 40.05)
    ax[0].set_axis_off()

    ax[1].imshow(controledI0.reshape(L, L), cmap=custom_cmap, extent=[0, L, 0, L]);  # clim=(0,11)
    ax[1].set_title('$I$ density controled')
    # ax[1].set_xlim(116.2, 116.6)
    # ax[1].set_ylim(39.85, 40.05)
    ax[1].set_axis_off()
    text = fig.suptitle('Control implement at time = ' + "{:.0f}".format(Controlsim_time), x=0.5, y=0.05, fontsize=16);
    # plt.savefig(FigurePath + 'Urban Susceptible ' + str(iter + 1) + '.png', bbox_inches='tight', pad_inches=0)
    plt.savefig(FigurePath + 'Control after Urban Infectious ' + "{:.0f}".format(Controlsim_time) + '.png',
                bbox_inches='tight',
                pad_inches=0.5)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].imshow(uncontroledI0.reshape(L, L), cmap=custom_cmap, extent=[0, L, 0, L]);  # clim=(0,11)
    ax[0].set_title(r'$I$ density uncontroled $\langle N \rangle \approx$' + str(
        np.round(np.mean(uncontroledS0 + uncontroledI0), 2)))
    # ax[0].set_xlim(116.2, 116.6)
    # ax[0].set_ylim(39.85, 40.05)
    ax[0].set_axis_off()

    ax[1].imshow(controledI0.reshape(L, L), cmap=custom_cmap, extent=[0, L, 0, L]);  # clim=(0,11)
    ax[1].set_title(
        r'$I$ density controled $\langle N\rangle\approx$' + str(np.round(np.mean(controledS0 + controledI0), 3)))
    # ax[1].set_xlim(116.2, 116.6)
    # ax[1].set_ylim(39.85, 40.05)
    ax[1].set_axis_off()
    text = fig.suptitle('Control implement at time = ' + "{:.0f}".format(Controlsim_time), x=0.5, y=0.05, fontsize=16);
    # plt.savefig(FigurePath + 'Urban Susceptible ' + str(iter + 1) + '.png', bbox_inches='tight', pad_inches=0)
    plt.savefig(FigurePath + 'Control compare Urban Infectious ' + "{:.0f}".format(Controlsim_time) + '.png',
                bbox_inches='tight',
                pad_inches=0.5)
    plt.close()
    # plt.show()

    ''' simulation loop '''
    S0 = controledS0.copy()
    I0 = controledI0.copy()

    savemat_name = 'ControledSpatialConfigurationPattern T=' + "{:.4f}".format(Controlsim_time) + '.mat'
    scio.savemat(Data_path + savemat_name,
                 {'S0': controledS0, 'I0': controledI0, 'Controlsim_time': Controlsim_time})
    ####################################################################################################################
    T = 4000
    delta_T = 0.0001
    interval_time = 10
    step = np.fix(interval_time/delta_T)

    # # test
    # delta_T = 0.0001
    # T = delta_T * 10
    # interval_time = delta_T
    # step = np.fix(interval_time/delta_T)

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
            print('current iteration', iter + 1, '/', total_iteration, 'sim_time', "{:.4f}".format(sim_time))
            savemat_name = 'SpatialConfigurationPattern T=' + "{:.4f}".format(sim_time) + '.mat'
            scio.savemat(Data_path+savemat_name, {'S0': S0, 'I0': I0, 'sim_time': sim_time})

    time_end = time.time()
    elapsed_time = time_end - time_start
    print("Simulation took      : %1.1f (s)" % (elapsed_time))

    ending_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("starting datetime: ", starting_datetime, "\nending datetime: ", ending_datetime)