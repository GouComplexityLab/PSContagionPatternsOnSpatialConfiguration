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

import networkx as nx
import shutil

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

if __name__ == '__main__':

    ####################################################################################################################
    L = 500
    node_number = np.array(L**2, dtype=np.int_)
    ####################################################################################################################
    # set the parameters
    beta = 0.01; # original beta
    onu = 0.2;
    nu = onu/4;
    gamma = 1/7;
    Omega = 13.36;


    Di = 1.0;

    ####################################################################################################################
    ''' save and plot control into '''
    Ds = 2.0; Data_path1 = './Step4 Cupy Delta12 Data DecreaseNu DecreaseDs/'
    Ds = 16.0; Data_path2 = './Step4 Cupy Delta12 Data DecreaseNu KeepDs/'
    Data_path_list = [Data_path1, Data_path2]

    min_I0 = 0;
    max_I0 = 70;
    for Data_path in Data_path_list:

        Data_path = Data_path.replace("Data", "Figure")

        if not os.path.exists(Data_path):
            print('Data_path:', Data_path)
            raise Exception("The expected Data Path does not exist. Please execute ...")

        FigurePath = Data_path.replace("Data", "Figure")

        NewFigurePath = FigurePath.replace("./", "./Choose Plot ")
        if not os.path.exists(NewFigurePath):
            os.makedirs(NewFigurePath)

        Controlsim_time = 2000.0

        savemat_name = 'UncontroledSpatialConfigurationPattern T=' + "{:.4f}".format(Controlsim_time) + '.mat'
        uncontroledS0 = scio.loadmat(Data_path + savemat_name)['S0'][0]
        uncontroledI0 = scio.loadmat(Data_path + savemat_name)['I0'][0]
        Controlsim_time = scio.loadmat(Data_path + savemat_name)['Controlsim_time'][0][0]

        print('Controlsim_time:' + "{:.4f}".format(Controlsim_time), 'mean uncontroledI0:', np.mean(uncontroledI0),
              ', mean uncontroledS0:', np.mean(uncontroledS0),
              'mean uncontroledS0+uncontroledI0:', np.mean(uncontroledS0+uncontroledI0),
              '\nmin uncontroledI0:', np.min(uncontroledI0),
              ', max uncontroledI0:', np.max(uncontroledI0))

        savemat_name = 'ControledSpatialConfigurationPattern T=' + "{:.4f}".format(Controlsim_time) + '.mat'
        controledS0 = scio.loadmat(Data_path + savemat_name)['S0'][0]
        controledI0 = scio.loadmat(Data_path + savemat_name)['I0'][0]

        print('Controlsim_time:' + "{:.4f}".format(Controlsim_time), 'mean controledI0:', np.mean(controledI0),
              ', mean controledS0:', np.mean(controledS0),
              'mean controledS0+controledI0:', np.mean(controledS0+controledI0),
              '\nmin controledI0:', np.min(controledI0),
              ', max controledI0:', np.max(controledI0))

        savemat_name = 'UncontroledSpatialConfigurationPattern T=' + "{:.4f}".format(Controlsim_time) + '.mat'
        scio.savemat(NewFigurePath + savemat_name,
                     {'S0': uncontroledS0, 'I0': uncontroledI0, 'Controlsim_time': Controlsim_time})

        savemat_name = 'ControledSpatialConfigurationPattern T=' + "{:.4f}".format(Controlsim_time) + '.mat'
        scio.savemat(NewFigurePath + savemat_name,
                     {'S0': controledS0, 'I0': controledI0, 'Controlsim_time': Controlsim_time})

        orig_map = plt.colormaps.get_cmap('RdYlGn')  # viridis  YlGn, summer
        custom_cmap = orig_map.reversed()

        colors = [(0., 0.40784314, 0.21568627, 1.0), (0.83929258, 0.18454441, 0.15286428, 1.0)]
        custom_cmap = LinearSegmentedColormap.from_list('black_to_green', colors, N=256)

        fig, ax = plt.subplots(figsize=(6, 6))
        im1 = ax.imshow(uncontroledI0.reshape(L, L), cmap=custom_cmap, extent=[0, L, 0, L], clim=(0, 45));
        cbar = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04);
        im1.set_clim(0, 45)
        cbar.set_ticks([0, 45])
        tick_color = 'k'

        cbar.ax.tick_params(labelsize=24, colors=tick_color)
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontname('Times New Roman')  # Set font to Times New Roman

        # Remove the colorbar's black border
        cbar.outline.set_visible(False)  # This removes the border of the colorbar
        ax.set_axis_off()
        plt.savefig(NewFigurePath + 'initial uncontroledI Urban Infectious ' + '.png', bbox_inches='tight',
                    pad_inches=0)
        plt.savefig(NewFigurePath + 'initial uncontroledI Urban Infectious ' + '.eps', bbox_inches='tight',
                    pad_inches=0)
        plt.close()
        # plt.show()

        fig, ax = plt.subplots(figsize=(6, 6))
        im1 = ax.imshow(controledI0.reshape(L, L), cmap=custom_cmap, extent=[0, L, 0, L], clim=(min_I0, max_I0));  # clim=(0,11) 这里设定你想控制的范围就行的

        ax.set_axis_off()
        plt.savefig(NewFigurePath + 'initial controledI Urban Infectious ' + '.png', bbox_inches='tight',
                    pad_inches=0)
        plt.savefig(NewFigurePath + 'initial controledI Urban Infectious ' + '.eps', bbox_inches='tight',
                    pad_inches=0)
        plt.close()
        # plt.show()

        ''' plot loop '''
        ####################################################################################################################
        T = 4000
        delta_T = 0.001
        interval_time = 1
        step = np.fix(interval_time/delta_T)

        # # test
        # delta_T = 0.0001
        # T = delta_T * 10
        # interval_time = delta_T
        # step = np.fix(interval_time/delta_T)

        time_start = time.time()
        starting_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_iteration = int(T / delta_T)

        choosed_iterations = [500 * step - 1, 1000 * step - 1, 2000 * step - 1, 4000 * step - 1]
        choosed_iterations = np.array(choosed_iterations, dtype=np.int_)
        for iter in choosed_iterations:
            if True:
                # ==============================================================================================================

                sim_time = (iter + 1) * delta_T
                savemat_name = 'SpatialConfigurationPattern T=' + "{:.4f}".format(sim_time) + '.mat'
                S0 = scio.loadmat(Data_path + savemat_name)['S0'][0]
                I0 = scio.loadmat(Data_path + savemat_name)['I0'][0]
                sim_time_ = scio.loadmat(Data_path + savemat_name)['sim_time'][0][0]
                # print('sim_time:', sim_time, 'sim_time_:', sim_time_)
                print('sim_time:' + "{:.4f}".format(sim_time), 'min and max of I0:', np.min(I0), np.max(I0),
                      np.mean(S0 + I0))

                savemat_name = 'SpatialConfigurationPattern T=' + "{:.4f}".format(sim_time_) + '.mat'
                scio.savemat(NewFigurePath + savemat_name, {'S0': S0, 'I0': I0, 'sim_time': sim_time_})

                fig, ax = plt.subplots(figsize=(6, 6))
                im1 = ax.imshow(I0.reshape(L, L), cmap=custom_cmap, extent=[0, L, 0, L], clim=(min_I0, max_I0));  # clim=(0,11) 这里设定你想控制的范围就行的
                ax.set_axis_off()
                plt.savefig(NewFigurePath + 'Urban Infectious ' + "{:.0f}".format(sim_time) + '.png', bbox_inches='tight',
                            pad_inches=0)
                plt.savefig(NewFigurePath + 'Urban Infectious ' + "{:.0f}".format(sim_time) + '.eps', bbox_inches='tight',
                            pad_inches=0)
                plt.close()
                # plt.show()
                if sim_time == 4000:
                    # min_I0 = np.min(I0); max_I0 = np.max(I0);
                    fig, ax = plt.subplots(figsize=(6, 6))
                    im1 = ax.imshow(I0.reshape(L, L), cmap=custom_cmap, extent=[0, L, 0, L],
                                    clim=(min_I0, max_I0));
                    cbar = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04);
                    im1.set_clim(min_I0, max_I0)
                    cbar.set_ticks([min_I0, max_I0])
                    tick_color = [0.850980392156863, 0.325490196078431, 0.0980392156862745]
                    # Set the colorbar tick parameters
                    cbar.ax.tick_params(labelsize=24, colors=tick_color)
                    for tick in cbar.ax.get_yticklabels():
                        tick.set_fontname('Times New Roman')  # Set font to Times New Roman

                    # Remove the colorbar's black border
                    cbar.outline.set_visible(False)  # This removes the border of the colorbar
                    ax.set_axis_off()

                    plt.savefig(NewFigurePath + 'Urban Infectious ' + "{:.0f}".format(sim_time) + ' with colorbar.png',
                                bbox_inches='tight', pad_inches=0)
                    plt.savefig(NewFigurePath + 'Urban Infectious ' + "{:.0f}".format(sim_time) + ' with colorbar.eps',
                                bbox_inches='tight', pad_inches=0)
                    plt.close()
                    # plt.show()
        time_end = time.time()
        elapsed_time = time_end - time_start
        print("Simulation took      : %1.1f (s)" % (elapsed_time))

        ending_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("starting datetime: ", starting_datetime, "\nending datetime: ", ending_datetime)