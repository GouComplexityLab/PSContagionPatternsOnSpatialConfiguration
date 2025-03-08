# -*- coding: utf-8 -*-
import os
import time
from datetime import datetime
import numpy as np
import scipy.io as scio
import scipy.sparse as sp

import geopandas as gpd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

if __name__ == '__main__':

    shp_file = "./dataBeijing/beijingshp/BeijingTownshipStreetBoundary.shp"
    gdf = gpd.read_file(shp_file)

    target_covariance = 0.6
    for target_covariance in np.asarray([0.6]):
        NetworkonUrbanLandscape_path = './Network on Gridded Urban Landscape Beijing Powerlaw12 cov=' + str(target_covariance)
        if not os.path.exists(NetworkonUrbanLandscape_path):
            raise ('There is not the path of data')

        mat_name = '/AllGriddedData_Beijing.mat'
        print('mat_name: ', mat_name)

        ''' load All Gridded Data Beijing '''
        notnan_position_xs = scio.loadmat(NetworkonUrbanLandscape_path + mat_name)['notnan_position_xs'][0]
        notnan_position_ys = scio.loadmat(NetworkonUrbanLandscape_path + mat_name)['notnan_position_ys'][0]
        notnan_position_populationdensity_many = scio.loadmat(NetworkonUrbanLandscape_path + mat_name)['notnan_position_populationdensity_many'][0]
        notnan_position_labels = scio.loadmat(NetworkonUrbanLandscape_path + mat_name)['notnan_position_labels'][0]
        node_areas = scio.loadmat(NetworkonUrbanLandscape_path + mat_name)['node_areas'][0]

        ####################################################################################################################
        laplacian_name_npz = '/laplacian_matrix_Beijing.npz'
        laplacian_matrix = sp.load_npz(NetworkonUrbanLandscape_path + laplacian_name_npz)

        node_number = np.array(laplacian_matrix.shape[0], dtype=np.int_)
        ####################################################################################################################
        # set the parameters
        beta = 0.01;
        nu = 0.2;
        gamma = 1 / 7;
        Omega = 13.36;

        Ds = 16.0;
        Di = 1.0;

        dX = 1  # Spatial step size
        dY = 1  # Spatial step size

        DS = Ds / dX / dY;  # scaled diffusion rate of A, corresponding PDE model
        DI = Di / dX / dY;  # scaled diffusion rate of B, corresponding PDE model

        ####################################################################################################################
        # prepare the initial conditions path with executing setting_initial_conditions.py
        ICData_path = './The same initial conditions for gridded urban landscape/'
        isExists = os.path.exists(ICData_path)
        if not isExists:
            raise Exception("The expected Intial Conditions Path does not exist. Please execute setting_initial_conditions.py")

        mat_name = 'initialconditionsN=' + "{:.6f}".format(Omega).replace('.', 'dot') + '.mat'
        # load initial conditions
        S0 = scio.loadmat(ICData_path + mat_name)['S_host'][0]
        I0 = scio.loadmat(ICData_path + mat_name)['I_host'][0]
        print('mean I0:', np.mean(I0), ', mean S0:', np.mean(S0))
        print('shape of S_host:', S0.shape)
        print('shape of I_host:', I0.shape)
        ####################################################################################################################
        T = 4000
        delta_T = 0.0001
        interval_time = 1
        step = np.fix(interval_time/delta_T)

        # # test
        # delta_T = 0.0001
        # T = delta_T * 10
        # interval_time = delta_T
        # step = np.fix(interval_time/delta_T)

        # Data_path = './Cupy Urban Epidemical Pattern Beijing Data Powerlaw12 N={} Omega={} beta={} nu={} cov={}/'.format(node_number, Omega, beta, nu, target_covariance)
        Data_path = './Cupy Urban Epidemical Pattern Beijing Figure Powerlaw12 N={} Omega={} beta={} nu={} cov={}/'.format(node_number, Omega, beta, nu, target_covariance)

        if not os.path.exists(Data_path):
            print(Data_path)
            raise Exception("The expected Data Path does not exist. Please execute SimUrbanEpidemicalPatternSISModel.py")


        FigurePath = Data_path.replace("Data", "Figure")
        if not os.path.exists(FigurePath):
            os.makedirs(FigurePath)

        orig_map = plt.colormaps.get_cmap('RdYlGn')  # viridis  YlGn, summer
        custom_cmap = orig_map.reversed()

        colors = [(0., 0.40784314, 0.21568627, 1.0), (0.83929258, 0.18454441, 0.15286428, 1.0)]
        # colors = [(0.09196463,0.57762399,0.3041138,1.0), (0.83929258,0.18454441,0.15286428,1.0)]
        custom_cmap = LinearSegmentedColormap.from_list('black_to_green', colors, N=256)


        '''simulation loop'''
        time_start = time.time()
        starting_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_iteration = int(T / delta_T)


        choosed_iterations = [500 * step - 1, 1000 * step - 1, 1500 * step - 1, 2000 * step - 1]
        choosed_iterations = np.array(choosed_iterations, dtype=np.int_)
        for iter in choosed_iterations:
            if True:

                sim_time = (iter + 1) * delta_T
                # print('current iteration', iter + 1, '/', total_iteration, 'sim_time', "{:.3f}".format(sim_time))
                savemat_name = 'UrbanEpidemicalPattern T=' + "{:.4f}".format(sim_time) + '.mat'
                S0 = scio.loadmat(Data_path + savemat_name)['S0'][0]
                I0 = scio.loadmat(Data_path + savemat_name)['I0'][0]
                sim_time = scio.loadmat(Data_path + savemat_name)['sim_time'][0][0]
                print('sim_time:' + "{:.4f}".format(sim_time), 'min and max of I0:', np.min(I0), np.max(I0), np.mean(S0 + I0))

                sim_time = (iter + 1) * delta_T
                savemat_name = 'UrbanEpidemicalPattern T=' + "{:.4f}".format(sim_time) + '.mat'
                scio.savemat(FigurePath + savemat_name, {'S0': S0, 'I0': I0, 'sim_time': sim_time})

                fig, ax = plt.subplots(figsize=(8, 8))
                ax.scatter(notnan_position_xs, notnan_position_ys, c=I0, s=node_areas * 5, marker='.', cmap=custom_cmap,
                           clim=(0, 45))
                gdf.plot(ax=ax, facecolor='none', edgecolor='gray', linewidth=0.3)
                ax.set_xlim(115.4, 117.53)
                ax.set_ylim(39.35, 41.08)
                ax.set_axis_off()

                x_min, x_max = 116.15-0.05, 116.605-0.05
                y_min, y_max = 39.70, 40.05

                ax.plot([x_min, x_max, x_max, x_min, x_min],
                        [y_min, y_min, y_max, y_max, y_min],
                        color='black', linewidth=0.8)

                ax.plot([x_max, 116.923],
                        [y_max, 39.76],
                        color='black', linewidth=0.8)
                ax.plot([x_max, 116.923],
                        [y_min, 39.35],
                        color='black', linewidth=0.8)

                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                ax_inset = inset_axes(ax, width="25%", height="25%", loc='upper left',
                       bbox_to_anchor=(0.7, -0.74, 1, 1), bbox_transform=ax.transAxes)

                ax_inset.scatter(notnan_position_xs, notnan_position_ys, c=I0, s=node_areas * 45, marker='.',
                                 cmap=custom_cmap, clim=(0, 45))
                gdf.plot(ax=ax_inset, facecolor='none', edgecolor='gray', linewidth=0.3)
                ax_inset.set_xlim(x_min, x_max)
                ax_inset.set_ylim(y_min, y_max)
                ax_inset.set_xticks([])
                ax_inset.set_yticks([])
                plt.savefig(FigurePath + 'Urban Infectious ' + "{:.0f}".format(sim_time) + '.eps', dpi=500, bbox_inches='tight', pad_inches=0.3)
                plt.savefig(FigurePath + 'Urban Infectious ' + "{:.0f}".format(sim_time) + '.png', dpi=500, bbox_inches='tight', pad_inches=0.3)
                plt.close()
                # plt.show()




        time_end = time.time()
        elapsed_time = time_end - time_start
        print("Simulation took      : %1.1f (s)" % (elapsed_time))

        ending_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("starting datetime: ", starting_datetime, "\nending datetime: ", ending_datetime)