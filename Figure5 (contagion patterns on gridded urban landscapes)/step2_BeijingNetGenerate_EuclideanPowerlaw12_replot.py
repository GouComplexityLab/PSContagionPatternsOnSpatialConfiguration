import numpy as np
import os
import geopandas as gpd
import time
from datetime import datetime
from matplotlib import font_manager
from scipy.io import savemat, loadmat
import scipy.sparse as sp

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors

font_path = 'C:/Windows/Fonts/STSONG.TTF'
font_prop = font_manager.FontProperties(fname=font_path)

time_start = time.time()
starting_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


GriddedUrbanLandscape_path = './Gridded Urban Landscape Beijing'
if not os.path.exists(GriddedUrbanLandscape_path):
    os.makedirs(GriddedUrbanLandscape_path)

for n in np.asarray([1]):

    for target_covariance in np.asarray([0.6]):

        max_iterations = 500000

        NetworkonUrbanLandscape_path = './Network on Gridded Urban Landscape Beijing Powerlaw12 cov=' + str(target_covariance)
        if not os.path.exists(NetworkonUrbanLandscape_path):
            os.makedirs(NetworkonUrbanLandscape_path)

        shp_file = "./dataBeijing/beijingshp/BeijingTownshipStreetBoundary.shp"

        gdf = gpd.read_file(shp_file)
        print(gdf.columns)

        # print(dir(gdf))
        # print('gdf.head():', gdf.head())

        # 获取唯一的 CITY_NAME 值
        property_name = '省级'
        property_name = '市级'
        property_name = '区县级'
        property_name = '乡镇级'
        unique_city_names = gdf[property_name].unique()

        population_density = gdf['密度']
        min_population_density = np.min(population_density)
        times_for_min_population_density = population_density/min_population_density
        max_times_for_min_population_density = np.max(times_for_min_population_density)
        base_numberpoints = 1

        numberpoints = np.asarray(np.ceil(times_for_min_population_density) * base_numberpoints, dtype=np.int_)
        max_numberpoints = np.max(numberpoints)
        total_numberpoints = np.sum(numberpoints)
        gdf["numberpoints"] = numberpoints


        ''' grid regions '''
        property_name = '区县级'
        property_name = '乡镇级'
        city_names = gdf['乡镇级']
        print('shape of city_names: ', city_names.shape)

        mat_name_all = '/AllGriddedData_Beijing.mat'
        print('mat_name_all: ', mat_name_all)


        ''' load All Gridded Data Beijing '''
        notnan_position_xs = loadmat(NetworkonUrbanLandscape_path + mat_name_all)['notnan_position_xs'][0]
        notnan_position_ys = loadmat(NetworkonUrbanLandscape_path + mat_name_all)['notnan_position_ys'][0]
        notnan_position_populationdensity_many = loadmat(NetworkonUrbanLandscape_path + mat_name_all)['notnan_position_populationdensity_many'][0]
        notnan_position_labels = loadmat(NetworkonUrbanLandscape_path + mat_name_all)['notnan_position_labels'][0]
        node_areas = loadmat(NetworkonUrbanLandscape_path + mat_name_all)['node_areas'][0]

        points = np.column_stack((notnan_position_xs, notnan_position_ys))

        ''' plot underlying network '''

        # Kmin=3; Kmax=100; mu=3.82370; # average 4
        # Kmin=5; Kmax=100; mu=3.25494; # average 8
        Kmin=8; Kmax=100; mu=3.62310; # average 12
        # Kmin=11; Kmax=100; mu=3.84492; # average 16
        # Kmin=13; Kmax=100; mu=3.51507; # average 20
        # Kmin=16; Kmax=100; mu=3.63496; # average 24

        ''' load info '''
        adjacency_name_npz = '/adjacency_matrix_Beijing.npz'
        adjacency_matrix = sp.load_npz(NetworkonUrbanLandscape_path + adjacency_name_npz)
        N = notnan_position_xs.shape[0]

        real_cov = np.cov(np.sum(adjacency_matrix, 0), notnan_position_populationdensity_many)[0, 1]
        print('real_cov:', real_cov)

        import networkx as nx
        G = nx.from_numpy_array(adjacency_matrix)

        pos = {i: (notnan_position_xs[i], notnan_position_ys[i]) for i in range(N)}
        fig, ax = plt.subplots(figsize=(8, 8))
        gdf.plot(ax=ax, facecolor='salmon', edgecolor='white', linewidth=0.3, alpha=1.0)
        nx.draw_networkx_edges(G, pos, edge_color='gray')
        plt.scatter(notnan_position_xs, notnan_position_ys, s=node_areas * 5, c='powderblue', marker='.')
        ax.set_xlim(115.4, 117.53)
        ax.set_ylim(39.35, 41.08)
        ax.set_axis_off()


        figure_name = '/Network on Gridded Urban Landscape Beijing.jpg'
        plt.savefig(NetworkonUrbanLandscape_path + figure_name, bbox_inches='tight', pad_inches=0.3)
        figure_name = '/Network on Gridded Urban Landscape Beijing.eps'
        plt.savefig(NetworkonUrbanLandscape_path + figure_name, format='eps', dpi=500, bbox_inches='tight', pad_inches=0.3)
        # plt.show()
        plt.close()

        time_end = time.time()
        elapsed_time = time_end - time_start
        print("Simulation took      : %1.1f (s)" % (elapsed_time))