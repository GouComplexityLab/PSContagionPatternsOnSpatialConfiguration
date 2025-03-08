import numpy as np
import os
# import pandas as pd
import geopandas as gpd
import time
from datetime import datetime
from matplotlib import font_manager
from scipy.io import savemat, loadmat
import scipy.sparse as sp

# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors


def my_generator_of_random_number_following_power_law_distribution(Kmin, KMax, mu):
    """
    Function to generate random number following power law distribution
    :param m:  the minumam degree
    :param K:  the maximum degree
    :return: random2 ( degree as a random variable)
    """
    x = np.arange(Kmin, KMax + 1, dtype=np.float_)
    y = np.power(x, - mu)
    y = np.true_divide(y, np.sum(y))
    yy = np.cumsum(y)
    random1 = np.random.rand(1)
    position = np.where(yy >= random1)
    first_position = int(position[0][0])
    random2 = x[first_position]

    return random2


font_path = 'C:/Windows/Fonts/STSONG.TTF'
font_prop = font_manager.FontProperties(fname=font_path)

time_start = time.time()
starting_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


GriddedUrbanLandscape_path = './Gridded Urban Landscape Beijing'
if not os.path.exists(GriddedUrbanLandscape_path):
    os.makedirs(GriddedUrbanLandscape_path)

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


    ''' grid process '''
    property_name = '区县级'
    property_name = '乡镇级'
    city_names = gdf['乡镇级']
    print('shape of city_names: ', city_names.shape)

    mat_name_all = '/AllGriddedData_Beijing.mat'
    print('mat_name_all: ', mat_name_all)

    ''' save  All Gridded Data Beijing '''
    notnan_position_xs = np.array([])
    notnan_position_ys = np.array([])
    notnan_position_populationdensity_many = np.array([])
    notnan_position_labels = np.array([])
    node_areas = np.array([])

    # for city_n in np.asarray([206]):
    for city_n in np.arange(city_names.shape[0]):

        city = city_names[city_n]

        sub_gdf = gdf[gdf[property_name] == city].copy()
        population_density_one = sub_gdf['密度'].values[0]
        sub_area = sub_gdf['面积'].values[0]

        expected_in_points = sub_gdf['numberpoints'].values[0]

        mat_name = '/GriddedData_' + str(city_n) + '_' + city + '.mat'
        gwresolution = loadmat(GriddedUrbanLandscape_path + mat_name)['gwresolution']
        data = loadmat(GriddedUrbanLandscape_path + mat_name)['data']
        x = loadmat(GriddedUrbanLandscape_path + mat_name)['x']
        y = loadmat(GriddedUrbanLandscape_path + mat_name)['y']
        x2d = loadmat(GriddedUrbanLandscape_path + mat_name)['x2d']
        y2d = loadmat(GriddedUrbanLandscape_path + mat_name)['y2d']
        notnan_position = loadmat(GriddedUrbanLandscape_path + mat_name)['notnan_position']
        notnan_position_x = loadmat(GriddedUrbanLandscape_path + mat_name)['notnan_position_x'][0]
        notnan_position_y = loadmat(GriddedUrbanLandscape_path + mat_name)['notnan_position_y'][0]

        gridded_in_points = notnan_position_y.shape[0]

        notnan_position_xs = np.append(notnan_position_xs, notnan_position_x)
        notnan_position_ys = np.append(notnan_position_ys, notnan_position_y)
        notnan_position_labels = np.append(notnan_position_labels, np.ones_like(notnan_position_y)*(city_n+1))
        notnan_position_populationdensity_many = np.append(notnan_position_populationdensity_many, np.ones_like(notnan_position_y)*population_density_one)

        node_area = sub_area / notnan_position_x.shape[0] * np.ones_like(notnan_position_x)
        node_areas = np.append(node_areas, node_area)
        # time.sleep(3)  # 暂停3秒钟

    savemat(NetworkonUrbanLandscape_path + mat_name_all,
            {'notnan_position_xs': notnan_position_xs,
             'notnan_position_ys': notnan_position_ys,
             'notnan_position_populationdensity_many': notnan_position_populationdensity_many,
             'notnan_position_labels': notnan_position_labels,
             'node_areas': node_areas})

    # input()
    ''' load All Gridded Data Beijing '''
    notnan_position_xs = loadmat(NetworkonUrbanLandscape_path + mat_name_all)['notnan_position_xs'][0]
    notnan_position_ys = loadmat(NetworkonUrbanLandscape_path + mat_name_all)['notnan_position_ys'][0]
    notnan_position_populationdensity_many = loadmat(NetworkonUrbanLandscape_path + mat_name_all)['notnan_position_populationdensity_many'][0]
    notnan_position_labels = loadmat(NetworkonUrbanLandscape_path + mat_name_all)['notnan_position_labels'][0]

    # 生成随机点集
    points = np.column_stack((notnan_position_xs, notnan_position_ys))
    print('shape of points:', points.shape)
    point_n = 1800
    target_point = np.array([points[point_n]])

    import joblib

    ''' Euclidean distance '''
    nbrs_euclidean = NearestNeighbors(n_neighbors=50000, algorithm='kd_tree').fit(points)
    joblib.dump(nbrs_euclidean, NetworkonUrbanLandscape_path+'nearest_neighbors_model_euclidean.joblib')
    nbrs = joblib.load(NetworkonUrbanLandscape_path+'nearest_neighbors_model_euclidean.joblib')
    distances, indices = nbrs.kneighbors(target_point)
    nearest_points = points[indices]

    ''' Non Euclidean distance '''
    # points_radians = np.radians(points)
    # n_neighbors=300
    # nbrs_NonEuclidean = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', metric='haversine').fit(points_radians)
    #
    # joblib.dump(nbrs_NonEuclidean, NetworkonUrbanLandscape_path+'/nearest_neighbors_model_NonEuclidean.joblib')
    #
    # nbrs = joblib.load(NetworkonUrbanLandscape_path+'/nearest_neighbors_model_NonEuclidean.joblib')
    #
    # target_point_radians = np.radians(target_point)
    # distances_radians, indices = nbrs.kneighbors(target_point_radians)
    # distances = distances_radians * 6371
    # nearest_points = points[indices]

    ''' generate network '''

    # Kmin=3; Kmax=100; mu=3.82370; # average 4
    # Kmin=5; Kmax=100; mu=3.25494; # average 8
    Kmin=8; Kmax=100; mu=3.62310; # average 12
    # Kmin=11; Kmax=100; mu=3.84492; # average 16
    # Kmin=13; Kmax=100; mu=3.51507; # average 20
    # Kmin=16; Kmax=100; mu=3.63496; # average 24


    from scipy.sparse import lil_matrix
    from scipy.sparse import csr_matrix
    from scipy.sparse import diags

    N = notnan_position_xs.shape[0]
    print('N:', N)

    # sparse adjacent matrix
    lil_adjacent_matrix = lil_matrix((N, N))

    # randomly sort the nodes
    randindex = np.array(range(N))
    np.random.shuffle(randindex)
    node_index = np.array(range(N), dtype=np.int64)
    node_index = node_index[randindex]

    prenode_degree = np.zeros(N, dtype=np.int64)
    for i in range(N):
        prenode_degree[i] = my_generator_of_random_number_following_power_law_distribution(Kmin, Kmax, mu)
        # prenode_degree[i] = 4

    def exchange_process(x, y, target_covariance=0.1, max_iterations=1000):
        """
        Adjust the covariance of the series of random variables (x, y) to reach the target_covariance.
        """
        cov = np.cov(x, y)[0, 1]

        iteration = 0
        while np.abs(cov - target_covariance) > 0.001 and iteration < max_iterations:
            p, q = np.random.choice(len(x), 2, replace=False)
            if (x[p] - x[q]) * (y[p] - y[q]) < 0:
                # Swap the x values
                x[p], x[q] = x[q], x[p]

            cov = np.cov(x, y)[0, 1]
            iteration += 1

        if iteration >= max_iterations:
            print('The maximum number of exchanges has been reached')
        if np.abs(cov - target_covariance) <= 0.001:
            print('The correlation level has reached the expected value')
        return x, y, cov

    cov = np.cov(prenode_degree, notnan_position_populationdensity_many)[0, 1]
    print('original cov:', cov)

    # target_covariance = 0.5
    # max_iterations = 50000
    adjusted_prenode_degree, adjusted_notnan_position_populationdensity_many, final_cov = exchange_process(prenode_degree, notnan_position_populationdensity_many, target_covariance, max_iterations)
    print('final cov:', final_cov)

    mat_name = '/adjusted_prenode_degree.mat'
    savemat(NetworkonUrbanLandscape_path + mat_name,{'adjusted_prenode_degree': adjusted_prenode_degree})

    prenode_degree = adjusted_prenode_degree

    adjusted_cov = np.cov(prenode_degree, notnan_position_populationdensity_many)[0, 1]
    print('adjusted cov:', adjusted_cov)

    real_node_degree = np.zeros(N, dtype=np.int64)

    node_count = 0
    attension_node_count = 0
    for node in node_index:
        node_count += 1
        if np.mod(node_count, 100) == 0:
            print('target_covariance:', target_covariance, adjusted_cov, 'node_count:', node_count)

        target_point = np.array([points[node]])
        distances_euclidean, indices_neighbors = nbrs.kneighbors(target_point)
        distances = distances_euclidean

        connected_nodes = np.squeeze(indices_neighbors)
        connected_distance = np.squeeze(distances)

        # Shuffle indices for points with the same distance
        unique_distances, inverse_indices = np.unique(connected_distance, return_inverse=True)

        shuffled_indices = []
        shuffled_same_distance_indices_extend = []

        for d in range(1, len(unique_distances)):

            same_distance_indices = np.where(inverse_indices == d)[0]
            shuffled_same_distance_indices = np.random.permutation(same_distance_indices)
            shuffled_indices_neighbors = connected_nodes[shuffled_same_distance_indices]


            should_break = False
            for neighbor_node in shuffled_indices_neighbors:

                if (real_node_degree[node] >= prenode_degree[node]):
                    should_break = True
                    break
                elif (lil_adjacent_matrix[node, neighbor_node] == 0) \
                        and (real_node_degree[neighbor_node] < prenode_degree[neighbor_node]) \
                        and (node != neighbor_node):
                    real_node_degree[node] += 1
                    real_node_degree[neighbor_node] += 1
                    lil_adjacent_matrix[node, neighbor_node] = 1
                    lil_adjacent_matrix[neighbor_node, node] = 1

            if should_break:
                break

        if (real_node_degree[node] < prenode_degree[node]):
            attension_node_count += 1
            print('attension case ===> ', 'attension_node_count', attension_node_count, 'node:', node,
                  'real_node_degree[node]:', real_node_degree[node], 'prenode_degree[node]:', prenode_degree[node])

    print('total counted node:', node_count)

    csr_adjacent_matrix = csr_matrix(lil_adjacent_matrix, shape=(N, N))
    degree = np.sum(lil_adjacent_matrix, 0).tolist()
    lil_laplacian_matrix = lil_adjacent_matrix - diags(degree[0])
    csr_laplacian_matrix = csr_matrix(lil_laplacian_matrix, shape=(N, N))

    ''' adjacency matrix '''
    adjacency_matrix = csr_adjacent_matrix
    adjacency_matrix = adjacency_matrix.astype(float)

    nodes_degree = np.array(adjacency_matrix.sum(axis=1)).flatten()

    ''' laplacian matrix '''
    laplacian_matrix = csr_laplacian_matrix
    laplacian_matrix = laplacian_matrix.astype(float)

    ''' save info '''

    # save matrices as npy
    laplacian_name_npz = '/laplacian_matrix_Beijing.npz'
    sp.save_npz(NetworkonUrbanLandscape_path + laplacian_name_npz, laplacian_matrix)
    adjacency_name_npz = '/adjacency_matrix_Beijing.npz'
    sp.save_npz(NetworkonUrbanLandscape_path + adjacency_name_npz, adjacency_matrix)

    # save matrices as mat
    adjacency_name_mat = '/adjacency_matrix_Beijing.mat'
    savemat(NetworkonUrbanLandscape_path + adjacency_name_mat, {'adjacency_matrix': adjacency_matrix})
    laplacian_name_mat = '/laplacian_matrix_Beijing.mat'
    savemat(NetworkonUrbanLandscape_path + laplacian_name_mat, {'laplacian_matrix': laplacian_matrix})

    import networkx as nx
    G = nx.from_numpy_array(adjacency_matrix)

    if nx.is_connected(G):
        print('network is connected.')

    pos = {i: (notnan_position_xs[i], notnan_position_ys[i]) for i in range(N)}
    fig, ax = plt.subplots(figsize=(8, 8))
    gdf.plot(ax=ax, facecolor='salmon', edgecolor='white', linewidth=0.3, alpha=1.0)
    nx.draw_networkx_edges(G, pos, edge_color='gray')
    plt.scatter(notnan_position_xs, notnan_position_ys, s=node_areas * 5, c='powderblue', marker='.')

    ax.set_xlim(115.4, 117.53)
    ax.set_ylim(39.35, 41.08)
    ax.set_axis_off()

    figure_name = './Network on Gridded Urban Landscape Beijing.jpg'
    plt.savefig(NetworkonUrbanLandscape_path + figure_name, bbox_inches='tight', pad_inches=0.3)
    figure_name = './Network on Gridded Urban Landscape Beijing.eps'
    plt.savefig(NetworkonUrbanLandscape_path + figure_name, format='eps', dpi=500, bbox_inches='tight', pad_inches=0.3)
    plt.close()

    time_end = time.time()
    elapsed_time = time_end - time_start
    print("Simulation took      : %1.1f (s)" % (elapsed_time))