import numpy as np
import os
import pandas as pd
import geopandas as gpd
import time
from datetime import datetime

from matplotlib import font_manager
font_path = 'C:/Windows/Fonts/STSONG.TTF'
font_prop = font_manager.FontProperties(fname=font_path)

time_start = time.time()
starting_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


UrbanLandscape_path = './Gridded Urban Landscape Beijing'
if not os.path.exists(UrbanLandscape_path):
    os.makedirs(UrbanLandscape_path)

shp_file = "./dataBeijing/beijingshp/BeijingTownshipStreetBoundary.shp"

gdf = gpd.read_file(shp_file)
print(gdf.columns)

# print(dir(gdf))
# print('gdf.head():', gdf.head())

property_name = '乡镇级'
unique_city_names = gdf[property_name].unique()

population_density = pd.DataFrame(gdf['密度'], columns=['密度'])
print('shape of population_density: ', population_density.shape)
min_population_density = np.min(population_density)
print('min_population_density: ', min_population_density)
times_for_min_population_density = population_density/min_population_density
max_times_for_min_population_density = np.max(times_for_min_population_density)
print('max_times_for_min_population_density: ', max_times_for_min_population_density)
base_numberpoints = 1

numberpoints = np.asarray(np.ceil(times_for_min_population_density) * base_numberpoints, dtype=np.int_)
# print('numberpoints: ', numberpoints)
max_numberpoints = np.max(numberpoints)
total_numberpoints = np.sum(numberpoints)
print('max_numberpoints: ', max_numberpoints)
print('total_numberpoints: ', total_numberpoints)

print('shape of gdf: ', gdf.shape)
gdf["numberpoints"] = numberpoints
print('shape of gdf: ', gdf.shape)


import matplotlib.pyplot as plt

''' grid administrative regions '''
import numpy as np

property_name = '区县级'
property_name = '乡镇级'
city_names = gdf['乡镇级']
print('city_names:\n ', city_names)
print('shape of city_names: ', city_names.shape)
# input()
for city_n in np.arange(city_names.shape[0]):

    city = city_names[city_n]

    sub_gdf = gdf[gdf[property_name] == city].copy()
    expected_in_points = sub_gdf['numberpoints'].values[0]


    from geocube.api.core import make_geocube

    np.random.seed(0)
    sub_gdf["variable"] = np.random.randint(1, 100, sub_gdf.shape[0])

    ########################################################################################################################
    top_resolution = 0.1; low_resolution = 0.0000000001;
    Iter = 0;
    gridded_in_points = 10000
    while abs(gridded_in_points - expected_in_points)>=1:
        Iter += 1
        if Iter >= 1000:
            if abs(gridded_in_points - expected_in_points) <= 5:
                print('The maximum number of iterations has been reached, and a poor resolution has been found, which is acceptable')
                break
            else:
                print('The maximum number of iterations has been reached, and the best resolution could not be found')
                gwresolution = np.NAN
                break
        gwresolution = (top_resolution+low_resolution) / 2
        out_dataset1 = make_geocube(vector_data=sub_gdf, output_crs="epsg:4326",
                                    resolution=(gwresolution, gwresolution), measurements=["variable"])

        data = out_dataset1["variable"].values
        notnan_position = np.where(~np.isnan(data))
        notnan_position_array = np.asarray(np.where(~np.isnan(data)))
        gridded_in_points = notnan_position_array.shape[1]

        if abs(gridded_in_points - expected_in_points)==0:
            print('Iter', Iter, 'gridded_in_points：', str(gridded_in_points) + ' <===> ' + str(expected_in_points), 'gwresolution: ', gwresolution)
            break

        if gridded_in_points > expected_in_points:
            low_resolution = gwresolution

        if gridded_in_points <= expected_in_points:
            top_resolution = gwresolution


    ########################################################################################################################

    if np.isnan(gwresolution):
        print(city + '  : ' + str(city_n) + ', unsuccessful！！！')
        raise(city + '  : ' + str(city_n) + ', unsuccessful！！！')
    out_dataset1 = make_geocube(vector_data=sub_gdf, output_crs="epsg:4326",
                                resolution=(gwresolution, gwresolution), measurements=["variable"])

    data = out_dataset1["variable"].values
    x = out_dataset1["x"].values
    y = out_dataset1["y"].values
    x2d, y2d = np.meshgrid(x, y)

    print('shape of data:', data.shape)
    print('shape of x:', x.shape, 'shape of y:', y.shape)

    notnan_position = np.where(~np.isnan(data))
    notnan_position_x = x2d[notnan_position]
    notnan_position_y = y2d[notnan_position]

    print('shape of notnan_position_x:', notnan_position_x.shape)
    print('shape of notnan_position_y:', notnan_position_y.shape)

    notnan_values = data[notnan_position]

    nan_position = np.where(~np.isnan(data))
    nan_position_x = x2d[nan_position]
    nan_position_y = y2d[nan_position]

    if notnan_position_y.shape[0] > sub_gdf['numberpoints'].values[0]:
        print(str(notnan_position_y.shape[0]) + ' <===> ' + str(sub_gdf['numberpoints'].values[0]) + '，tune larger')
    if notnan_position_y.shape[0] < sub_gdf['numberpoints'].values[0]:
        print(str(notnan_position_y.shape[0]) + ' <===> ' + str(sub_gdf['numberpoints'].values[0]) + '，tune smaller')
    if notnan_position_y.shape[0] == sub_gdf['numberpoints'].values[0]:
        print(str(notnan_position_y.shape[0]) + ' <===> ' + str(sub_gdf['numberpoints'].values[0]))

    grid_z = data.copy()
    CData = np.random.randint(100, 1000, notnan_position_y.shape)
    for each in np.arange(notnan_position_x.shape[0]):
        x_in = notnan_position[0][each]
        y_in = notnan_position[1][each]
        v_in = CData[each]

        grid_z[x_in, y_in] = v_in


    grid_z = np.flipud(grid_z)
    data = np.flipud(data)

    fig, ax = plt.subplots(figsize=(8, 8))
    gdf.plot(ax=ax)
    sub_gdf.plot(ax=ax, facecolor='red', edgecolor='black', alpha=0.3)
    plt.title(city + '  : ' + str(city_n) + ', ' + str(notnan_position_y.shape[0]) + '<===>' + str(sub_gdf['numberpoints'].values[0]), fontproperties=font_prop)
    plt.scatter(notnan_position_x, notnan_position_y, s=1, c='red', marker='s')
    figure_name = '/GriddedData_' + str(city_n) + '_' + city + ' large.jpg'
    plt.savefig(UrbanLandscape_path + figure_name, bbox_inches='tight', pad_inches=0.5)

    fig, ax = plt.subplots(figsize=(8, 8))
    sub_gdf.plot(ax=ax, facecolor='blue', edgecolor='black', alpha=0.3)
    plt.title(city + '  : ' + str(city_n) + ', ' + str(notnan_position_y.shape[0]) + '<===>' + str(sub_gdf['numberpoints'].values[0]), fontproperties=font_prop)
    plt.scatter(notnan_position_x, notnan_position_y, s=1, c='red', marker='s')
    figure_name = '/GriddedData_' + str(city_n) + '_' + city + '.jpg'
    plt.savefig(UrbanLandscape_path + figure_name, bbox_inches='tight', pad_inches=0.5)
    # plt.ion()
    # plt.show()
    # plt.pause(3)
    plt.close()

    from scipy.io import savemat, loadmat

    mat_name = '/GriddedData_' + str(city_n) + '_' + city + '.mat'
    print('mat_name: ', mat_name)
    if np.abs(notnan_position_y.shape[0] - sub_gdf['numberpoints'].values[0])<10:
        savemat(UrbanLandscape_path + mat_name, {'gwresolution': gwresolution, 'data': data, 'x': x, 'y': y, 'x2d': x2d, 'y2d': y2d, 'notnan_position': notnan_position,
                                            'notnan_position_x': notnan_position_x, 'notnan_position_y': notnan_position_y})

    time.sleep(3)

time_end = time.time()
elapsed_time = time_end - time_start
print("Simulation took      : %1.1f (s)" % (elapsed_time))