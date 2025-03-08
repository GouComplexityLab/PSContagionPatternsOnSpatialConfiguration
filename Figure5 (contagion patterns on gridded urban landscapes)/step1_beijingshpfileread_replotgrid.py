import numpy as np
import os
import pandas as pd
import geopandas as gpd
import time
from datetime import datetime

from scipy.io import savemat, loadmat

import matplotlib.patches as mpatches
import numpy as np
import matplotlib as mpl

from matplotlib import font_manager
font_path = 'C:/Windows/Fonts/STSONG.TTF'
font_prop = font_manager.FontProperties(fname=font_path)


def add_north(ax, labelsize=12, loc_x=0.9, loc_y=1, width=0.04, height=0.04, pad=0.2):
    """
    add north arrow
    """
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    ylen = maxy - miny
    xlen = maxx - minx
    left = [minx + xlen * (loc_x - width * .5), miny + ylen * (loc_y - pad)]
    right = [minx + xlen * (loc_x + width * .5), miny + ylen * (loc_y - pad)]
    top = [minx + xlen * loc_x, miny + ylen * (loc_y - pad + height)]
    center = [minx + xlen * loc_x, left[1] + (top[1] - left[1]) * .4]
    triangle = mpatches.Polygon([left, top, right, center], color='k')
    ax.text(s='N',
            x=minx + xlen * loc_x,
            y=miny + ylen * (loc_y - pad + height),
            fontsize=labelsize,
            horizontalalignment='center',
            verticalalignment='bottom',
            fontname='Times New Roman')
    ax.add_patch(triangle)

def add_scalebar(ax, lon0, lat0, length, size=0.03):
    '''
    add scalebar
    '''
    # style 3
    ax.hlines(y=lat0, xmin=lon0, xmax=lon0 + length / 111, colors="black", ls="-", lw=3, label='%d km' % (length))
    ax.vlines(x=lon0, ymin=lat0 - size, ymax=lat0 + size, colors="black", ls="-", lw=3)
    ax.vlines(x=lon0 + length / 111, ymin=lat0 - size, ymax=lat0 + size, colors="black", ls="-", lw=3)
    ax.text(lon0 + length / 111, lat0 + size + 0.02, '%d' % (length), horizontalalignment='center',fontsize=24, fontname='Times New Roman')
    ax.text(lon0, lat0 + size + 0.02, '0', horizontalalignment='center',fontsize=24, fontname='Times New Roman')
    ax.text(lon0 + length / 111 + 0.08, lat0 + size + 0.02, 'km', horizontalalignment='left',fontsize=24, fontname='Times New Roman')


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

property_name = '省级'
property_name = '市级'
property_name = '区县级'
property_name = '乡镇级'
# property_name = '乡镇码'
unique_city_names = gdf[property_name].unique()


population_density = pd.DataFrame(gdf['密度'], columns=['密度'])
min_population_density = np.min(population_density)
times_for_min_population_density = population_density/min_population_density
max_times_for_min_population_density = np.max(times_for_min_population_density)

base_numberpoints = 1

numberpoints = np.asarray(np.ceil(times_for_min_population_density) * base_numberpoints, dtype=np.int_)
max_numberpoints = np.max(numberpoints)
total_numberpoints = np.sum(numberpoints)

gdf["numberpoints"] = numberpoints

import matplotlib.pyplot as plt

''' grid process'''
import numpy as np

property_name = '区县级'
property_name = '乡镇级'
city_names = gdf['乡镇级']

''' plot density '''

fig, ax = plt.subplots(figsize=(8, 8))

bounds = [0, 0.04, 0.10, 0.20, 0.40, 0.60, 1.0, 1.6, 2.2, 3.0, 6.2] # 定义区间
labels = ['0 - 400', '400-1000', '1000-2000', '2000-4000', '4000-6000', '6000-10000', '10000-16000', '16000-22000', '22000-30000', '30000-62000']

cmap = plt.get_cmap('jet')
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
gdf.plot(column=gdf['密度'], ax=ax, edgecolor='white', linewidth=0.3, alpha=1.0, cmap=cmap, norm=norm)
pos_x_text = 0.73
pos_y_text = 0.30
for i in range(len(bounds)-1):
    color = cmap(norm(bounds[i]))
    ax.add_patch(mpatches.Rectangle((pos_x_text, pos_y_text - i * 0.035), 0.1, 0.03,
                                    facecolor=color, transform=ax.transAxes, clip_on=False))
    ax.text(pos_x_text+0.12, pos_y_text - i * 0.035 + 0.01, labels[i], transform=ax.transAxes,
            verticalalignment='center', fontname='Times New Roman', fontsize=18)

ax.text(0.70, 0.35, 'population density', transform=ax.transAxes, fontsize=24, fontname='Times New Roman')
add_north(ax, labelsize=36)
ax.set_xlim(115.4, 117.53)
ax.set_ylim(39.35, 41.08)

add_scalebar(ax, lon0=115.5, lat0=39.4, length=50)

ax.set_axis_off()
figure_name = './Heat map of population density in towns and streets in Beijing.jpg'
plt.savefig(figure_name, dpi=500, bbox_inches='tight', pad_inches=0.3)
figure_name = './Heat map of population density in towns and streets in Beijing.eps'
plt.savefig(figure_name, format='eps', dpi=500, bbox_inches='tight', pad_inches=0.3)


''' plot nodes '''
AllGriddedData_path = './Gridded Urban Landscape Beijing'
mat_name_all = '/AllGriddedData_Beijing.mat'

''' save  All Gridded Data Beijing '''
notnan_position_xs = np.array([])
notnan_position_ys = np.array([])
notnan_position_populationdensity_many = np.array([])
notnan_position_labels = np.array([])
node_areas = np.array([])

for city_n in np.arange(city_names.shape[0]):
    city = city_names[city_n]

    sub_gdf = gdf[gdf[property_name] == city].copy()
    population_density_one = sub_gdf['密度'].values[0]
    sub_area = sub_gdf['面积'].values[0]

    expected_in_points = sub_gdf['numberpoints'].values[0]

    mat_name = '/GriddedData_' + str(city_n) + '_' + city + '.mat'

    gwresolution = loadmat(AllGriddedData_path + mat_name)['gwresolution']
    data = loadmat(AllGriddedData_path + mat_name)['data']
    x = loadmat(AllGriddedData_path + mat_name)['x']
    y = loadmat(AllGriddedData_path + mat_name)['y']
    x2d = loadmat(AllGriddedData_path + mat_name)['x2d']
    y2d = loadmat(AllGriddedData_path + mat_name)['y2d']
    notnan_position = loadmat(AllGriddedData_path + mat_name)['notnan_position']
    notnan_position_x = loadmat(AllGriddedData_path + mat_name)['notnan_position_x'][0]
    notnan_position_y = loadmat(AllGriddedData_path + mat_name)['notnan_position_y'][0]

    gridded_in_points = notnan_position_y.shape[0]

    notnan_position_xs = np.append(notnan_position_xs, notnan_position_x)
    notnan_position_ys = np.append(notnan_position_ys, notnan_position_y)
    notnan_position_labels = np.append(notnan_position_labels, np.ones_like(notnan_position_y) * (city_n + 1))
    notnan_position_populationdensity_many = np.append(notnan_position_populationdensity_many,
                                                       np.ones_like(notnan_position_y) * population_density_one)

    node_area = sub_area / notnan_position_x.shape[0] * np.ones_like(notnan_position_x)
    node_areas = np.append(node_areas, node_area)
    # time.sleep(3)

savemat(AllGriddedData_path + mat_name_all,
        {'notnan_position_xs': notnan_position_xs,
         'notnan_position_ys': notnan_position_ys,
         'notnan_position_populationdensity_many': notnan_position_populationdensity_many,
         'notnan_position_labels': notnan_position_labels,
         'node_areas': node_areas})

''' load All Gridded Data Beijing '''
notnan_position_xs = loadmat(AllGriddedData_path + mat_name_all)['notnan_position_xs'][0]
notnan_position_ys = loadmat(AllGriddedData_path + mat_name_all)['notnan_position_ys'][0]
notnan_position_populationdensity_many = \
loadmat(AllGriddedData_path + mat_name_all)['notnan_position_populationdensity_many'][0]
notnan_position_labels = loadmat(AllGriddedData_path + mat_name_all)['notnan_position_labels'][0]
node_areas = loadmat(AllGriddedData_path + mat_name_all)['node_areas'][0]

fig, ax = plt.subplots(figsize=(8, 8))
gdf.plot(ax=ax, facecolor='salmon', edgecolor='white', linewidth=0.3, alpha=1.0)
plt.scatter(notnan_position_xs, notnan_position_ys, s=node_areas*5, c='powderblue', marker='.')

ax.set_xlim(115.4, 117.53)
ax.set_ylim(39.35, 41.08)
ax.set_axis_off()
figure_name = './Grid Points for Urban Landscape Beijing.jpg'
plt.savefig(figure_name, bbox_inches='tight', dpi=500, pad_inches=0.3)
figure_name = './Grid Points for Urban Landscape Beijing.eps'
plt.savefig(figure_name, format='eps', dpi=500, bbox_inches='tight', pad_inches=0.3)

time_end = time.time()
elapsed_time = time_end - time_start
print("Simulation took      : %1.1f (s)" % (elapsed_time))