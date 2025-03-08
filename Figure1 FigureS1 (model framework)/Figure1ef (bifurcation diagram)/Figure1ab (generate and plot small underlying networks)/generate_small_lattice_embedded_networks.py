# -*- coding: utf-8 -*-
"""
Created on Tuesday, 28 May 2019

generate some kinds of lattice embedded networks

"""
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import diags
import scipy.io as scio

import networkx as nx

def setting_node_arrangement(L=200):
    """
    Function to the arrangement of nodes
    L: lattice length
    """

    # node lattice arrangement

    # generate node position coordinates
    X = np.linspace(0, L, L+1)
    [X, Y] = np.meshgrid(X, X)

    X2 = X[0:-1, 0:-1] + 1 / 2
    Y2 = Y[0:-1, 0:-1] + 1 / 2

    X2 = np.transpose(X2)
    Y2 = np.transpose(Y2)

    x = X2.flatten()
    y = Y2.flatten()

    return x, y

def plotting_node_arrangement(L=7, nodes_position_case='in',text_state=True):
    """
    Function to the arrangement of nodes
    L: lattice length
    """

    # node lattice arrangement

    # set parameters
    N = L * L # lattice size

    # generate node position coordinates
    X = np.linspace(0, L, L+1)
    [X, Y] = np.meshgrid(X, X)


    if nodes_position_case == 'in':
        X2 = X[0:-1, 0:-1] + 1 / 2
        Y2 = Y[0:-1, 0:-1] + 1 / 2
    elif nodes_position_case == 'left_bottom':
        X2 = X[0:-1, 0:-1]
        Y2 = Y[0:-1, 0:-1]
    else:
        raise Exception('An Error nodes position case setting')

    X2 = np.transpose(X2)
    Y2 = np.transpose(Y2)

    x = X2.flatten()
    y = Y2.flatten()

    size = 0.5
    # color = np.random.normal(0, 1, len(x))
    fig = plt.figure(figsize=(6, 6))

    ax = fig.add_subplot(1, 1, 1)

    # ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)

    ax.scatter(x, y, s=size, color='black', alpha=1) # scatter plot
    if text_state:
        for index in range(0, N):
            ax.text(x[index] + 0.1, y[index] - 0.1, str(index+1), fontsize=8)


    ax.plot([0, L], [L, L], color='black', lw=3, alpha=0.3)
    ax.plot([L, L], [0, L], color='black', lw=3, alpha=0.3)
    ax.plot([0, L], [0, 0], color='black', lw=3, alpha=0.3)
    ax.plot([0, 0], [0, L], color='black', lw=3, alpha=0.3)

    # plt.xticks([])
    # plt.yticks([])
    T = 0.01
    plt.xlim([0 - T, L + T])
    plt.ylim([0 - T, L + T])

    plt.axis('off')
    # ax.axis('off')

    plt.tight_layout()

    return L, x, y, ax

def computing_distance_between_two_nodes_with_periodic_boundary(x1, y1, x2, y2, L):
    """
    function to compute the distance between two nodes in domain [0,L] * [0,L] with periodic boundary
    (x1, y1): the coordinate of one node,
    (x2, y2): the coordinate of the other node,
    L:  the length of square domain
    return: distance, position_cases

    """

    DisVector = np.zeros(9)

    DisVector[0] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) # case 1
    DisVector[1] = np.sqrt((x1 + L - x2) ** 2 + (y1 - y2) ** 2)  # case 2
    DisVector[2] = np.sqrt((x1 - x2) ** 2 + (y1 + L - y2) ** 2)  # case 3
    DisVector[3] = np.sqrt((x1 - L - x2) ** 2 + (y1 - y2) ** 2)  # case 4
    DisVector[4] = np.sqrt((x1 - x2) ** 2 + (y1 -L - y2) ** 2)  # case 5

    DisVector[5] = np.sqrt((x1 + L - x2) ** 2 + (y1 + L - y2) ** 2)  # case 6
    DisVector[6] = np.sqrt((x1 - L - x2) ** 2 + (y1 - L - y2) ** 2)  # case 7
    DisVector[7] = np.sqrt((x1 + L - x2) ** 2 + (y1 - L - y2) ** 2)  # case 8
    DisVector[8] = np.sqrt((x1 - L - x2) ** 2 + (y1 + L - y2) ** 2)  # case 9
    # print(DisVector)
    distance = np.min(DisVector)
    position_cases = np.argmin(DisVector) + 1

    return distance, position_cases

def myplot(ax, x1, y1, x2, y2, L, position_case, color_name, mylinewidth=1.5):
    """
    function to plot the line between two nodes in domain [0,L] * [0,L] with periodic boundary
    (x1, y1): the coordinate of one node,
    (x2, y2): the coordinate of the other node,
    L:  the length of square domain
    position_case:  the position case of the two nodes
    """
    # print("(", x1, ",", y1, ")")
    # print("(", x2, ",", y2, ")")
    # mylinewidth = 3
    alpha_v = 1
    if position_case == 1:
        ax.plot([x2, x1], [y2, y1], c=color_name, lw=mylinewidth, alpha=alpha_v)
    elif position_case == 2:
        x01 = x1 + L
        y01 = y1
        # x02 = x2 - L
        # y02 = y2
        y10 = (y01 - y2) / (x01 - x2) * (L - x01) + y01
        ax.plot([x2, L], [y2, y10], c=color_name, lw=mylinewidth, alpha=alpha_v)
        ax.plot([0, x1], [y10, y1], c=color_name, lw=mylinewidth, alpha=alpha_v)
    elif position_case == 3:
        x01 = x1
        y01 = y1 + L
        x02 = x2
        y02 = y2 - L
        x10 = (x01 - x2) / (y01 - y2) * (L - y01) + x01
        ax.plot([x2, x10], [y2, L], c=color_name, lw=mylinewidth, alpha=alpha_v)
        ax.plot([x10, x1], [0, y1], c=color_name, lw=mylinewidth, alpha=alpha_v)
    elif position_case == 4:
        x01 = x1 - L
        y01 = y1
        x02 = x2 + L
        y02 = y2
        y10 = (y01 - y2) / (x01 - x2) * (0 - x01) + y01
        ax.plot([x2, 0], [y2, y10], c=color_name, lw=mylinewidth, alpha=alpha_v)
        ax.plot([L, x1], [y10, y1], c=color_name, lw=mylinewidth, alpha=alpha_v)
    elif position_case == 5:
        x01 = x1
        y01 = y1 - L
        x02 = x2
        y02 = y2 + L
        x10 = (x01 - x2) / (y01 - y2) * (0 - y01) + x01
        ax.plot([x2, x10], [y2, 0], c=color_name, lw=mylinewidth, alpha=alpha_v)
        ax.plot([x10, x1], [L, y1], c=color_name, lw=mylinewidth, alpha=alpha_v)
    elif position_case == 6:
        x01 = x1 + L
        y01 = y1 + L
        x02 = x2 - L
        y02 = y2 - L
        y10 = (y01 - y2) / (x01 - x2) * (L - x01) + y01
        x10 = (x01 - x2) / (y01 - y2) * (L - y01) + x01
        if x10 <= L:
            x00 = x10
            y00 = L
        else:
            x00 = L
            y00 = y10
        y20 = (y02 - y1) / (x02 - x1) * (0 - x02) + y02
        x20 = (x02 - x1) / (y02 - y1) * (0 - y02) + x02
        if x20 >= 0:
            x11 = x20
            y11 = 0
        else:
            x11 = 0
            y11 = y20

        ax.plot([x2, x00], [y2, y00], c=color_name, lw=mylinewidth, alpha=alpha_v)
        ax.plot([x11, x1], [y11, y1], c=color_name, lw=mylinewidth, alpha=alpha_v)
    elif position_case == 7:
        x01 = x1 - L
        y01 = y1 - L
        x02 = x2 + L
        y02 = y2 + L
        y10 = (y01 - y2) / (x01 - x2) * (0 - x01) + y01
        x10 = (x01 - x2) / (y01 - y2) * (0 - y01) + x01
        if x10 >= 0:
            x00 = x10
            y00 = 0
        else:
            x00 = 0
            y00 = y10
        y20 = (y02 - y1) / (x02 - x1) * (L - x02) + y02
        x20 = (x02 - x1) / (y02 - y1) * (L - y02) + x02
        if x20 <= L:
            x11 = x20
            y11 = L
        else:
            x11 = L
            y11 = y20
        ax.plot([x2, x00], [y2, y00], c=color_name, lw=mylinewidth, alpha=alpha_v)
        ax.plot([x11, x1], [y11, y1], c=color_name, lw=mylinewidth, alpha=alpha_v)
    elif position_case == 8:
        x01 = x1 + L
        y01 = y1 - L
        x02 = x2 - L
        y02 = y2 + L
        y10 = (y01 - y2) / (x01 - x2) * (L - x01) + y01
        x10 = (x01 - x2) / (y01 - y2) * (0 - y01) + x01
        if x10 <= L:
            x00 = x10
            y00 = 0
        else:
            x00 = L
            y00 = y10
        y20 = (y02 - y1) / (x02 - x1) * (0 - x02) + y02
        x20 = (x02 - x1) / (y02 - y1) * (L - y02) + x02
        if x20 >= 0:
            x11 = x20
            y11 = L
        else:
            x11 = 0
            y11 = y20
        ax.plot([x2, x00], [y2, y00], c=color_name, lw=mylinewidth, alpha=alpha_v)
        ax.plot([x11, x1], [y11, y1], c=color_name, lw=mylinewidth, alpha=alpha_v)
    elif position_case == 9:
        x01 = x1 - L
        y01 = y1 + L
        x02 = x2 + L
        y02 = y2 - L
        y10 = (y01 - y2) / (x01 - x2) * (0 - x01) + y01
        x10 = (x01 - x2) / (y01 - y2) * (L - y01) + x01
        if x10 >= 0:
            x00 = x10
            y00 = L
        else:
            x00 = 0
            y00 = y10
        y20 = (y02 - y1) / (x02 - x1) * (L - x02) + y02
        x20 = (x02 - x1) / (y02 - y1) * (0 - y02) + x02
        if x20 <= L:
            x11 = x20
            y11 = 0
        else:
            x11 = L
            y11 = y20
        ax.plot([x2, x00], [y2, y00], c=color_name, lw=mylinewidth, alpha=alpha_v)
        ax.plot([x11, x1], [y11, y1], c=color_name, lw=mylinewidth, alpha=alpha_v)

def my_generator_of_random_number_following_power_law_distribution(m, K, mu):
    """
    Function to generate random number following power law distribution
    :param m:  the minumam degree
    :param K:  the maximum degree
    :return: random2 ( degree as a random variable)
    """
    x = np.arange(m, K+1, dtype=np.float_)
    y = np.power(x, - mu)
    y = np.true_divide(y, np.sum(y))
    yy = np.cumsum(y)
    random1 = np.random.rand(1)
    position = np.where(yy>=random1)
    first_position = int(position[0][0])
    random2 = x[first_position]

    return random2

def my_generetor_of_random_number_following_mypoisson_distribution(lam, K):
    """
    Function to generate random number following poisson distribution
    :param lam: the exponent
    :param K: the restrictive maximum degree, restrict the minimum degree to 1
    :return: random2 ( degree as a random variable)
    """

    x = np.linspace(1, K, K, dtype=np.float64)
    w = np.zeros(K, dtype=np.float64)
    w[0] = 1
    for i in np.arange(K - 1, dtype=np.int64):
        w[i+1] = (i+2) * w[i]

    y = np.zeros(K, float)
    for j in np.linspace(1, K, K, dtype=np.int64):
        y[j-1] = lam ** x[j-1] * np.exp(-lam) / w[j-1]


    y = np.true_divide(y, np.sum(y))
    yy = np.cumsum(y)
    random1 = np.random.rand(1)
    position = np.where(yy>=random1)
    first_position = position[0][0]
    random2 = x[first_position]

    return random2

def find_k_step_closest_neighbors(number_x, number_y, L, k):
    """
    Function to find the (single) k step closest neighbors of the central node with coordinate (number_x, number_y)
    (number_x, number_y) : the coordinate of the central node
    L: the lattice length
    k: the step distance to find closest neighbors
    return k_step_neighbors, k_step_neighbor_cases
    """
    k_step_neighbors = np.zeros(4 * k)
    k_step_neighbor_cases = np.zeros(4 * k)

    preneighbor_number_x = np.zeros((4 * k))
    preneighbor_number_y = np.zeros((4 * k))

    preneighbor_number_x[0] = number_x - k
    preneighbor_number_y[0] = number_y
    num = 1
    for s in range(-(k-1),k):
        a = s
        b = k - np.abs(s)
        preneighbor_number_x[num] = number_x + a
        preneighbor_number_y[num] = number_y + b
        num = num + 1
        preneighbor_number_x[num] = number_x + a
        preneighbor_number_y[num] = number_y - b
        num = num + 1
    preneighbor_number_x[num] = number_x + k
    preneighbor_number_y[num] = number_y
    # print("num=",num)
    if (num+1) == 4 * k:
        for t in range(num+1):
            neighbor_number_x = preneighbor_number_x[t]
            neighbor_number_y = preneighbor_number_y[t]
            # print("L=", L, "neighbor_number_x=", neighbor_number_x, "neighbor_number_y=", neighbor_number_y)
            if neighbor_number_x > 0 and neighbor_number_x <= L and neighbor_number_y > 0 and neighbor_number_y <= L:
                neighbor_case = 1
            if neighbor_number_x <= 0 and neighbor_number_y > 0 and neighbor_number_y <= L:
                neighbor_case = 2
            if neighbor_number_x > 0 and neighbor_number_x <= L and neighbor_number_y <= 0:
                neighbor_case = 3
            if neighbor_number_x > L and neighbor_number_y > 0 and neighbor_number_y <= L:
                neighbor_case = 4
            if neighbor_number_x > 0 and neighbor_number_x <= L and neighbor_number_y > L:
                neighbor_case = 5
            if neighbor_number_x <= 0 and neighbor_number_y <= 0:
                neighbor_case = 6
            if neighbor_number_x > L and neighbor_number_y > L:
                neighbor_case = 7
            if neighbor_number_x <= 0 and neighbor_number_y > L:
                neighbor_case = 8
            if neighbor_number_x > L and neighbor_number_y <= 0:
                neighbor_case = 9
            k_step_neighbor_cases[t] = neighbor_case

            if neighbor_number_x <= 0:
                neighbor_number_x = neighbor_number_x + L
            if neighbor_number_y <= 0:
                neighbor_number_y = neighbor_number_y + L
            if neighbor_number_x > L:
                neighbor_number_x = neighbor_number_x - L
            if neighbor_number_y > L:
                neighbor_number_y = neighbor_number_y - L

            k_step_neighbors[t] = (neighbor_number_x -1) * L + neighbor_number_y
            # print("t=",t,"neighbor_node_index=", (neighbor_number_x - 1) * L + neighbor_number_y)

    return k_step_neighbors, k_step_neighbor_cases

def find_step_neighbors(number_x, number_y, L, k):
    """
    Function to find the (multiple) k steps closest neighbors of the central node with coordinate (number_x, number_y)
    (number_x, number_y) : the coordinate of the central node
    L: the lattice length
    k: the step distance to find closest neighbors
    return neighbors, neighbor_cases
    """
    neighbors = np.zeros(4 * np.sum(range(1,k+1)))
    neighbor_cases = np.zeros(4 * np.sum(range(1,k+1)))

    for k_step in range(1,k+1):
        k_step_neighbors, k_step_neighbor_cases = find_k_step_closest_neighbors(number_x, number_y, L, k_step)

        # print("k_step:", k_step, "k_step_neighbors：", len(k_step_neighbors), "neighbors:\n", k_step_neighbors)
        # print("k_step:", k_step, "k_step_neighbor_cases：", len(k_step_neighbors), "neighbor_cases:\n",
        #       k_step_neighbor_cases)

        # randomly sort the nodes with the same step distance apart from the central node
        randindex = np.array(range(len(k_step_neighbors)))
        np.random.shuffle(randindex)
        k_step_neighbors = k_step_neighbors[randindex]
        k_step_neighbor_cases = k_step_neighbor_cases[randindex]

        neighbors[
        int(4 * np.sum(range(1, k_step))):1+int(4 * np.sum(range(1, k_step + 1)) - 1)] = np.array(k_step_neighbors)
        neighbor_cases[
        int(4 * np.sum(range(1, k_step))):1+int(4 * np.sum(range(1, k_step + 1)) - 1)] = np.array(k_step_neighbor_cases)

    return neighbors, neighbor_cases

def generate_powerlaw_lam_LEN(path, a_name_npz, l_name_npz, L=20, m=3, K=100, mu=2.5, k=10):
    """
    function to generate the lattice embedded power law network
    :param L: lattice length
    :param m: the minimum degree
    :param K: the maximum degree
    :param mu: the power law exponent
    :param k: step distance
    Output the adjacent matrix and laplacian matrix of the generated network
    """

    N = L * L  # lattice size


    # sparse adjacent matrix
    lil_adjacent_matrix = lil_matrix((N, N))

    # randomly sort the nodes
    randindex = np.array(range(N))
    np.random.shuffle(randindex)
    node_index = np.array(range(1,N+1), dtype=np.int64)
    node_index = node_index[randindex]

    prenode_degree = np.zeros(N, dtype=np.int64)
    for i in range(N):
        prenode_degree[i] = my_generator_of_random_number_following_power_law_distribution(m, K, mu)

    real_node_degree = np.zeros(N, dtype=np.int64)

    node_num = 1
    for node in node_index:

        node_num = node_num + 1

        number_y = node % L
        if number_y == 0:
            number_y = L
            number_x = node // L
        else:
            number_x = node // L + 1

        # neighbor_num = 1
        for k_step in range(1, k + 1):
            if (real_node_degree[node - 1] < prenode_degree[node - 1]):
                k_step_neighbors, k_step_neighbor_cases = find_k_step_closest_neighbors(number_x, number_y, L, k_step)

                # randomly sort the nodes with the same step distance apart from the central node
                randindex = np.array(range(len(k_step_neighbors)))
                np.random.shuffle(randindex)
                k_step_neighbors = np.array(k_step_neighbors[randindex])
                k_step_neighbor_cases = np.array(k_step_neighbor_cases[randindex])

                for each in range(len(k_step_neighbors)):
                    neighbor_node = int(k_step_neighbors[each])

                    if node != neighbor_node:

                        if (real_node_degree[node - 1] >= prenode_degree[node - 1]):
                            # continue
                            break
                        elif (lil_adjacent_matrix[node - 1, neighbor_node - 1] == 0) \
                                and (real_node_degree[neighbor_node - 1] < prenode_degree[neighbor_node - 1]):

                            real_node_degree[node - 1] = real_node_degree[node - 1] + 1
                            real_node_degree[neighbor_node - 1] = real_node_degree[neighbor_node - 1] + 1
                            lil_adjacent_matrix[node - 1, neighbor_node - 1] = 1
                            lil_adjacent_matrix[neighbor_node - 1, node - 1] = 1

    csr_adjacent_matrix = csr_matrix(lil_adjacent_matrix, shape=(N, N))
    # lil_laplacian_matrix = lil_matrix((N, N))
    degree = np.sum(lil_adjacent_matrix, 0).tolist()
    lil_laplacian_matrix = lil_adjacent_matrix - diags(degree[0])
    csr_laplacian_matrix = csr_matrix(lil_laplacian_matrix, shape=(N, N))

    sp.save_npz(path + a_name_npz, csr_adjacent_matrix)
    sp.save_npz(path + l_name_npz, csr_laplacian_matrix)

    print("Finish generating the expected power law lattice embedded network,"
          " and saving the adjacent and laplacian matrices of the generated network.")

def generate_mypoisson_lam_LEN(path, a_name_npz, l_name_npz, L=20, lam=4, K=100, k=6):
    """
    function to generate the lattice embedded poisson network
    :param L: lattice length
    :param lam: the Poisson exponent
    :param k: step distance
    Output the adjacent matrix and laplacian matrix of the generated network
    """

    N = L * L  # lattice size


    # sparse adjacent matrix
    lil_adjacent_matrix = lil_matrix((N, N))

    # randomly sort the nodes
    randindex = np.array(range(N))
    np.random.shuffle(randindex)
    node_index = np.array(range(1, N + 1), dtype=np.int64)
    node_index = node_index[randindex]

    prenode_degree = np.zeros(N, dtype=np.int64)
    for i in range(N):
        prenode_degree[i] = my_generetor_of_random_number_following_mypoisson_distribution(lam, K)

    real_node_degree = np.zeros(N, dtype=np.int64)

    node_num = 1
    for node in node_index:

        node_num = node_num + 1

        number_y = node % L
        if number_y == 0:
            number_y = L
            number_x = node // L
        else:
            number_x = node // L + 1

        # neighbor_num = 1
        for k_step in range(1, L+1):
            if (real_node_degree[node - 1] < prenode_degree[node - 1]):
                k_step_neighbors, k_step_neighbor_cases = find_k_step_closest_neighbors(number_x, number_y, L, k_step)

                # randomly sort the nodes with the same step distance apart from the central node
                randindex = np.array(range(len(k_step_neighbors)))
                np.random.shuffle(randindex)
                k_step_neighbors = np.array(k_step_neighbors[randindex], dtype=np.int_)
                k_step_neighbor_cases = np.array(k_step_neighbor_cases[randindex], dtype=np.int_)

                for each in range(len(k_step_neighbors)):
                    neighbor_node = k_step_neighbors[each]

                    if node != neighbor_node:

                        if (real_node_degree[node - 1] == prenode_degree[node - 1]):
                            # continue
                            break
                        elif (lil_adjacent_matrix[node - 1, neighbor_node - 1] == 0) \
                                and (real_node_degree[neighbor_node - 1] < prenode_degree[neighbor_node - 1]):

                            real_node_degree[node - 1] = real_node_degree[node - 1] + 1
                            real_node_degree[neighbor_node - 1] = real_node_degree[neighbor_node - 1] + 1
                            lil_adjacent_matrix[node - 1, neighbor_node - 1] = 1
                            lil_adjacent_matrix[neighbor_node - 1, node - 1] = 1


    csr_adjacent_matrix = csr_matrix(lil_adjacent_matrix, shape=(N, N))
    # lil_laplacian_matrix = lil_matrix((N, N))
    degree = np.sum(lil_adjacent_matrix, 0).tolist()
    lil_laplacian_matrix = lil_adjacent_matrix - diags(degree[0])
    csr_laplacian_matrix = csr_matrix(lil_laplacian_matrix, shape=(N, N))

    sp.save_npz(path + a_name_npz, csr_adjacent_matrix)
    sp.save_npz(path + l_name_npz, csr_laplacian_matrix)

    print("Finish generating the expected mypoisson lattice embedded network,"
          " and saving the adjacent and laplacian matrices of the generated network.")

def generate_poisson_lam_LEN(path, a_name_npz, l_name_npz, L=20, lam=6, k=6):
    """
    function to generate the lattice embedded poisson network
    :param L: lattice length
    :param lam: the Poisson exponent
    :param k: step distance
    Output the adjacent matrix and laplacian matrix of the generated network
    """

    N = L * L  # lattice size


    # sparse adjacent matrix
    lil_adjacent_matrix = lil_matrix((N, N))

    # randomly sort the nodes
    randindex = np.array(range(N))
    np.random.shuffle(randindex)
    node_index = np.array(range(1, N + 1), dtype=np.int64)
    node_index = node_index[randindex]

    prenode_degree = np.random.poisson(lam, size=N)
    real_node_degree = np.zeros(N, dtype=np.int64)

    node_num = 1
    for node in node_index:
        # print('node_num:', node_num)
        # input()

        if node_num % 5000 == 0:
            print('node_num:', node_num)

        node_num = node_num + 1

        number_y = node % L
        if number_y == 0:
            number_y = L
            number_x = node // L
        else:
            number_x = node // L + 1

        # neighbor_num = 1
        for k_step in range(1, k + 1):
            if (real_node_degree[node - 1] < prenode_degree[node - 1]):
                k_step_neighbors, k_step_neighbor_cases = find_k_step_closest_neighbors(number_x, number_y, L, k_step)

                # randomly sort the nodes with the same step distance apart from the central node
                randindex = np.array(range(len(k_step_neighbors)))
                np.random.shuffle(randindex)
                k_step_neighbors = np.array(k_step_neighbors[randindex])
                k_step_neighbor_cases = np.array(k_step_neighbor_cases[randindex])

                for each in range(len(k_step_neighbors)):
                    neighbor_node = int(k_step_neighbors[each])

                    if node != neighbor_node:

                        if (real_node_degree[node - 1] >= prenode_degree[node - 1]):
                            # continue
                            break
                        elif (lil_adjacent_matrix[node - 1, neighbor_node - 1] == 0) \
                                and (real_node_degree[neighbor_node - 1] < prenode_degree[neighbor_node - 1]):

                            # print(real_node_degree[node - 1], 'neighbor_num:', neighbor_num)
                            # input()
                            # neighbor_num = neighbor_num + 1

                            real_node_degree[node - 1] = real_node_degree[node - 1] + 1
                            real_node_degree[neighbor_node - 1] = real_node_degree[neighbor_node - 1] + 1
                            lil_adjacent_matrix[node - 1, neighbor_node - 1] = 1
                            lil_adjacent_matrix[neighbor_node - 1, node - 1] = 1

    csr_adjacent_matrix = csr_matrix(lil_adjacent_matrix, shape=(N, N))
    # lil_laplacian_matrix = lil_matrix((N, N))
    degree = np.sum(lil_adjacent_matrix, 0).tolist()
    lil_laplacian_matrix = lil_adjacent_matrix - diags(degree[0])
    csr_laplacian_matrix = csr_matrix(lil_laplacian_matrix, shape=(N, N))

    sp.save_npz(path + a_name_npz, csr_adjacent_matrix)
    sp.save_npz(path + l_name_npz, csr_laplacian_matrix)

    print("Finish generating the expected poisson lattice embedded network,"
          " and saving the adjacent and laplacian matrices of the generated network.")

def generate_delta_dg_LEN(path, a_name_npz, l_name_npz, L=20, dg=6, k=6):
    """
    function to generate the lattice embedded poisson network
    :param L: lattice length
    :param lam: the Poisson exponent
    :param k: step distance
    Output the adjacent matrix and laplacian matrix of the generated network
    """

    N = L * L  # lattice size


    # sparse adjacent matrix
    lil_adjacent_matrix = lil_matrix((N, N))

    # randomly sort the nodes
    randindex = np.array(range(N))
    np.random.shuffle(randindex)
    node_index = np.array(range(1, N + 1), dtype=np.int64)
    node_index = node_index[randindex]

    prenode_degree = np.ones(N, dtype=np.int64) * dg
    real_node_degree = np.zeros(N, dtype=np.int64)

    node_num = 1
    for node in node_index:
        # print('node_num:', node_num)
        # input()

        # if node_num % 5000 == 0:
        #     print('node_num:', node_num)

        node_num = node_num + 1

        number_y = node % L
        if number_y == 0:
            number_y = L
            number_x = node // L
        else:
            number_x = node // L + 1

        # neighbor_num = 1
        for k_step in range(1, k + 1):
            if (real_node_degree[node - 1] < prenode_degree[node - 1]):
                k_step_neighbors, k_step_neighbor_cases = find_k_step_closest_neighbors(number_x, number_y, L, k_step)

                # randomly sort the nodes with the same step distance apart from the central node
                randindex = np.array(range(len(k_step_neighbors)))
                np.random.shuffle(randindex)
                k_step_neighbors = np.array(k_step_neighbors[randindex], dtype=np.int_)
                k_step_neighbor_cases = np.array(k_step_neighbor_cases[randindex], dtype=np.int_)

                for each in range(len(k_step_neighbors)):
                    neighbor_node = int(k_step_neighbors[each])

                    if node != neighbor_node:

                        if (real_node_degree[node - 1] >= prenode_degree[node - 1]):
                            # continue
                            break
                        elif (lil_adjacent_matrix[node - 1, neighbor_node - 1] == 0) \
                            and (real_node_degree[neighbor_node - 1] < prenode_degree[neighbor_node - 1]):

                            real_node_degree[node - 1] = real_node_degree[node - 1] + 1
                            real_node_degree[neighbor_node - 1] = real_node_degree[neighbor_node - 1] + 1
                            lil_adjacent_matrix[node - 1, neighbor_node - 1] = 1
                            lil_adjacent_matrix[neighbor_node - 1, node - 1] = 1


    csr_adjacent_matrix = csr_matrix(lil_adjacent_matrix, shape=(N, N))
    # lil_laplacian_matrix = lil_matrix((N, N))
    degree = np.sum(lil_adjacent_matrix, 0).tolist()
    lil_laplacian_matrix = lil_adjacent_matrix - diags(degree[0])
    csr_laplacian_matrix = csr_matrix(lil_laplacian_matrix, shape=(N, N))

    sp.save_npz(path + a_name_npz, csr_adjacent_matrix)
    sp.save_npz(path + l_name_npz, csr_laplacian_matrix)

    print("Finish generating the expected delta lattice embedded network,"
          " and saving the adjacent and laplacian matrices of the generated network.")

def generate_k_steps_HoClosestNLEN(path, a_name_npz, l_name_npz, L=5, k=2):
    """
    Function to generate the k steps homogeneous closest neighborhoods lattice-embedded network
    Output the Figure and Adjacent Matrix
    L: lattice length
    """

    N = L * L # lattice size

    # sparse adjacent matrix
    lil_adjacent_matrix = lil_matrix((N, N))

    # print(lil_adjacent_matrix.toarray())
    # print(l.rows)


    # randomly sort the nodes with the same step distance apart from the central node
    randindex = np.array(range(N))
    np.random.shuffle(randindex)
    node_index = np.array(range(1,N+1))
    # node_index = node_index[randindex]
    for node in node_index:

        number_y = node % L
        if number_y == 0:
            number_y = L
            number_x = node // L
        else:
            number_x = node // L + 1
        k_steps_neighbors, k_steps_neighbor_cases = find_k_steps_closest_neighbors(number_x, number_y, L, k)
        for each  in range(len(k_steps_neighbors)):
            neighbor_node = k_steps_neighbors[each]
            position_case = k_steps_neighbor_cases[each]

            if node == neighbor_node:
                 print("Error: node = neighbor_node.")

            lil_adjacent_matrix[int(node)-1, int(neighbor_node)-1] = 1


    csr_adjacent_matrix = csr_matrix(lil_adjacent_matrix, shape=(N, N))
    # lil_laplacian_matrix = lil_matrix((N, N))
    degree = np.sum(lil_adjacent_matrix, 0).tolist()
    lil_laplacian_matrix = lil_adjacent_matrix -diags(degree[0])
    csr_laplacian_matrix = csr_matrix(lil_laplacian_matrix, shape=(N, N))

    sp.save_npz(path + a_name_npz, csr_adjacent_matrix)
    sp.save_npz(path + l_name_npz, csr_laplacian_matrix)

    print("已保存网络的邻接矩阵和拉普拉斯矩阵")

def find_k_steps_closest_neighbors(number_x, number_y, L, k):
    """
    Function to find the (multiple) k steps closest neighbors of the central node with coordinate (number_x, number_y)
    (number_x, number_y) : the coordinate of the central node
    L: the lattice length
    k: the step distance to find closest neighbors
    return k_steps_neighbors, k_steps_neighbor_cases
    """
    k_steps_neighbors = np.zeros(4 * np.sum(range(1,k+1)))
    k_steps_neighbor_cases = np.zeros(4 * np.sum(range(1,k+1)))

    for k_step in range(1,k+1):
        k_step_neighbors, k_step_neighbor_cases = find_k_step_closest_neighbors(number_x, number_y, L, k_step)

        # print("k_step:", k_step, "k_step_neighbors：", len(k_step_neighbors), "neighbors:\n", k_step_neighbors)
        # print("k_step:", k_step, "k_step_neighbor_cases：", len(k_step_neighbors), "neighbor_cases:\n",
        #       k_step_neighbor_cases)

        # # randomly sort the nodes with the same step distance apart from the central node
        # randindex = np.array(range(len(k_step_neighbors)))
        # np.random.shuffle(randindex)
        # k_step_neighbors = k_step_neighbors[randindex]
        # k_step_neighbor_cases = k_step_neighbor_cases[randindex]

        k_steps_neighbors[
        int(4 * np.sum(range(1, k_step))):1+int(4 * np.sum(range(1, k_step + 1)) - 1)] = np.array(k_step_neighbors)
        k_steps_neighbor_cases[
        int(4 * np.sum(range(1, k_step))):1+int(4 * np.sum(range(1, k_step + 1)) - 1)] = np.array(k_step_neighbor_cases)

    return k_steps_neighbors, k_steps_neighbor_cases

def plot_adjacent_and_laplacian_matrixs_of_small_networks(path, a_name_npz, l_name_npz, transform_data=True):
    """
    Function to plot the elements ot adjacent matrix and laplacian matrix
    Output the figures of adjacent matrix and laplacian matrix
    """
    M_A = sp.load_npz(path + a_name_npz).toarray()

    fig1, ax1 = plt.subplots()
    im = ax1.imshow(np.array(M_A, dtype=int))
    for i in range(M_A.shape[0]):
        for j in range(M_A.shape[1]):
            text = ax1.text(j, i, int(M_A[i, j]), ha="center", va="center", color="w")

    plt.xticks(np.linspace(0, M_A.shape[0] - 1, M_A.shape[0]))
    plt.yticks(np.linspace(0, M_A.shape[0] - 1, M_A.shape[0]))

    fig1.tight_layout()
    plt.ion()
    plt.show()
    plt.pause(0.1)

    save_state = True
    if save_state:
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)
        plt.savefig(path + a_name_npz.replace('.npz', '.png'), bbox_inches='tight',pad_inches=0)

    # plot the elements of laplacian matrix  k steps HoClosestNLEN

    M_L = sp.load_npz(path + l_name_npz).toarray()

    fig1, ax1 = plt.subplots()
    im = ax1.imshow(np.array(M_L, dtype=int))
    for i in range(M_L.shape[0]):
        for j in range(M_L.shape[1]):
            text = ax1.text(j, i, int(M_L[i, j]), ha="center", va="center", color="w")

    plt.xticks(np.linspace(0, M_A.shape[0] - 1, M_A.shape[0]))
    plt.yticks(np.linspace(0, M_A.shape[0] - 1, M_A.shape[0]))

    fig1.tight_layout()
    plt.ion()
    plt.show()
    plt.pause(0.1)

    save_state = True
    if save_state:
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)
        plt.savefig(path + l_name_npz.replace('.npz', '.png') ,bbox_inches='tight',pad_inches=0)

    if transform_data:
        transform_path = path + 'txt & mat/'
        isExists = os.path.exists(transform_path)
        if not isExists:
            os.makedirs(transform_path)

        a_name_txt = a_name_npz.replace('.npz', '.txt')
        a_name_mat = a_name_npz.replace('.npz', '.mat')
        M_A.dump(transform_path + a_name_txt)
        M_A.dump(transform_path + a_name_mat)

        l_name_txt = l_name_npz.replace('.npz', '.txt')
        l_name_mat = l_name_npz.replace('.npz', '.mat')
        M_L.dump(transform_path + l_name_txt)
        M_L.dump(transform_path + l_name_mat)

def load_adjacent_and_laplacian_matrixs(path, a_name_npz, l_name_npz):
    """
    load the adjacent matrix and laplacian matrix of one network

    """

    adjacent_matrix = sp.load_npz(path + a_name_npz)
    laplacian_matrix = sp.load_npz(path + l_name_npz)

    # DataName = path + 'MatricesData' + '.mat'
    # scio.savemat(DataName, {'adjacent_matrix':adjacent_matrix,'laplacian_matrix':laplacian_matrix})

    print("Finish loading the adjacent and laplacian matrices.")
    return adjacent_matrix, laplacian_matrix

def generate_many_deltaLEN(L=100, k=6, dg=20, numbers=5):
    # L = 100
    # dg = 18
    # k = 6
    path = './Generated delta Lattice Embedded Networks L=' + str(L) + ' dg=' + str(dg) + '/'
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

    mat_path = path + 'mat_matrices/'
    isExists = os.path.exists(mat_path)
    if not isExists:
        os.makedirs(mat_path)

    number = 1
    while number <= numbers:
        a_name_npz = 'adjacent_matrix_deltaLEN_L=' + str(L) + '_dg=' + str(dg) + '_number=' + str(
            number) + '.npz'
        l_name_npz = 'laplacian_matrix_deltaLEN_L=' + str(L) + '_dg=' + str(dg) + '_number=' + str(
            number) + '.npz'
        # ####################################################################################################################
        generate_delta_dg_LEN(path, a_name_npz, l_name_npz, L, dg, k)

        ####################################################################################################################
        adjacent_matrix, laplacian_matrix = load_adjacent_and_laplacian_matrixs(path, a_name_npz, l_name_npz)

        G = nx.from_scipy_sparse_array(adjacent_matrix)
        print('number=', number, "Is G conneted:", nx.is_connected(G))
        if nx.is_connected(G):
            name_mat = 'matrices_deltaLEN_L=' + str(L) + '_dg=' + str(dg) + '_number=' + str(number) + '.mat'
            scio.savemat(mat_path + name_mat,
                         {'adjacent_matrix': adjacent_matrix, 'laplacian_matrix': laplacian_matrix})
            print('number=', number, "Is G conneted:", nx.is_connected(G))
            number = number + 1

    figure_name = 'the generated deltaLEN with L=' + str(L) + ' k=' + str(k)
    # nodes_position_case = 'left_bottom'
    nodes_position_case = 'in'
    plot_any_LEN(path, figure_name, adjacent_matrix, L, nodes_position_case, text_state=False)

def generate_many_mypoissonLEN(L=100, k=6, lam=6, K=100, numbers=5):
    # numbers = 1
    # L = 20
    # lam = 6
    # K = 100
    # k = 10
    path = './Generated mypoisson Lattice Embedded Networks L=' + str(L) + ' lam=' + str(lam) + '/'
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

    mat_path = path + 'mat_matrices/'
    isExists = os.path.exists(mat_path)
    if not isExists:
        os.makedirs(mat_path)

    number = 1
    while number <= numbers:
        a_name_npz = 'adjacent_matrix_mypoissonLEN L=' + str(L) + ' lam=' + str(lam) + '_number=' + str(number) + '.npz'
        l_name_npz = 'laplacian_matrix_mypoissonLEN L=' + str(L) + ' lam=' + str(lam) + '_number=' + str(number) + '.npz'
        # ####################################################################################################################
        generate_mypoisson_lam_LEN(path, a_name_npz, l_name_npz, L, lam, K, k)

        ####################################################################################################################
        adjacent_matrix, laplacian_matrix = load_adjacent_and_laplacian_matrixs(path, a_name_npz, l_name_npz)

        G = nx.from_scipy_sparse_array(adjacent_matrix)
        print('number=', number, "Is G conneted:", nx.is_connected(G))
        if nx.is_connected(G):
            name_mat = 'matrices_mypoissonLEN_L=' + str(L) + '_lam=' + str(lam) + '_number=' + str(number) + '.mat'
            scio.savemat(mat_path + name_mat,
                         {'adjacent_matrix': adjacent_matrix, 'laplacian_matrix': laplacian_matrix})
            print('number=', number, "Is G conneted:", nx.is_connected(G))
            number = number + 1

    figure_name = 'the generated mypoissonLEN with L=' + str(L) + ' lam=' + str(lam)
    # nodes_position_case = 'left_bottom'
    nodes_position_case = 'in'
    plot_any_LEN(path, figure_name, adjacent_matrix, L, nodes_position_case, text_state=False)

def generate_many_powerlawLEN(L=100, k=6, mu=2.5, m=3, K=100, numbers=50):
    # numbers = 2

    # L = 20
    # mu = 2.5
    # m = 3
    # K = 100
    # k = 10

    path = './Generated powerlaw Lattice Embedded Networks L=' + str(L) + ' mu=' + str(
        mu) + '/'
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    a_name_npz = 'adjacent_matrix_powerlawLEN L=' + str(L) + ' mu=' + str(mu) + '.npz'
    l_name_npz = 'laplacian_matrix_powerlawLEN L=' + str(L) + ' mu=' + str(mu) + '.npz'

    mat_path = path + 'mat_matrices/'
    isExists = os.path.exists(mat_path)
    if not isExists:
        os.makedirs(mat_path)

    number = 1
    while number <= numbers:

        a_name_npz = 'adjacent_matrix_powerlawLEN L=' + str(L) + ' mu=' + str(mu) + '_number=' + str(number) + '.npz'
        l_name_npz = 'laplacian_matrix_powerlawLEN L=' + str(L) + ' mu=' + str(mu) + '_number=' + str(number) + '.npz'
        # ####################################################################################################################
        generate_powerlaw_lam_LEN(path, a_name_npz, l_name_npz, L, m, K, mu, k)

        ####################################################################################################################
        adjacent_matrix, laplacian_matrix = load_adjacent_and_laplacian_matrixs(path, a_name_npz, l_name_npz)

        G = nx.from_scipy_sparse_array(adjacent_matrix)
        print('number=', number, "Is G conneted:", nx.is_connected(G))
        if nx.is_connected(G):
            name_mat = 'matrices_mypowerlawLEN_L=' + str(L) + '_mu=' + str(mu) + '_number=' + str(number) + '.mat'
            scio.savemat(mat_path + name_mat,
                         {'adjacent_matrix': adjacent_matrix, 'laplacian_matrix': laplacian_matrix})
            print('number=', number, "Is G conneted:", nx.is_connected(G))
            number = number + 1

    figure_name = 'the generated mypowerlawLEN with L=' + str(L) + '_mu=' + str(mu)
    # nodes_position_case = 'left_bottom'
    nodes_position_case = 'in'
    plot_any_LEN(path, figure_name, adjacent_matrix, L, nodes_position_case, text_state=False)

def plot_any_LEN(path, figure_name, adjacent_matrix, L, nodes_position_case='in', text_state=True):


    # set the node lattice arrangement
    # text_state = True
    # nodes_position_case = 'left_bottom'
    # nodes_position_case = 'in'

    L, x, y, ax = plotting_node_arrangement(L, nodes_position_case, text_state)
    # plt.show()
    N = L * L
    for u in np.arange(N, dtype=np.int_):
        # print('u:', u)
        if u % 100 == 0:
            print('u:', u)
        x_1 = x[u]
        y_1 = y[u]
        connected_nodes = np.array(np.where(adjacent_matrix.getrow(u).toarray()[0] == 1), dtype=np.int_)
        # count = 0
        for v in connected_nodes[0]:
            # count = count + 1
            # print('count:', count)
            x_3 = x[v]
            y_3 = y[v]
            distance, position_cases_j_to_each = \
                computing_distance_between_two_nodes_with_periodic_boundary(x_1, y_1, x_3, y_3, L)
            myplot(ax, x_1, y_1, x_3, y_3, L, position_cases_j_to_each, color_name='blue', mylinewidth=0.5)

    plt.savefig(path + figure_name + nodes_position_case + '.png', bbox_inches='tight', pad_inches=0)
    print("Finish plotting the specific lattice-embedded network.")
    plt.close()

def test_my_generator_of_random_number_following_power_law_distribution():
    """
    To test function my_generator_of_random_number_following_power_law_distribution()
    :return:
    """
    m = 3
    K = 100
    mu = 3.5
    degree = my_generator_of_random_number_following_power_law_distribution(m, K, mu)
    print(degree)
    vector_length = 1000
    vector_degree = np.zeros(vector_length)
    for i in range(vector_length):
        print("atcing:", i, "to ", vector_length)
        degree = my_generator_of_random_number_following_power_law_distribution(m, K, mu)
        vector_degree[i] = degree

    # plot hist
    bins_value = len(range(int(vector_degree.min()), int(vector_degree.max())))
    n, bins, patches = plt.hist(x=vector_degree, bins=bins_value, normed='density', color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Degree')
    plt.ylabel('Probability')

    # plt.ylabel('Frequency')
    # maxfreq = n.max()
    # # 设置y轴的上限
    # plt.ylim(top=np.ceil(maxfreq / 5) * 5 if maxfreq % 5 else maxfreq + 5)

    print(bins)

    plt.ion()
    plt.show()
    plt.pause(0.5)
    plt.close()

    min_degree = int(vector_degree.min())
    max_degree = int(vector_degree.max())
    degree_seq = np.array(np.arange(min_degree, max_degree+1), dtype=np.int_)
    print('min_degree:', min_degree, 'max_degree:', max_degree, 'degree_seq:\n', degree_seq)
    i = 0
    pk = np.zeros(len(degree_seq), dtype=np.float_)
    for k in degree_seq:
        num = len(np.where(vector_degree==k)[0])
        print('k:', k, 'num:', num, 'probility:', num/len(vector_degree))
        pk[i] = num/len(vector_degree)
        i = i + 1

    plt.plot(degree_seq, pk)
    plt.show()
    plt.pause(5)

    # save_state = True
    # if save_state:
    #     path = 'E:/my coupled map lattice model simulation/Figures of Generated Networks/'
    #     isExists = os.path.exists(path)
    #     if not isExists:
    #         os.makedirs(path)
    #     plt.savefig(path + 'testing The degree distribution of power law random variable generator' + '.png', bbox_inches='tight',
    #                 pad_inches=0)

if __name__ == '__main__':

    ####################################################################################################################
    time_start = time.time()

    L_value = 100
    generate_many_deltaLEN(L=L_value, k=L_value, dg=4, numbers=1)
    generate_many_deltaLEN(L=L_value, k=L_value, dg=12, numbers=1)
    generate_many_mypoissonLEN(L=L_value, k=L_value, lam=12, K=100, numbers=1)
    generate_many_powerlawLEN(L=L_value, k=L_value, mu=3.6231, m=8, K=100, numbers=1)

    ####################################################################################################################

    time_end = time.time()
    print('totally action time: ', time_end - time_start)