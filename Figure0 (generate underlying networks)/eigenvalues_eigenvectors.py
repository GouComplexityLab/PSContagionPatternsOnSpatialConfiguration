# -*- coding: utf-8 -*-
"""
Created on Sunday, 16 June 2019

generate 2D lattice embedded networks

"""
import os
import time
import datetime

import numpy as np
import scipy.io as scio
import math
from random import choice

import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist

import scipy.sparse as sp
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import eigsh


###Delta###############################################################

if __name__ == '__main__':

    time_start = time.time()
    ''' merge many'''
    print(range(50,210,10))

    for L in np.array([100]):
        print('--------------------------> L=' + str(L))
        k = 1


        data_npzmat_path = './Generated delta Lattice Embedded Networks L=' + str(L) + ' dg=12' + '/'
        l_name_npz = 'laplacian_matrix_deltaLEN_L=' + str(L) + '_dg=12_number=1.npz'

        print(data_npzmat_path + l_name_npz)
        laplacian_matrix = sp.load_npz(data_npzmat_path + l_name_npz)
        print(laplacian_matrix.shape)

        # 想要计算的特征值的数量
        num_of_eigenvalues = 3


        first_smallest_eigenvalues, first_smallest_eigenvectors = eigsh(laplacian_matrix, k=num_of_eigenvalues, which='SM')
        print("Smallest", num_of_eigenvalues, "eigenvalues of the first\n:", np.real(first_smallest_eigenvalues))

        eigenvalues_name_mat = 'Delta8eigenvalues_with_L=' + str(L) + '.mat'
        scio.savemat(data_npzmat_path + eigenvalues_name_mat,
                     {'first_smallest_eigenvalues': first_smallest_eigenvalues})

    time_end = time.time()
    print('totally action time: ', time_end - time_start)

###poisson####################################################################

if __name__ == '__main__':

    time_start = time.time()
    ''' merge many'''

    for L in np.array([100]):
        print('poisson --------------------------> L=' + str(L))
        k = 1

        data_npzmat_path = './Generated mypoisson Lattice Embedded Networks L=' + str(L) + ' lam=12' + '/'
        l_name_npz = 'laplacian_matrix_mypoissonLEN L=' + str(L) + ' lam=12_number=1.npz'

        print(data_npzmat_path + l_name_npz)
        laplacian_matrix = sp.load_npz(data_npzmat_path + l_name_npz)
        print(laplacian_matrix.shape)

        num_of_eigenvalues = 3

        first_smallest_eigenvalues, first_smallest_eigenvectors = eigsh(laplacian_matrix, k=num_of_eigenvalues, which='SM')
        print("Smallest", num_of_eigenvalues, "eigenvalues of the first\n:", np.real(first_smallest_eigenvalues))

        eigenvalues_name_mat = 'poisson8eigenvalues_with_L=' + str(L) + '.mat'
        scio.savemat(data_npzmat_path + eigenvalues_name_mat,
                     {'first_smallest_eigenvalues': first_smallest_eigenvalues})

    time_end = time.time()
    print('totally action time: ', time_end - time_start)

###Powerlaw#################################################################

if __name__ == '__main__':

    time_start = time.time()
    ''' merge many'''
    print(range(50, 210, 10))

    for L in np.array([100]):
        print('Powerlaw --------------------------> L=' + str(L))
        k = 1

        data_npzmat_path = './Generated powerlaw Lattice Embedded Networks L=' + str(L) + ' mu=3.6231' + '/'
        l_name_npz = 'laplacian_matrix_powerlawLEN L=' + str(L) + ' mu=3.6231_number=1.npz'

        print(data_npzmat_path + l_name_npz)
        laplacian_matrix = sp.load_npz(data_npzmat_path + l_name_npz)
        print(laplacian_matrix.shape)

        num_of_eigenvalues = 3

        first_smallest_eigenvalues, first_smallest_eigenvectors = eigsh(laplacian_matrix, k=num_of_eigenvalues, which='SM')
        print("Smallest", num_of_eigenvalues, "eigenvalues of the first\n:", np.real(first_smallest_eigenvalues))

        eigenvalues_name_mat = 'powerlaw8eigenvalues_with_L=' + str(L) + '.mat'
        scio.savemat(data_npzmat_path + eigenvalues_name_mat,
                     {'first_smallest_eigenvalues': first_smallest_eigenvalues})

    time_end = time.time()
    print('totally action time: ', time_end - time_start)
