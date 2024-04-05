import torch
import math
from matplotlib import cm
import matplotlib.pyplot as plt
import functools
from data_utils import *
from utils import *
from Poisson_2d_gen_batch import gen_batch, gen_batch_all
import os
import argparse
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


parser = argparse.ArgumentParser(description='Data preparation for Poisson 2D')

parser.add_argument('-ns', '--ns', type=int, metavar='', help='number of samples')
parser.add_argument('-nex', '--nex', type=int, metavar='', help='number of extra')
parser.add_argument('-nx', '--nx', type=int, metavar='', help='num of grids')
parser.add_argument('-m', '--m', type=int, metavar='', help='num of upscale')

parser.add_argument('-seed', '--seed', type=int, metavar='', help='random seed')
parser.add_argument('-d0', '--d0', type=float, metavar='', help='d0')
parser.add_argument('-d1', '--d1', type=float, metavar='', help='d1')
parser.add_argument('-d2', '--d2', type=float, metavar='', help='d2')

parser.add_argument('-alp', '--alp', type=float, metavar='', help='alpha')
parser.add_argument('-tau', '--tau', type=float, metavar='', help='tau')

parser.add_argument('-flg', '--flg', type=float, metavar='', help='flag=1 means training and flag=1 means testing')
parser.add_argument('-path', '--path', type=str, required= False, metavar='', help='path to save')

args = parser.parse_args()

if __name__ == "__main__":

    ### define parameters from args ############
    Ns, Nx_c, m = args.ns, args.nx, args.m
    d0, d1, d2 = args.d0, args.d1, args.d2

    N_extra = args.nex

    m2 = int(m * 2)
    Nx_f = m * (Nx_c + 1) - 1

    alpha, tau = args.alp, args.tau

    seed = args.seed
    path = args.path
    flag = args.flg

    # alpha = 2.1
    # tau = 10
    # Nx_c = 16
    # m = 8
    # Nx_f = m * (Nx_c + 1) - 1
    # m2 = m*2
    # Ns = 20
    # N_extra = 20
    # d0, d1, d2 = -0.0001, 1, 1
    #
    # ### change here
    # flag = 2
    # bid = 20
    # ###########

    L_f = get_L(Nx_f, dim=2)
    myfunc_f, myjac_f = functools.partial(myfunc, L_f, d0, d1), functools.partial(myjac, L_f, d0, d1)

    if flag==1:
        x_c, x_f, Set_c_train_up, Set_f_train,  time_c, time_f = gen_batch(Ns, N_extra, Nx_c, m, alpha, tau, d0, d1, d2, seed)
        avg_time_c, avg_time_f = time_c/Set_c_train_up.shape[0], time_f/Set_c_train_up.shape[0]
    else:
        x_c, x_f, x_r, Set_c_train_up, Set_f_train, Set_r_train, time_c, time_f, time_r = gen_batch_all(Ns, N_extra, Nx_c, m, alpha, tau, d0, d1, d2, seed)
        avg_time_c, avg_time_f, avg_time_r = time_c / Set_c_train_up.shape[0], time_f / Set_c_train_up.shape[0], time_r / Set_c_train_up.shape[0]

    common_name = '_Ns_' + str(Ns) + '_Nx_' + str(Nx_c) + '_m_' + num2str_deciaml(
        m) + '_d0_' + num2str_deciaml(d0) + '_d1_' + num2str_deciaml(d1) + '_d2_' + num2str_deciaml(
        d2) + '_alp_' + num2str_deciaml(alpha) + '_tau_' + num2str_deciaml(tau) + '.npy'

    cwd = os.getcwd()

    if flag==1:
        npy_name = cwd + '/data/' + 'Pos_2D_train' + common_name

        with open(npy_name, 'wb') as ss:
            np.save(ss, Set_c_train_up)
            np.save(ss, Set_f_train)
            np.save(ss, avg_time_c)
            np.save(ss, avg_time_f)
    else:
        npy_name = cwd + '/data/' + 'Pos_2D_test' + common_name

        with open(npy_name, 'wb') as ss:
            np.save(ss, Set_c_train_up)
            np.save(ss, Set_f_train)
            np.save(ss, Set_r_train)
            np.save(ss, avg_time_c)
            np.save(ss, avg_time_f)
            np.save(ss, avg_time_r)
