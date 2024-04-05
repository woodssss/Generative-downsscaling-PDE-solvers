import torch
import math
import numpy as np
import functools
from data_utils import *
import time

def gen_batch(Ns, N_extra, Nx_c, m, alpha, tau, d0, d1, d2, bid):
    m2 = m*2
    Nx_f = m * (Nx_c + 1) - 1
    NN = Ns + N_extra
    x_c, a_mat_c, x_f, a_mat_f, x_r, a_mat_r = generate_sample_2d(NN, alpha, tau, Nx_c, m2, seed=bid)

    L_f = get_L(Nx_f, dim=2)
    L_c = get_L(Nx_c, dim=2)

    ############### prepare training ###########################################
    Set_f = np.zeros((NN, Nx_f, Nx_f, 2))
    Set_c = np.zeros((NN, Nx_c, Nx_c, 2))

    myfunc_c, myj_c = functools.partial(myfunc, L_c, d0, d1), functools.partial(myjac, L_c, d0, d1)
    myfunc_f, myj_f = functools.partial(myfunc, L_f, d0, d1), functools.partial(myjac, L_f, d0, d1)

    ct = 0
    time_c, time_f = 0, 0

    for i in range(NN):
        ### coarser solver
        tmp_a_c = a_mat_c[i, :, :]
        tic = time.time()
        tmp_u_c = levenberg_marquardt_one(myfunc_c, myj_c, np.zeros((Nx_c ** 2)),
                                          a_mat_c[i, :, :].flatten(), d2, max_iter=200, tol=1e-5, lmbda0=1e-3)
        if tmp_u_c is None:
            continue
        toc = time.time() - tic
        time_c +=toc

        # finer solver
        tic = time.time()
        tmp_a_f = a_mat_f[i, :, :]
        tmp_u_f = levenberg_marquardt_one(myfunc_f, myj_f, np.zeros((Nx_f ** 2)), a_mat_f[i, :, :].flatten(), d2,
                                          max_iter=200, tol=1e-5, lmbda0=1e-3)
        if tmp_u_f is None:
            continue
        toc =time.time()-tic

        # tic = time.time()
        # tmp_a_f = a_mat_f[i, :, :]
        # tmp_u_f2 = KL(L_f, d0, d1, d2, a_mat_f[i, :, :].flatten(), np.zeros((Nx_f ** 2)))
        # toc2 = time.time() - tic
        #
        # print('time for 2 methods:', toc, toc2)

        time_f +=toc

        tmp_u_f = np.array(tmp_u_f)

        tmp_u_c_up = up_sample_2d(Nx_c, Nx_f, x_c, x_f, tmp_u_c.reshape(Nx_c, Nx_c)).flatten()

        if np.linalg.norm(tmp_u_c_up - tmp_u_f) / np.linalg.norm(tmp_u_f) > 1.0 or np.max(np.abs(tmp_u_f))>1.01:
            continue

        Set_c[ct, :, :, 0] = tmp_a_c
        Set_c[ct, :, :, 1] = tmp_u_c.reshape(Nx_c, Nx_c)

        Set_f[ct, :, :, 0] = tmp_a_f
        Set_f[ct, :, :, 1] = tmp_u_f.reshape(Nx_f, Nx_f)

        ct += 1

        if ct >= Ns:
            break

    if ct < Ns:
        print('valid is', ct, ' increase Ne for more valid samples')
        exit()
    else:
        print('valid Ne')

    Set_f_train = Set_f[:Ns, ...]
    Set_c_train = Set_c[:Ns, ...]

    Set_c_train_up = np.zeros_like(Set_f_train)
    for i in range(Set_f_train.shape[0]):
        Set_c_train_up[i, :, :, 0] = up_sample_2d(Nx_c, Nx_f, x_c, x_f, Set_c_train[i, :, :, 0])
        Set_c_train_up[i, :, :, 1] = up_sample_2d(Nx_c, Nx_f, x_c, x_f, Set_c_train[i, :, :, 1])

    return x_c, x_f, Set_c_train_up, Set_f_train, time_c, time_f


def gen_batch_all(Ns, N_extra, Nx_c, m, alpha, tau, d0, d1, d2, bid):
    m2 = m*2
    Nx_f = m * (Nx_c + 1) - 1
    Nx_r = m2 * (Nx_c + 1) - 1

    NN = Ns + N_extra

    x_c, a_mat_c, x_f, a_mat_f, x_r, a_mat_r = generate_sample_2d(NN, alpha, tau, Nx_c, m2, seed=bid+20)

    L_f = get_L(Nx_f, dim=2)
    L_c = get_L(Nx_c, dim=2)
    L_r = get_L(Nx_r, dim=2)

    ############### prepare training ###########################################
    Set_f = np.zeros((NN, Nx_f, Nx_f, 2))
    Set_c = np.zeros((NN, Nx_c, Nx_c, 2))
    Set_r = np.zeros((NN, Nx_r, Nx_r, 2))

    myfunc_c, myj_c = functools.partial(myfunc, L_c, d0, d1), functools.partial(myjac, L_c, d0, d1)
    myfunc_f, myj_f = functools.partial(myfunc, L_f, d0, d1), functools.partial(myjac, L_f, d0, d1)
    myfunc_r, myj_r = functools.partial(myfunc, L_r, d0, d1), functools.partial(myjac, L_r, d0, d1)

    ct = 0
    time_c, time_f, time_r = 0, 0, 0

    for i in range(NN):
        ### coarser solver
        tmp_a_c = a_mat_c[i, :, :]
        tic = time.time()
        tmp_u_c = levenberg_marquardt_one(myfunc_c, myj_c, np.zeros((Nx_c ** 2)),
                                          a_mat_c[i, :, :].flatten(), d2, max_iter=200, tol=1e-5, lmbda0=1e-4)
        if tmp_u_c is None:
            continue
        toc = time.time() - tic
        time_c +=toc

        # finer solver
        tic = time.time()
        tmp_a_f = a_mat_f[i, :, :]
        tmp_u_f = levenberg_marquardt_one(myfunc_f, myj_f, np.zeros((Nx_f ** 2)), a_mat_f[i, :, :].flatten(), d2,
                                          max_iter=200, tol=1e-5, lmbda0=1e-4)
        if tmp_u_f is None:
            continue
        toc =time.time()-tic
        time_f +=toc

        tmp_u_f = np.array(tmp_u_f)

        tmp_u_c_up = up_sample_2d(Nx_c, Nx_f, x_c, x_f, tmp_u_c.reshape(Nx_c, Nx_c)).flatten()

        # ref solver
        tic = time.time()
        tmp_a_r = a_mat_r[i, :, :]

        tmp_u_r = levenberg_marquardt_one(myfunc_r, myj_r, np.zeros((Nx_r ** 2)),
                                          a_mat_r[i, :, :].flatten(), d2, max_iter=200, tol=1e-5, lmbda0=1e-4)
        if tmp_u_r is None:
            continue

        toc = time.time()-tic
        time_r += toc

        if np.linalg.norm(tmp_u_c_up - tmp_u_f) / np.linalg.norm(tmp_u_f) > 1.0 or np.max(np.abs(tmp_u_f))>1.01:
            continue

        Set_c[ct, :, :, 0] = tmp_a_c
        Set_c[ct, :, :, 1] = tmp_u_c.reshape(Nx_c, Nx_c)

        Set_f[ct, :, :, 0] = tmp_a_f
        Set_f[ct, :, :, 1] = tmp_u_f.reshape(Nx_f, Nx_f)

        Set_r[ct, :, :, 0] = tmp_a_r
        Set_r[ct, :, :, 1] = tmp_u_r.reshape(Nx_r, Nx_r)

        ct += 1

        if ct >= Ns:
            break

    if ct < Ns:
        print('valid is', ct, ' increase Ne for more valid samples')
        exit()
    else:
        print('valid Ne')

    Set_f_train = Set_f[:Ns, ...]
    Set_c_train = Set_c[:Ns, ...]
    Set_r_train = Set_r[:Ns, ...]

    Set_c_train_up = np.zeros_like(Set_f_train)
    for i in range(Set_f_train.shape[0]):
        Set_c_train_up[i, :, :, 0] = up_sample_2d(Nx_c, Nx_f, x_c, x_f, Set_c_train[i, :, :, 0])
        Set_c_train_up[i, :, :, 1] = up_sample_2d(Nx_c, Nx_f, x_c, x_f, Set_c_train[i, :, :, 1])

    return x_c, x_f, x_r, Set_c_train_up, Set_f_train, Set_r_train, time_c, time_f, time_r