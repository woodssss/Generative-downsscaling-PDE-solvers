import torch
import math
import time
import numpy as np
from scipy.sparse import linalg
from scipy import sparse
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp2d
from GRF import GaussianRF
from scipy.interpolate import interpn
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg
from scipy import optimize
import functools

#################################################################
# This file generate data for
# - d_0 Lap u + d_1 u^3 = d_2 a
# in 2D and 3D
# with zero boundary condition
#################################################################

#########
def get_L(Nx, dim):
    dx = 1 / (1 + Nx)
    Lap = np.zeros((Nx, Nx))
    for i in range(Nx):
        Lap[i][i] = -2

    for i in range(Nx - 1):
        Lap[i + 1][i] = 1
        Lap[i][i + 1] = 1

    L = Lap / dx ** 2
    Ls = sparse.csr_matrix(L)

    if dim==2:
        Lx = sparse.kron(Ls, sparse.eye(Nx))
        Ly = sparse.kron(sparse.eye(Nx), Ls)
        Lap = Lx + Ly
    elif dim==3:
        Lx = sparse.kron(Ls, sparse.kron(sparse.eye(Nx), sparse.eye(Nx)))
        Ly = sparse.kron(sparse.eye(Nx), sparse.kron(Ls, sparse.eye(Nx)))
        Lz = sparse.kron(sparse.eye(Nx), sparse.kron(sparse.eye(Nx), Ls))
        Lap = Lx + Ly + Lz
    return Lap

#######################

def get_AC_mat(Nx, dt, d, k):
    dx = 1 / Nx
    Lap = np.zeros((Nx, Nx))
    for i in range(Nx):
        Lap[i][i] = -2

    for i in range(Nx - 1):
        Lap[i + 1][i] = 1
        Lap[i][i + 1] = 1

    Lap[0][-1] = 1
    Lap[-1][0] = 1

    L = Lap / dx ** 2
    Ls = sparse.csr_matrix(L)

    Lx = sparse.kron(Ls, np.eye(Nx))
    Ly = sparse.kron(sparse.eye(Nx), Ls)

    Lap = Lx + Ly

    A = (1 - 4 * dt * k) * sparse.eye(Nx ** 2) - dt * d * Lap

    return Lap, A

#####################################

def myfunc(L, d0, d1, x):
    return d0 * L.dot(x) + np.power(x, 3) * d1

def myjac(L, d0, d1, x):
    return d0 * L + 3 * sparse.diags(np.square(x)) * d1

### LM method ####################################################
def levenberg_marquardt_one(f, jac, x0, a, d2, max_iter=1000, tol=1e-5, lmbda0=1e-4):
    # for ac, a is u0, d2 is 1

    x = x0.copy()
    lmbda = lmbda0

    for iter in range(max_iter):
        # Evaluate the function and Jacobian at the current point
        F = f(x)
        J = jac(x)

        # Compute the residual vector
        r = a * d2 - F

        JTJ = (J.T).dot(J)
        JTJ_diag = (JTJ).diagonal()
        H = JTJ + lmbda * sparse.diags(JTJ_diag)
        g = (J.T).dot(r)

        #delta = sparse.linalg.spsolve(H, g)
        delta, _ = cg(H, g)

        # Check if the update is small enough
        # if np.linalg.norm(delta) < tol:
        #     print('converged at delta', iter, x.shape)
        #     break

        # Update the parameter estimates
        x_new = x + delta

        # Compute the new residuals
        F_new = f(x_new)
        r_new = a * d2 - F_new
        # r_new = -np.matmul(L, a) + a * c2 - F_new
        # Compute the error and check for convergence
        error = np.linalg.norm(r_new)

        if error < tol:
            print('converged at error', iter)
            break

        # Adjust the damping parameter lmbda
        if np.linalg.norm(r_new) < np.linalg.norm(r):
            lmbda /= 10
            # lmbda = lmbda0
            x = x_new.copy()
        else:
            lmbda *= 10

    print('final error', error, iter)

    if iter>max_iter-2:
        return None

    return x
################################################################

#### krylov ######################
def kfunc(L, d0, d1, d2, a, x):
    return d0 * L.dot(x) + np.power(x, 3) * d1 - d2 * a

def KL(L, d0, d1, d2, a, x0):
    func = functools.partial(kfunc, L, d0, d1, d2, a)
    x = optimize.newton_krylov(func, x0)
    return x
#####################################

def GNA_one(f, jac, d2, x0, a):
    x = x0.copy()

    # Evaluate the function and Jacobian at the current point
    F = f(x)
    J = jac(x)

    # Compute the residual vector
    r = a * d2 - F

    JTJ = (J.T).dot(J)
    H = JTJ
    g = (J.T).dot(r)
    #delta = sparse.linalg.spsolve(H, g)
    delta, _ = cg(H, g)

    # Update the parameter estimates
    x_new = x + delta

    return x_new

def get_GNA(f, jac, d2, x, a):
    # convert to numpy array
    if torch.is_tensor(x) == True:
        x, a = x.cpu().detach().numpy(), a.cpu().detach().numpy()

    bs = x.shape[0]
    x_new = np.zeros_like(x)
    Nx_f = x.shape[1]

    for i in range(bs):
        x_tmp, a_tmp = x[i, ...].flatten(), a[i, ...].flatten()

        x_new_tmp = GNA_one(f, jac, d2, x_tmp, a_tmp)

        x_new[i, ..., 0] = x_new_tmp.reshape(Nx_f, Nx_f)

    # back to tensor
    #x_new = torch.from_numpy(x_new).float()
    return x_new

### upsample ###
def up_sample_1d(Nx_c, Nx_f, x_c, x_f, f_c):
    x_c = np.concatenate([np.zeros((1, 1)), x_c, np.ones((1, 1))], axis=0)
    x_f = np.concatenate([np.zeros((1, 1)), x_f, np.ones((1, 1))], axis=0)
    f_c = np.concatenate([np.zeros((1)), f_c, np.zeros((1))], axis=0)
    func = CubicSpline(x_c[:, 0], f_c)
    f_f = func(x_f[1:-1, 0])
    f_f = f_f[:Nx_f]

    return f_f


def up_sample_2d(Nx_c, Nx_f, x_c, x_f, f_c):

    f_f_1 = np.zeros((Nx_c, Nx_f))
    f_f_2 = np.zeros((Nx_f, Nx_f))

    for i in range(Nx_c):
        f_f_1[i, :] = up_sample_1d(Nx_c, Nx_f, x_c, x_f, f_c[i, :])

    for j in range(Nx_f):
        f_f_2[:, j] = up_sample_1d(Nx_c, Nx_f, x_c, x_f, f_f_1[:, j])

    f_f = f_f_2[:Nx_f, :Nx_f]

    return f_f


def up_sample_3d(Nx_c, Nx_f, x_c, x_f, f_c):

    f_f_1 = np.zeros((Nx_f, Nx_c, Nx_c))
    f_f_2 = np.zeros((Nx_f, Nx_f, Nx_f))

    for i in range(Nx_c):
        for j in range(Nx_c):
            f_f_1[:, i, j] = up_sample_1d(Nx_c, Nx_f, x_c, x_f, f_c[:, i, j])

    for k in range(Nx_f):
        f_f_2[k, :, :] = up_sample_2d(Nx_c, Nx_f, x_c, x_f, f_f_1[k, :, :])

    return f_f_2

### generate samples #########

def generate_sample_3d(Ns, alpha, tau, Nx_c, m2, seed):
    lx = 1
    Nx = m2 * (Nx_c + 1) - 1
    Nxp = m2 * (Nx_c + 1)
    dx = lx / (Nx + 1)
    dxp = lx / Nxp

    points_x = np.linspace(dx, lx - dx, Nx).T
    x = points_x[:, None]
    points_xp = np.linspace(0, lx - dxp, Nxp).T
    xp = points_xp[:, None]

    X, Y, Z = np.meshgrid(x, x, x)

    xx, yy = np.meshgrid(x, x)
    xxp, yyp = np.meshgrid(xp, xp)

    pts = (points_xp, points_xp, points_xp)
    # pts1 = np.array(np.meshgrid(x, x, x)).T.reshape(-1, 3)
    pts1 = np.transpose(np.array(np.meshgrid(x, x, x)), (3, 2, 1, 0)).reshape(-1, 3)

    GRF = GaussianRF(3, Nxp, alpha, tau, seed=seed)
    g_mat_p = GRF.sample(Ns)
    g_mat_r = np.zeros((Ns, Nx, Nx, Nx))

    for i in range(Ns):
        ### interpolation ####
        tmp_p = g_mat_p[i, ...].numpy()

        tmp_f = interpn(pts, tmp_p, pts1)
        tmp_f = tmp_f.reshape(Nx, Nx, Nx) * X * (1 - X) * Y * (1 - Y) * Z * (1 - Z)
        tmp_f = tmp_f / np.max(np.abs(tmp_f))  # normalize

        g_mat_r[i, ...] = tmp_f.reshape(Nx, Nx, Nx)

    # plt.figure(1)
    # cp = plt.contourf(xx, yy, g_mat_r[0, ..., int(Nx/2)])
    # plt.colorbar(cp)
    #
    # plt.figure(2)
    # cp = plt.contourf(xxp, yyp, g_mat_p[0, ..., int(Nx/2)])
    # plt.colorbar(cp)
    #
    # plt.show()


    x_c = x[m2 - 1::m2, :]
    x_f = x[1::2, :]
    g_mat_c = g_mat_r[:, m2 - 1::m2, m2 - 1::m2, m2 - 1::m2]
    g_mat_f = g_mat_r[:, 1::2, 1::2, 1::2]

    return x_c, g_mat_c, x_f, g_mat_f, x, g_mat_r

def generate_sample_2d(Ns, alpha, tau, Nx_c, m2, seed):
    dim=2
    lx = 1
    Nx = m2 * (Nx_c + 1) - 1
    Nxp = m2 * (Nx_c)
    dx = lx / (Nx + 1)
    dxp = lx / Nxp

    points_x = np.linspace(dx, lx - dx, Nx).T
    x = points_x[:, None]
    points_xp = np.linspace(0, lx - dxp, Nxp).T
    xp = points_xp[:, None]

    xx, yy = np.meshgrid(x, x)

    GRF = GaussianRF(dim, Nxp, alpha, tau, seed=seed)
    g_mat_p = GRF.sample(Ns).numpy()
    g_mat_r = np.zeros((Ns, Nx, Nx))

    for i in range(Ns):
        tmp_p = g_mat_p[i, ...]
        # print(points_xp.shape, tmp_p.shape)
        # zxc
        tmp_func = interp2d(points_xp, points_xp, tmp_p, kind='cubic')
        tmp_f = tmp_func(points_x, points_x) * xx * (1 - xx) * yy * (1 - yy)  # zero bc
        tmp_f = tmp_f / np.max(np.abs(tmp_f))  # normalize

        g_mat_r[i, ...] = tmp_f.reshape(Nx, Nx)

    x_c = x[m2 - 1::m2, :]
    x_f = x[1::2, :]
    g_mat_c = g_mat_r[:, m2 - 1::m2, m2 - 1::m2]
    g_mat_f = g_mat_r[:, 1::2, 1::2]

    return x_c, g_mat_c, x_f, g_mat_f, x, g_mat_r