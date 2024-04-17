import numpy as np
import torch
import torch.nn as nn
from data_utils import get_GNA
import torch.nn.functional as F
import time
from matplotlib import cm
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import griddata
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def mylogger(filename, content):
    with open(filename, 'a') as fw:
        print(content, file=fw)

def nump2tensor(x):
    return torch.from_numpy(x).float()

def tensor2nump(x):
    return x.cpu().detach().numpy()

def myL2L(a, b):
    return torch.mean(torch.square(a - b))

def myRL2_np(a, b):
    error = np.mean( np.sqrt( np.sum((a-b)**2, axis=(1, 2, 3)) / np.sum((a)**2, axis=(1, 2, 3)) ) )
    return error

def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

def sigmoid_beta_schedule(beta_start, beta_end, timesteps):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def linear_beta_schedule(beta_start, beta_end, timesteps):
    return torch.linspace(beta_start, beta_end, timesteps)

def get_parameters(beta_start, beta_end, Nt):
    betas = torch.linspace(beta_start, beta_end, steps=Nt)
    alphas = 1 - betas
    alphas_bar = torch.cumprod(alphas, 0)
    alphas_bar_sqrt = torch.sqrt(alphas_bar)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_bar)
    sigma = torch.sqrt(betas)
    return betas, alphas, alphas_bar, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, sigma

# def get_parameters2(Nt):
#     betas = torch.linspace(start=1e-4, end=0.02, steps=Nt)
#     alphas = 1. - betas
#     alphas_cumprod = torch.cumprod(alphas, axis=0)
#     alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
#     sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
#     sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
#     sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
#     # posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
#     posterior_variance = betas
#     return betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance


def rand_noise(x):
    return torch.randn_like(x)

def q_x_t_cond_x_0(alphas_bar_sqrt, one_minus_alphas_bar_sqrt, x_0, t):
    # compute q(x_t|x_0) = N (mean = sqrt(alpha bar))
    noise = rand_noise(x_0)
    x_0_coeff = extract(alphas_bar_sqrt, t, x_0)
    noise_ceoff = extract(one_minus_alphas_bar_sqrt, t, x_0)
    return x_0_coeff.to(device) * x_0.to(device) + noise_ceoff.to(device) * noise.to(device), noise.to(device)

def get_loss(model, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, Nt, a, x_0, x_c):
    # we sample x_t at any specific t
    bs = x_0.shape[0]
    t = torch.randint(0, Nt, size=(bs // 2,))
    t = torch.cat([t, Nt - t - 1], dim=0)
    #t = torch.randint(0, Nt, (bs,), device=device).long()

    # use q_x_t_cond_x_0 tp generate x_t from x_0
    x_noise, noise = q_x_t_cond_x_0(alphas_bar_sqrt, one_minus_alphas_bar_sqrt, x_0, t)

    noise_approx = model(x_noise.to(device), x_c.to(device), a.to(device), t.to(device))

    return myL2L(noise, noise_approx)

####### smoothify ###############
# def smoothy(f, delta=10):
#     N = f.shape[0]
#     for i in range(1,N-1):
#         for j in range(1, N - 1):
#             dx = torch.abs(f[i, j] - f[i-1, j])*N
#             dy = torch.abs(f[i, j] - f[i, j-1]) * N
#             if torch.max(dx, dy)>delta:
#                 f[i, j] = 0.25*torch.sum(f[i-1:i+1, j-1:j+1])
#
#     return f

def max_grad(f):
    #f = f.detach().cpu()
    N = f.shape[0]
    max_d = 1e-4 * torch.ones(1,).to(device)
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            dx = torch.abs(f[i, j] - f[i - 1, j]) * N
            dy = torch.abs(f[i, j] - f[i, j - 1]) * N

            max_d = torch.max(max_d, dx)
            max_d = torch.max(max_d, dy)

            # max_d = np.maximum(max_d, dx)
            # max_d = np.maximum(max_d, dy)

    return max_d

def replace_peaks(f, delta = 10):
    f = f.cpu().numpy()
    N = f.shape[0]
    dx = 1/ (N + 1)
    x = np.linspace(dx, 1 - dx, N)
    xx, yy = np.meshgrid(x, x)
    x_rep_ls = []
    y_rep_ls = []
    for i in range(2, N-2):
        for j in range(2, N - 2):
            dx = np.abs(f[i+1, j] - f[i-1, j])*N/2
            dy = np.abs(f[i, j+1] - f[i, j-1])*N/2

            if np.maximum(dx, dy) > delta:
                x_rep_ls.append(x[i])
                y_rep_ls.append(x[j])

    if len(x_rep_ls)>0:

        x_rep = np.stack(x_rep_ls)
        y_rep = np.stack(y_rep_ls)

        xx_flat, yy_flat, f_flat = xx.ravel(), yy.ravel(), f.ravel()

        mask = ~((xx_flat[:, None] == x_rep.ravel()[None, :]) & (yy_flat[:, None] == y_rep.ravel()[None, :])).any(axis=1)

        f_new = griddata((xx_flat[mask], yy_flat[mask]), f_flat[mask], (xx_flat[~mask], yy_flat[~mask]), method='cubic')

        f_flat[~mask] = f_new

        f = f_flat.reshape(N, N)

        return f

    else:
        return f


def smoothy_avg(f, delta=10):
    N = f.shape[0]
    delta = delta * torch.ones(1,)
    x = torch.linspace(0, 1, N)
    xx,yy = torch.meshgrid(x,x)
    for i in range(2, N-2):
        for j in range(2, N - 2):
            dx = torch.abs(f[i, j] - f[i-1, j])*N/2
            dy = torch.abs(f[i, j] - f[i, j-1])*N/2
            if torch.max(dx, dy)>delta:
                f[i, j] = torch.sum(f[i-2:i+1, j-2:j+1])/9

    return f

def smoothy(f, delta=10):
    f_new = replace_peaks(f, delta=delta)
    return f_new


######### vanilla DDPM ###########
def sample_backward_ddpm_step(model, sigma, alphas, one_minus_alphas_bar_sqrt, a, x_t, x_c, t):
    t = torch.tensor([t])
    sigma_t = extract(sigma, t, x_t).to(device)
    xt_coeff = (1 / extract(alphas, t, x_t)).sqrt().to(device)
    eps_coeff = -(1 / extract(alphas, t, x_t).sqrt() * (
            1 - extract(alphas, t, x_t)) / extract(one_minus_alphas_bar_sqrt, t,
                                                   x_t)).to(device)
    with torch.no_grad():
        eps_theta = model(x_t.to(device), x_c.to(device), a.to(device), t.to(device))
    z = rand_noise(x_t).to(device)

    mean = (xt_coeff * x_t + eps_coeff * eps_theta).to(device)

    if t == 0:
        return mean
    else:
        torch.cuda.empty_cache()
        return mean + sigma_t * z

def sample_backward_loop_ddpm(model, clip, sigma, alphas, one_minus_alphas_bar_sqrt, Nt, a, x_c):
    # x_T from guidance info
    x_T = torch.randn_like(x_c)
    x_seq = [x_T.detach().cpu()]
    x_tmp_cpu = x_T

    for i in range(Nt)[::-1]: # check here

        x_tmp = sample_backward_ddpm_step(model, sigma, alphas, one_minus_alphas_bar_sqrt, a.to(device), x_tmp_cpu.to(device), x_c.to(device), i)

        x_tmp_cpu = x_tmp.detach().cpu()

        x_tmp_cpu = torch.clamp(x_tmp_cpu, -clip, clip)

        del x_tmp

        torch.cuda.empty_cache()

        x_seq.append(x_tmp_cpu)

    #x_seq = np.concatenate(x_seq, axis=0)
    x_seq = torch.cat(x_seq, dim=0)

    return x_seq

def solver_ddpm(model, clip, sigma, alphas, one_minus_alphas_bar_sqrt, Nt, a_mat, u_c_mat):
    bs = a_mat.shape[0]
    N = a_mat.shape[1]
    DM_mat = np.zeros((bs, N, N, 1))
    P_ls = []

    for i in range(bs):
        a = a_mat[[i], ...]
        x_c = u_c_mat[[i], ...]

        x_generate_process = sample_backward_loop_ddpm(model, clip, sigma, alphas, one_minus_alphas_bar_sqrt, Nt, a, x_c)
        DM_mat[i, ...] = x_generate_process[-1, ...]
        P_ls.append(x_generate_process)

    return DM_mat, P_ls

def get_plot_sample_ddpm(Nt, xx, yy, Process, pd, test_c, test_f, fig_name, step):
    idx = 0

    md_c = max_grad(test_c[idx, ..., 1])
    md_f = max_grad(test_f[idx, ..., 1])
    md_p = max_grad(Process[idx][-1, ..., 0])

    print('max grad', md_c, md_f, md_p)

    fig, ax = plt.subplots(1, 11, figsize=(20, 3))

    ax[0].contourf(xx, yy, Process[idx][-1, ..., 0], 36, cmap=cm.jet)
    ax[0].set_title(r'final')

    for i in range(10):
        ax[i+1].contourf(xx, yy, Process[idx][int(Nt / 10) * i, ..., 0], 36, cmap=cm.jet)
        ax[i+1].set_title(r'$sample$' + f"-{int(Nt / 10) * i}")

    for axs in ax.flat:
        axs.set_xticks([])
        axs.set_yticks([])

    plt.tight_layout()
    fig.savefig(fig_name + '_epoch_step_' + str(step) + '_generate.jpg')

    fig2, ax = plt.subplots(2, 2, figsize=(8, 8))
    ax[0, 0].contourf(xx, yy, test_c[idx, ..., 1].detach().cpu().numpy(), 36, cmap=cm.jet)
    ax[0, 0].set_title('uc')

    cp = ax[0, 1].contourf(xx, yy, test_f[idx, ..., 1].detach().cpu().numpy(), 36, cmap=cm.jet)
    ax[0, 1].set_title('uf')
    plt.colorbar(cp)

    cp = ax[1, 0].contourf(xx, yy, pd[idx, ..., 0], 36, cmap=cm.jet)
    ax[1, 0].set_title('u pd')
    plt.colorbar(cp)

    cp = ax[1, 1].contourf(xx, yy, np.abs(pd[idx, ..., 0] - test_f[idx, ..., 1].detach().cpu().numpy()), 36, cmap=cm.jet)
    ax[1, 1].set_title('error')
    plt.colorbar(cp)
    fig2.savefig(fig_name + '_epoch_step_' + str(step) + '_pd.jpg')

    plt.show()

####################################################################################

######## ddim #########################################################
def sample_backward_ddim_step(model, alphas_bar, a, x_t, x_c, t, t_next, *args):
    # t_next is the t-Delta t
    t = torch.tensor([t])
    tn = torch.tensor([t_next])

    at = extract(alphas_bar, t, x_t)
    atn = extract(alphas_bar, tn, x_t)

    xt_coeff = (atn / at).sqrt()
    coeff1 = -((atn / at) * (1 - at)).sqrt()
    coeff2 = (1 - atn).sqrt()
    eps_coeff = coeff1 + coeff2

    with torch.no_grad():
        eps_theta = model(x_t.to(device), x_c.to(device), a.to(device), t.to(device))

    x_tn = (xt_coeff.to(device) * x_t + eps_coeff.to(device) * eps_theta).to(device)

    torch.cuda.empty_cache()
    return x_tn


def sample_backward_loop_ddim(model, clip, alphas_bar, a, x_c, my_t_list, *args):
    # x_T from guidance info
    x_T = torch.randn_like(x_c)
    x_seq = [x_T.detach().cpu()]
    l = len(my_t_list)
    x_tmp_cpu = x_T

    for i in range(1, l): # check here
        t = my_t_list[l-i]
        t_next = my_t_list[l-i-1]

        x_tmp = sample_backward_ddim_step(model, alphas_bar, a.to(device), x_tmp_cpu.to(device), x_c.to(device), t, t_next, *args)

        x_tmp_cpu = x_tmp.detach().cpu()

        #x_tmp_cpu = torch.clamp(x_tmp_cpu, -1.5, 1.5)
        x_tmp_cpu = torch.clamp(x_tmp_cpu, -clip, clip)

        del x_tmp

        torch.cuda.empty_cache()

        x_seq.append(x_tmp_cpu)

    #x_seq = np.concatenate(x_seq, axis=0)
    x_seq = torch.cat(x_seq, dim=0)

    return x_seq


def solver_ddim(model, clip, alphas_bar, a_mat, u_c_mat, my_t_list, *args):
    bs = a_mat.shape[0]
    N = a_mat.shape[1]
    DM_mat = np.zeros((bs, N, N, 1))
    P_ls = []

    for i in range(bs):

        a = a_mat[[i], ...]
        x_c = u_c_mat[[i], ...]

        x_generate_process = sample_backward_loop_ddim(model, clip, alphas_bar, a, x_c, my_t_list, *args)
        DM_mat[i, ...] = x_generate_process[-1, ...]
        P_ls.append(x_generate_process)

    return DM_mat, P_ls

def get_plot_sample_all(xx, yy, pd, pd_ddim, pd_guid, test_c, test_f, test_r, fno_pd, fig_name, idx):

    fig5, ax = plt.subplots(2, 6, figsize=(24, 8))
    plt.rcParams['axes.titlesize'] = 20
    cp0 = ax[0, 0].contourf(xx, yy, test_r[idx, ..., 1].detach().cpu().numpy(), 36, cmap=cm.jet)
    ax[0, 0].set_title('Reference')

    ax[0, 1].contourf(xx, yy, test_c[idx, ..., 1].detach().cpu().numpy(), 36, cmap=cm.jet)
    ax[0, 1].set_title('CSI')

    ax[0, 2].contourf(xx, yy, test_f[idx, ..., 1].detach().cpu().numpy(), 36, cmap=cm.jet)
    ax[0, 2].set_title('Fine')

    ax[0, 3].contourf(xx, yy, fno_pd[idx, ..., 0], 36, cmap=cm.jet)
    ax[0, 3].set_title('FNO')

    ax[0, 4].contourf(xx, yy, pd[idx, ..., 0], 36, cmap=cm.jet)
    ax[0, 4].set_title('DDPM')

    ax[0, 5].contourf(xx, yy, pd_guid[idx, ..., 0], 36, cmap=cm.jet)
    ax[0, 5].set_title('PGDM')

    big_error = np.abs(fno_pd[idx, ..., 0] - test_r[idx, ..., 1].detach().cpu().numpy())
    big_error2 = np.abs(test_c[idx, ..., 1].detach().cpu().numpy() - test_r[idx, ..., 1].detach().cpu().numpy())

    vmin = min(big_error.min(), big_error2.min())
    vmax = max(big_error.max(), big_error.max())

    # ax[1, 0].contourf(xx, yy, test_f[idx, ..., 0].detach().cpu().numpy(), 36, cmap=cm.jet)
    # ax[1, 0].set_title('source')
    ax[1, 0].axis('off')

    cp1 = ax[1, 1].contourf(xx, yy,
                        np.abs(test_c[idx, ..., 1].detach().cpu().numpy() - test_r[idx, ..., 1].detach().cpu().numpy()),
                        36, cmap=cm.jet, vmin=vmin, vmax=vmax)
    ax[1, 1].set_title('CSI error')

    ax[1, 2].contourf(xx, yy,
                   np.abs(test_f[idx, ..., 1].detach().cpu().numpy() - test_r[idx, ..., 1].detach().cpu().numpy()),
                   36, cmap=cm.jet, vmin=vmin, vmax=vmax)
    ax[1, 2].set_title('Fine error')

    ax[1, 3].contourf(xx, yy,
                   np.abs(fno_pd[idx, ..., 0] - test_r[idx, ..., 1].detach().cpu().numpy()),
                   36, cmap=cm.jet, vmin=vmin, vmax=vmax)
    ax[1, 3].set_title('FNO error')

    ax[1, 4].contourf(xx, yy, np.abs(pd[idx, ..., 0] - test_r[idx, ..., 1].detach().cpu().numpy()), 36,
                   cmap=cm.jet, vmin=vmin, vmax=vmax)
    ax[1, 4].set_title('DDPM error')

    ax[1, 5].contourf(xx, yy, np.abs(pd_guid[idx, ..., 0] - test_r[idx, ..., 1].detach().cpu().numpy()), 36,
                   cmap=cm.jet, vmin=vmin, vmax=vmax)
    ax[1, 5].set_title('PGDM error')

    for axs in ax.flat:
        axs.set_xticks([])
        axs.set_yticks([])

    fig5.subplots_adjust(left=0.01, right=0.95, top=0.95, bottom=0.02)
    cbar_ax = fig5.add_axes([0.96, 0.53, 0.01, 0.42])
    cbar_ax2 = fig5.add_axes([0.96, 0.02, 0.01, 0.42])
    cb1 = fig5.colorbar(cp0, cax=cbar_ax)
    cb2 = fig5.colorbar(cp1, cax=cbar_ax2)

    cb1.ax.tick_params(labelsize=15)
    cb2.ax.tick_params(labelsize=15)

    fig5.savefig(fig_name + '_pdall.jpg')

    plt.show()
############################################################################################





