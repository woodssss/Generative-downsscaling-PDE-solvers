import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import *
from data_utils import *
from train_utils import *
import functools
from Net_2D import UNet, UNet_attn, FNO2d2
import os
from Load_2D import *

if __name__ == "__main__":
    # define grid
    Nx_f = m * (Nx_c + 1) - 1
    dx = 1 / (Nx_f + 1)
    points_x = np.linspace(dx, 1 - dx, Nx_f).T
    xx, yy = np.meshgrid(points_x, points_x)

    ### load testing data ####
    cwd = os.getcwd()
    npy_name = cwd + '/data/' + 'Pos_2D_test' + '_Ns_' + str(Ns2) + '_Nx_' + str(Nx_c) + '_m_' + num2str_deciaml(
        m) + '_d0_' + num2str_deciaml(
        d0) + '_d1_' + num2str_deciaml(d1) + '_d2_' + num2str_deciaml(d2) + '_alp_' + num2str_deciaml(
        alpha) + '_tau_' + num2str_deciaml(tau) + '.npy'

    with open(npy_name, 'rb') as ss:
        Set_c = np.load(ss)
        Set_f = np.load(ss)
        Set_r = np.load(ss)
        tc = np.load(ss)
        tf = np.load(ss)
        tr = np.load(ss)

    g_mat = torch.from_numpy(Set_f).float()
    g_c_mat = torch.from_numpy(Set_c).float()

    clip = torch.max(torch.abs(g_mat[..., 1])) * clip_coeff

    content = 'In test example, we clip at : %3f, avg time for coarse solver are: %3f, for fine solver are: %3f, for ref solver are: %3f' % (
    clip, tc, tf, tr)
    print(content)

    test_c, test_f, test_r = Set_c[:Nte, ...], Set_f[:Nte, ...], Set_r[:Nte, ...]
    test_c, test_f, test_r = make_tensor(test_c, test_f, test_r[:, 1::2, 1::2, :]) # test_r is the ref downsample

    ##################### load UNet ##############################################
    load_iter = 20000
    if model_name=='UNet':
        model = UNet(Nt, embed_dim, Down_config, Up_config, Mid_config).to(device)
    elif model_name == 'UNet_attn':
        model = UNet_attn(Nt, embed_dim, Down_config, Up_config, Mid_config).to(device)

    model_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('UNet num of parameters', model_trainable_params)

    ### define diffusion parameters
    betas, alphas, alphas_bar, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, sigma = get_parameters(beta_start, beta_end, Nt)

    L_f = get_L(Nx_f, dim=2)
    myfunc_f, myjac_f = functools.partial(myfunc, L_f, d0, d1), functools.partial(myjac, L_f, d0, d1)

    generate_method = 'ddpm'

    ### load name
    save_name = 'DM_P_2D_' + model_name + '_c0_' + str(c0) + '_Nt_' + str(Nt) + '_Ntr_' + str(Ntr) + '_Nx_' + str(Nx_c) + '_m_' + num2str_deciaml(
        m) + '_d0_' + num2str_deciaml(
        d0) + '_d1_' + num2str_deciaml(d1) + '_d2_' + num2str_deciaml(d2) + '_alp_' + num2str_deciaml(
        alpha) + '_tau_' + num2str_deciaml(tau)

    chkpts_name = cwd + '/mdls/' + save_name + str(load_iter) + '.pth'
    fig_name = cwd + '/figs/' + save_name

    checkpoint = torch.load(chkpts_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    ################################################################################################

    ##################### load FNO ##############################################
    ### load FNO model ###
    tic_s = time.time()
    FNO_name = 'FNO_P_2D_' + '_Ntr_' + str(Ntr) + '_Nx_' + str(Nx_c) + '_m_' + num2str_deciaml(
        m) + '_d0_' + num2str_deciaml(
        d0) + '_d1_' + num2str_deciaml(d1) + '_d2_' + num2str_deciaml(d2) + '_alp_' + num2str_deciaml(
        alpha) + '_tau_' + num2str_deciaml(tau) + '_final.pth'
    FNO_checkpoint_path = cwd + '/mdls/' + FNO_name
    FNO_checkpoint = torch.load(FNO_checkpoint_path)

    FNO_mdl = FNO2d2(modes1, modes2, nl, width)
    FNO_mdl.load_state_dict(FNO_checkpoint['model_state_dict'])
    ###################################################################

    ############## FNO solver result #################################
    tic_fno = time.time()
    fno_pd = FNO_mdl(test_c[..., [1]].detach().cpu(), test_f[..., [0]].detach().cpu())

    fno_pd = tensor2nump(fno_pd)
    toc = time.time() - tic_fno
    avg_time_fno = toc / test_f.shape[0]
    ##############################################################

    ############## Diffusion model result ####################################

    ### ddpm ###
    tic_ddpm = time.time()
    pd, _ = solver_ddpm(model, clip, sigma, alphas, one_minus_alphas_bar_sqrt, Nt, test_f[..., [0]], test_c[..., [1]])
    toc = time.time() - tic_ddpm
    avg_time_ddpm = toc / test_f.shape[0]

    ### ddim ###
    tic_ddim = time.time()
    pd_ddim, _ = solver_ddim(model, clip, alphas_bar, test_f[..., [0]], test_c[..., [1]], my_t_list_skip)
    toc = time.time() - tic_ddim
    avg_time_ddim = toc / test_f.shape[0]

    ### fine tune ###
    coarse_ft = tensor2nump(test_c[..., [1]])
    fno_pd_ft = fno_pd.copy()
    pd_ft = pd.copy()
    pd_ddim_ft = pd_ddim.copy()
    tic_ft = time.time()
    for kk in range(1):
        coarse_ft = get_GNA(myfunc_f, myjac_f, d2, coarse_ft, tensor2nump(test_f[..., [0]]))
        fno_pd_ft = get_GNA(myfunc_f, myjac_f, d2, fno_pd_ft, tensor2nump(test_f[..., [0]]))
        pd_ft = get_GNA(myfunc_f, myjac_f, d2, pd_ft, tensor2nump(test_f[..., [0]]))
        pd_ddim_ft = get_GNA(myfunc_f, myjac_f, d2, pd_ddim_ft, tensor2nump(test_f[..., [0]]))

    toc_ft = time.time() - tic_ft
    avg_ft_time = toc_ft / test_f.shape[0]/4

    error_pd = myRL2_np(tensor2nump(test_r[..., [1]]), pd)
    error_pd_ddim = myRL2_np(tensor2nump(test_r[..., [1]]), pd_ddim)
    error_fno = myRL2_np(tensor2nump(test_r[..., [1]]), fno_pd)
    error_c = myRL2_np(tensor2nump(test_r[..., [1]]), tensor2nump(test_c[..., [1]]))
    error_f = myRL2_np(tensor2nump(test_r[..., [1]]), tensor2nump(test_f[..., [1]]))

    error_c_ft = myRL2_np(tensor2nump(test_r[..., [1]]), (coarse_ft))
    error_fno_ft = myRL2_np(tensor2nump(test_r[..., [1]]), (fno_pd_ft))
    error_pd_ft = myRL2_np(tensor2nump(test_r[..., [1]]), (pd_ft))
    error_pd_ddim_ft = myRL2_np(tensor2nump(test_r[..., [1]]), (pd_ddim_ft))


    content1 = 'compute time of coarse solver is: %3f, fine solver is: %3f, FNO is: %3f, ddpm is: %3f, ddim is: %3f' % (
        tc, tf, avg_time_fno, avg_time_ddpm, avg_time_ddim)

    content2 = 'compute time of coarse+ft is is: %3f, FNO+ft is: %3f, PGDM is: %3f' % (
        tc + avg_ft_time, avg_time_fno + avg_ft_time, avg_time_ddim + avg_ft_time)

    content3 = 'Relative L2 error of coarse solver is: %3f, fine solver is: %3f, FNO is: %3f, ddpm is: %3f, ddim is: %3f' % (
        error_c, error_f, error_fno, error_pd, error_pd_ddim)

    content4 = 'Relative L2 error of coarse solver + ft is: %3f, fine solver is: %3f, FNO + ft is: %3f, PGDM is: %3f' % (
        error_c_ft, error_f, error_fno_ft, error_pd_ddim_ft)

    print(content1)
    print(content2)
    print(content3)
    print(content4)

    get_plot_sample_all(xx, yy, pd, pd_ddim, pd_ddim_ft, test_c, test_f, test_r, fno_pd, fig_name, idx=1)







