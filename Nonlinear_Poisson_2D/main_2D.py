import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import *
from data_utils import *
from train_utils import *
import functools

from Net_2D import UNet, UNet_attn

import os
from config_2D import *

if __name__ == "__main__":
    # define grid
    Nx_f = m * (Nx_c + 1) - 1
    dx = 1 / (Nx_f + 1)
    points_x = np.linspace(dx, 1 - dx, Nx_f).T
    xx, yy = np.meshgrid(points_x, points_x)

    cwd = os.getcwd()

    ### load training data ####
    npy_name = cwd + '/data/' + 'P_2D_train' + '_Ns_' + str(Ns1) + '_Nx_' + str(Nx_c) + '_m_' + num2str_deciaml(
        m) + '_d0_' + num2str_deciaml(
        d0) + '_d1_' + num2str_deciaml(d1) + '_d2_' + num2str_deciaml(d2) + '_alp_' + num2str_deciaml(
        alpha) + '_tau_' + num2str_deciaml(tau) + '.npy'

    with open(npy_name, 'rb') as ss:
        Set_c = np.load(ss)
        Set_f = np.load(ss)
        tc = np.load(ss)
        tf = np.load(ss)

    g_mat = torch.from_numpy(Set_f).float()
    g_c_mat = torch.from_numpy(Set_c).float()

    # print(Set_f.shape, Set_c.shape)
    # zxc

    clip = torch.max(torch.abs(g_mat[..., 1])) * clip_coeff

    content = 'In train example, we clip at : %3f, avg time for coarse solver are: %3f, for fine solver are: %3f' % (clip, tc, tf)
    print(content)

    dataset = []
    for i in range(Ntr):
        # contain a, u, u_c
        tmp_ls = []
        tmp_ls.append(g_mat[i, :, :, [0]])
        tmp_ls.append(g_mat[i, :, :, [1]])
        tmp_ls.append(g_c_mat[i, :, :, [1]])

        dataset.append(tmp_ls)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    ############################################################

    ### load testing data ####
    npy_name = cwd + '/data/' + 'P_2D_test' + '_Ns_' + str(Ns2) + '_Nx_' + str(Nx_c) + '_m_' + num2str_deciaml(
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

    ###################################################################
    if model_name=='UNet':
        model = UNet(Nt, embed_dim, Down_config, Up_config, Mid_config).to(device)
    elif model_name == 'UNet_attn':
        model = UNet_attn(Nt, embed_dim, Down_config, Up_config, Mid_config).to(device)

    model_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('UNet num of parameters', model_trainable_params)


    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)
    ###################################################################


    ### define diffusion parameters ##################################################
    betas, alphas, alphas_bar, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, sigma = get_parameters(beta_start, beta_end, Nt)

    L_f = get_L(Nx_f, dim=2)
    myfunc_f, myjac_f = functools.partial(myfunc, L_f, d0, d1), functools.partial(myjac, L_f, d0, d1)
    ################################################################################################
    log_name = cwd + '/logs/' + save_name + '_log.txt'
    fig_name = cwd + '/figs/' + save_name
    chkpts_name = cwd + '/mdls/' + save_name

    mylogger(log_name, content)



    ### start training
    tic = time.time()
    for k in range(num_epoch):
        model.train()
        for data in data_loader:
            a, x, x_c = data[0], data[1], data[2]
            a, x, x_c = a.to(device), x.to(device), x_c.to(device)
            loss = get_loss(model, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, Nt, a, x, x_c)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            scheduler.step()

        if k % record_epoch == 0 and k>0:
            model.eval()
            ### record time and loss ###
            elapsed_time = time.time() - tic
            content = 'at epoch %d the total training time is %3f and the empirical loss is: %3f' % (k, elapsed_time, loss)
            print(content)
            mylogger(log_name, content)


            ### generate at the intermedia step

            pd, Process = solver_ddpm(model, clip, sigma, alphas, one_minus_alphas_bar_sqrt, Nt, test_f[..., [0]], test_c[..., [1]])

            get_plot_sample_ddpm(Nt, xx, yy, Process, pd, test_c, test_f, fig_name, k)

            error_pd = myRL2_np(tensor2nump(test_r[..., [1]]), pd)
            error_c = myRL2_np(tensor2nump(test_r[..., [1]]), tensor2nump(test_c[..., [1]]))
            error_f = myRL2_np(tensor2nump(test_r[..., [1]]), tensor2nump(test_f[..., [1]]))

            content = 'at step: %d, Relative L2 error of coarse solver is: %3f, fine solver is: %3f, ddim is: %3f' % (
                k, error_c, error_f,  error_pd)

            mylogger(log_name, content)

            print(content)

            if k%1000 == 0:
                ### save model ###
                current_lr = scheduler.get_last_lr()[0]
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),  # Saving scheduler state is optional but can be useful
                    'current_lr': current_lr,
                    'epoch': k,  # Optional, add if you want to keep track of epochs
                    'loss': loss,
                    # ... include any other things you want to save
                }
                torch.save(checkpoint, chkpts_name + str(k) + '.pth')


