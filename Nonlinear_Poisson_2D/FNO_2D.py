import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import *
from train_utils import *
from Net_2D import FNO2d2
import os
from config_2D import *

if __name__ == "__main__":
    # define grid
    Nx_f = m * (Nx_c + 1) - 1
    dx = 1 / (Nx_f + 1)
    points_x = np.linspace(dx, 1 - dx, Nx_f).T
    xx, yy = np.meshgrid(points_x, points_x)

    ### load training data ####
    cwd = os.getcwd()
    npy_name = cwd + '/data/' + 'Pos_2D_train' + '_Ns_' + str(Ns1) + '_Nx_' + str(Nx_c) + '_m_' + num2str_deciaml(
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

    clip = torch.max(torch.abs(g_mat[..., 1]))*1.05

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
    npy_name = cwd + '/data/' +'Pos_2D_test' + '_Ns_' + str(Ns2) + '_Nx_' + str(Nx_c) + '_m_' + num2str_deciaml(
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

    clip = torch.max(torch.abs(g_mat[..., 1])) * 1

    content = 'In test example, we clip at : %3f, avg time for coarse solver are: %3f, for fine solver are: %3f, for ref solver are: %3f' % (
        clip, tc, tf, tr)
    print(content)

    test_c, test_f, test_r = Set_c[:Nte, ...], Set_f[:Nte, ...], Set_r[:Nte, ...]
    test_c, test_f, test_r = make_tensor(test_c, test_f, test_r[:, 1::2, 1::2, :])  # test_r is the ref downsample

    ###############################################################################
    model = FNO2d2(modes1, modes2, nl, width).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.95)
    loss_func = nn.MSELoss()
    ###################################################################

    ### define save name
    save_name = 'FNO_P_2D_' + '_Ntr_' + str(Ntr) + '_Nx_' + str(Nx_c) + '_m_' + num2str_deciaml(
        m) + '_d0_' + num2str_deciaml(
        d0) + '_d1_' + num2str_deciaml(d1) + '_d2_' + num2str_deciaml(d2) + '_alp_' + num2str_deciaml(
        alpha) + '_tau_' + num2str_deciaml(tau)

    cwd = os.getcwd()
    log_name = cwd + '/logs/' + save_name + '_log.txt'
    fig_name = cwd + '/figs/' + save_name
    chkpts_name = cwd + '/mdls/' + save_name

    mylogger(log_name, content)

    ### start training
    tic = time.time()
    for k in range(FNO_epoch):
        model.train()
        for data in data_loader:
            a, x, x_c = data[0], data[1], data[2]
            a, x, x_c = a.to(device), x.to(device), x_c.to(device)
            # print(a.shape, x.shape, x_c.shape)

            pd = model(x_c, a)
            #pd = model(a)
            loss = loss_func(pd, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        if k % FNO_record_epoch == 0 and k > 0:
            ### record time and loss ###
            elapsed_time = time.time() - tic

            model.eval()

            fno_pd = model(test_c[..., [1]], test_f[..., [0]])
            # fno_pd = model(test_f[..., [0]])

            fno_pd = tensor2nump(fno_pd)

            error_fno = myRL2_np(tensor2nump(test_r[..., [1]]), fno_pd)

            content = 'at epoch %d the total training time is %3f and the empirical loss is: %3f, the test loss is: %3f' % (
                k, elapsed_time, loss, error_fno)
            print(content)
            mylogger(log_name, content)

    ### save model ###
    model.eval()
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
    torch.save(checkpoint, chkpts_name + '_final.pth')
