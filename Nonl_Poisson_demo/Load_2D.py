################## define config for 2D  poisson ###
import torch
import torch.nn as nn
import numpy as np
from utils import num2str_deciaml
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(25)
### data set config
Ns1 = 100
Ns2 = 30

Ntr = 30
Nte = 10

Nx_c = 16
m = 8
d0 = -0.0005
d1 = 1.0
d2 = 1.0
alpha = 1.6
tau = 7.0

### FNO config ###
modes1, modes2, nl, width = 16, 16, 2, 32
FNO_epoch = 5000
FNO_record_epoch = 5000

### diffussion model config ###
Nt = 600
beta_start, beta_end = 1e-4, 0.02
clip_coeff = 1.01
c0 = 128
batch_size = 2
model_name = 'UNet'
embed_dim = 256
gn = 8

load_iter = 8000

############## UNet ############################
W_np = np.random.randn(embed_dim // 2) * 30
W = nn.Parameter(torch.from_numpy(W_np).float(), requires_grad=False).to(device)
Down_config = [
    # Tuple: (in_cn, out_cn, kernel_size, stride, gn, embed_dim)
    (3, c0, 9, 1, gn, embed_dim),
    (c0, 2*c0, 3, 2, gn, embed_dim),
    (2*c0, 4 * c0, 3, 2, gn, embed_dim),
    (4 * c0, 8 * c0, 3, 2, gn, embed_dim),
]

Mid_config = [
    # Tuple: (in_cn, out_cn, kernel_size, stride, embed_dim, n_head, head_dim)
    (8 * c0, 8 * c0, 1, 1, gn, embed_dim),
    (8 * c0, 8 * c0, 1, 1, gn, embed_dim),
]

Up_config = [
    # Tuple: (in_cn, out_cn, kernel_size, stride, embed_dim, n_head, head_dim)
    (8 * c0, 4 * c0, 3, 2, gn, embed_dim),
    (4 * c0 + 4*c0, 2 * c0, 3, 2, gn, embed_dim),
    (2 * c0 + 2*c0, c0, 3, 2, gn, embed_dim),
    (c0 + c0, 1, 9, 1, gn, embed_dim),
]

### define save name
save_name = 'DM_P_2D_' + model_name + '_c0_' + str(c0) + '_Nt_' + str(Nt) + '_Ntr_' + str(Ntr) + '_Nx_' + str(Nx_c) + '_m_' + num2str_deciaml(
    m) + '_d0_' + num2str_deciaml(
    d0) + '_d1_' + num2str_deciaml(d1) + '_d2_' + num2str_deciaml(d2) + '_alp_' + num2str_deciaml(
    alpha) + '_tau_' + num2str_deciaml(tau)


### training config ##
num_epoch = 20001
record_epoch = 2000

### generating config
generate_method = 'ddpm'

samp_start, samp_step = 5, 5
my_t_list_1 = [i for i in range(0, samp_start)]
my_t_list_2 = [j for j in range(samp_start, Nt, samp_step)]

my_t_list_skip = my_t_list_1 + my_t_list_2
my_t_list = [i for i in range(0, Nt)]