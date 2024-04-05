################## define config for 2D  poisson ###
from utils import num2str_deciaml
### data set config
Ns1 = 200
Ns2 = 10

Ntr = 100
Nte = 2

Nx_c = 16
m = 8
d0 = -0.0005
d1 = 1.0
d2 = 1.0
alpha = 1.6
tau = 7.0

### FNO config ###
modes1, modes2, nl, width = 12, 12, 3, 32
FNO_epoch = 5000
FNO_record_epoch = 1000

### diffussion model config ###
Nt = 400
beta_start, beta_end = 1e-4, 0.02
clip_coeff = 1.01
c0 = 128
batch_size = 4
model_name = 'UNet'
#model_name = 'UNet_attn'
embed_dim = 256
gn = 32

############## UNet ############################
Down_config = [
    # Tuple: (in_cn, out_cn, kernel_size, stride, embed_dim, n_head, head_dim)
    (3, c0, 9, 1, embed_dim),
    (c0, 2*c0, 3, 2, embed_dim),
    (2*c0, 4 * c0, 3, 2, embed_dim),
    (4 * c0, 8 * c0, 3, 2, embed_dim),
]

Mid_config = [
    # Tuple: (in_cn, out_cn, kernel_size, stride, embed_dim, n_head, head_dim)
    (8 * c0, 8 * c0, 1, 1, embed_dim),
]

Up_config = [
    # Tuple: (in_cn, out_cn, kernel_size, stride, embed_dim, n_head, head_dim)
    (8 * c0, 4 * c0, 3, 2, embed_dim),
    (4 * c0 + 4*c0, 2 * c0, 3, 2, embed_dim),
    (2 * c0 + 2*c0, c0, 3, 2, embed_dim),
    (c0 + c0, 1, 9, 1, embed_dim),
]

### define save name
if len(Mid_config)>0:
    save_name = 'DM_P_2D_' + model_name + '_c0_' + str(c0) + '_Nt_' + str(Nt) + '_Ntr_' + str(Ntr) + '_Nx_' + str(Nx_c) + '_m_' + num2str_deciaml(
        m) + '_d0_' + num2str_deciaml(
        d0) + '_d1_' + num2str_deciaml(d1) + '_d2_' + num2str_deciaml(d2) + '_alp_' + num2str_deciaml(
        alpha) + '_tau_' + num2str_deciaml(tau)
else:
    save_name = 'DM_Pnm_2D_' + model_name + '_c0_' + str(c0) + '_Nt_' + str(Nt) + '_Ntr_' + str(Ntr) + '_Nx_' + str(Nx_c) + '_m_' + num2str_deciaml(
        m) + '_d0_' + num2str_deciaml(
        d0) + '_d1_' + num2str_deciaml(d1) + '_d2_' + num2str_deciaml(d2) + '_alp_' + num2str_deciaml(
        alpha) + '_tau_' + num2str_deciaml(tau)

### training config ##
num_epoch = 20001
record_epoch = 400

### generating config
generate_method = 'ddpm'

samp_start, samp_step = 5, 5
my_t_list_1 = [i for i in range(0, samp_start)]
my_t_list_2 = [j for j in range(samp_start, Nt, samp_step)]

my_t_list_skip = my_t_list_1 + my_t_list_2
my_t_list = [i for i in range(0, Nt)]