import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### FNO ###
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FourierLayer2D(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FourierLayer2D, self).__init__()
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.w0 = nn.Conv1d(width, width, 1)
        self.width = width

    def forward(self, x):
        # x shape [bs, width, Nx, Nx]
        bs, N = x.shape[0], x.shape[-1]
        y1 = self.conv0(x)
        y2 = self.w0(x.view(bs, self.width, -1)).view(bs, self.width, N, N)
        y = y1 + y2
        y = F.gelu(y)
        out = y
        return out

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, nl, width):
        super(FNO2d, self).__init__()
        self.fc0 = nn.Linear(1, width)

        self.layers_ls = nn.ModuleList()

        for i in range(nl):
            self.layers_ls.append(FourierLayer2D(modes1, modes2, width))

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x shape [bs, Nx, Nx, 1]
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        for layer in self.layers_ls:
            x = layer(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class FNO2d2(nn.Module):
    def __init__(self, modes1, modes2, nl, width):
        super(FNO2d2, self).__init__()
        self.fc0 = nn.Linear(2, width)

        self.layers_ls = nn.ModuleList()

        for i in range(nl):
            self.layers_ls.append(FourierLayer2D(modes1, modes2, width))

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)
        #self.dropout = nn.Dropout(0.05)

    def forward(self, x, a):
        # x shape [bs, Nx, Nx, 1]
        x = torch.cat([x, a], dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        #x = self.dropout(x)

        for layer in self.layers_ls:
            x = layer(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, W):
        super().__init__()
        self.W = W

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


class DownBlock(nn.Module):
    def __init__(self, in_cn, out_cn, kernel_size, stride, gn, embed_dim):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv2d(in_cn, out_cn, kernel_size, stride, bias=False)
        self.dense = Dense(embed_dim, out_cn)
        if out_cn > 32:
            self.groupnorm = nn.GroupNorm(num_groups=gn, num_channels=out_cn)
        else:
            self.groupnorm = nn.GroupNorm(num_groups=4, num_channels=out_cn)
        #self.groupnorm = nn.BatchNorm1d(num_features=out_cn)
        self.act = lambda x: x * torch.sigmoid(x)

        #self.conv2 = nn.Conv2d(in_cn, out_cn, kernel_size, stride, bias=False)

    def forward(self, x, embed):
        h = self.conv(x)
        h += self.dense(embed)
        h = self.groupnorm(h)
        return self.act(h)

class UpBlock(nn.Module):
    def __init__(self, in_cn, out_cn, kernel_size, stride, gn, embed_dim):
        super(UpBlock, self).__init__()
        self.out_cn = out_cn
        if out_cn >= 32:
            self.conv = nn.ConvTranspose2d(in_cn, out_cn, kernel_size, stride, bias=False)
            self.groupnorm = nn.GroupNorm(num_groups=gn, num_channels=out_cn)
            #self.groupnorm = nn.BatchNorm1d(num_features=out_cn)
            self.dense = Dense(embed_dim, out_cn)
        else:
            self.conv = nn.ConvTranspose2d(in_cn, out_cn, kernel_size, stride)
            self.dense = Dense(embed_dim, out_cn)
        self.act = lambda x: x * torch.sigmoid(x)
    def forward(self, x, embed):
        h = self.conv(x)
        if self.out_cn >= 32:
            h += self.dense(embed)
            h = self.groupnorm(h)
            return self.act(h)
        else:
            #return self.act(h + self.dense(embed))
            return h

class MidBlock(nn.Module):
    def __init__(self, in_cn, out_cn, kernel_size, stride, gn, embed_dim):
        super(MidBlock, self).__init__()
        self.conv = nn.Conv2d(in_cn, out_cn, kernel_size, stride)
        self.dense = Dense(embed_dim, out_cn)
        self.groupnorm = nn.GroupNorm(num_groups=gn, num_channels=in_cn, eps=1e-6, affine=True)
        #self.groupnorm = nn.BatchNorm1d(num_features=out_cn)
        self.act = lambda x: x * torch.sigmoid(x)
    def forward(self, x, embed):
        h = self.conv(x)
        h += self.dense(embed)
        h = self.groupnorm(h)
        return self.act(h)


class UNet(nn.Module):
    def __init__(self, Nt, embed_dim, W, Down_config, Mid_config, Up_config):
        super(UNet, self).__init__()
        self.act = lambda x: x * torch.sigmoid(x)
        self.embed = nn.Sequential(GaussianFourierProjection(W),
                                   nn.Linear(embed_dim, embed_dim))

        self.Down_layers = self._create_down_layer(Down_config)
        self.Mid_layers = self._create_mid_layer(Mid_config)
        self.Up_layers = self._create_up_layer(Up_config)
        self.Nt = Nt

    def forward(self, x, x_c, a, t):
        t = t / self.Nt
        embed = self.act(self.embed(t))

        x = torch.cat([x, x_c, a], dim=-1)
        h = x.permute(0, 3, 1, 2)

        h_ls = [h]

        for layer in self.Down_layers:
            h = layer(h, embed)
            h_ls.append(h)
            #print(h.shape)

        for layer in self.Mid_layers:
            #print('mm', h.shape)
            h = layer(h, embed)

        h_ls.pop()
        h = self.Up_layers[0](h, embed)

        for layer in self.Up_layers[1:]:
            bh = h_ls.pop()
            #print('uu', h.shape, bh.shape)
            h = layer(torch.cat([h, bh], dim=1), embed)

        out = h.permute(0, 2, 3, 1)
        return out

    def _create_up_layer(self, config):
        layers = nn.ModuleList()
        #print(config)
        for k in config:
            #print(k[0], k[1], k[2], k[3], k[4], k[5])
            tmp_layer = UpBlock(k[0], k[1], k[2], k[3], k[4], k[5])
            layers.append(tmp_layer)

        return layers

    def _create_down_layer(self, config):
        layers = nn.ModuleList()
        for k in config:
            tmp_layer = DownBlock(k[0], k[1], k[2], k[3], k[4], k[5])
            layers.append(tmp_layer)

        return layers

    def _create_mid_layer(self, config):
        layers = nn.ModuleList()
        for k in config:
            tmp_layer = MidBlock(k[0], k[1], k[2], k[3], k[4], k[5])
            layers.append(tmp_layer)

        return layers



