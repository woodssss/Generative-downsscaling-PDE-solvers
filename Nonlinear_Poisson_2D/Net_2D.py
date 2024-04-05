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
        self.dropout = nn.Dropout(0.05)

    def forward(self, x, a):
        # x shape [bs, Nx, Nx, 1]
        x = torch.cat([x, a], dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x = self.dropout(x)

        for layer in self.layers_ls:
            x = layer(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

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


class UNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, Nt, embed_dim, Down_config, Up_config, Mid_config):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.Nt = Nt
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))
        # Down layers
        self.conv1 = nn.Conv2d(Down_config[0][0], Down_config[0][1], Down_config[0][2], Down_config[0][3], bias=False)
        self.dense1 = Dense(embed_dim, Down_config[0][1])
        self.gnorm1 = nn.GroupNorm(4, num_channels=Down_config[0][1])

        self.conv2 = nn.Conv2d(Down_config[1][0], Down_config[1][1], Down_config[1][2], Down_config[1][3], bias=False)
        self.dense2 = Dense(embed_dim, Down_config[1][1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=Down_config[1][1])

        self.conv3 = nn.Conv2d(Down_config[2][0], Down_config[2][1], Down_config[2][2], Down_config[2][3], bias=False)
        self.dense3 = Dense(embed_dim, Down_config[2][1])
        self.gnorm3 = nn.GroupNorm(32, num_channels=Down_config[2][1])

        self.conv4 = nn.Conv2d(Down_config[3][0], Down_config[3][1], Down_config[3][2], Down_config[3][3], bias=False)
        self.dense4 = Dense(embed_dim, Down_config[3][1])
        self.gnorm4 = nn.GroupNorm(32, num_channels=Down_config[3][1])

        ### mid layers
        if len(Mid_config)>0:
            self.mconv1 = nn.Conv2d(Mid_config[0][0], Mid_config[0][1], Mid_config[0][2], Mid_config[0][3], bias=False)
            self.mdense1 = Dense(embed_dim, Mid_config[0][1])
            self.mgnorm1 = nn.GroupNorm(32, num_channels=Mid_config[0][1])
        else:
            self.mconv1 = None

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(Up_config[0][0], Up_config[0][1], Up_config[0][2], Up_config[0][3], bias=False)
        self.dense5 = Dense(embed_dim, Up_config[0][1])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=Up_config[0][1])

        self.tconv3 = nn.ConvTranspose2d(Up_config[1][0], Up_config[1][1], Up_config[1][2], Up_config[1][3], bias=False)
        self.dense6 = Dense(embed_dim, Up_config[1][1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=Up_config[1][1])

        self.tconv2 = nn.ConvTranspose2d(Up_config[2][0], Up_config[2][1], Up_config[2][2], Up_config[2][3], bias=False)
        self.dense7 = Dense(embed_dim, Up_config[2][1])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=Up_config[2][1])

        self.tconv1 = nn.ConvTranspose2d(Up_config[3][0], Up_config[3][1], Up_config[3][2], Up_config[3][3])

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, x_c, a, t):
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.embed(t/self.Nt))
        x = torch.cat([x, x_c, a], dim=-1)
        x = x.permute(0, 3, 1, 2)



        # Encoding path
        h1 = self.conv1(x)
        ## Incorporate information from t

        h1 += self.dense1(embed)
        ## Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)

        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        ### mid ###
        if self.mconv1!=None:
            h4 = self.mconv1(h4)
            h4 += self.mdense1(embed)
            h4 = self.mgnorm1(h4)
            h4 = self.act(h4)

        # Decoding path
        h = self.tconv4(h4)
        ## Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))
        return h.permute(0, 2, 3, 1)

######### attention #################################
class Attention(nn.Module):
    def __init__(self, dim, n_head=2, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = n_head
        hidden_dim = dim_head * n_head

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)


class UNet_attn(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, Nt, embed_dim, Down_config, Up_config, Mid_config):
        """Initialize a time-dependent score-based network.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        #self.dropout = nn.Dropout(0.05)
        self.Nt = Nt
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))
        # Down layers
        self.conv1 = nn.Conv2d(Down_config[0][0], Down_config[0][1], Down_config[0][2], Down_config[0][3], bias=False)
        self.dense1 = Dense(embed_dim, Down_config[0][1])
        self.gnorm1 = nn.GroupNorm(4, num_channels=Down_config[0][1])
        self.attn1 = Attention(Down_config[0][1])

        self.conv2 = nn.Conv2d(Down_config[1][0], Down_config[1][1], Down_config[1][2], Down_config[1][3], bias=False)
        self.dense2 = Dense(embed_dim, Down_config[1][1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=Down_config[1][1])
        #self.attn2 = Attention(Down_config[1][1])

        self.conv3 = nn.Conv2d(Down_config[2][0], Down_config[2][1], Down_config[2][2], Down_config[2][3], bias=False)
        self.dense3 = Dense(embed_dim, Down_config[2][1])
        self.gnorm3 = nn.GroupNorm(32, num_channels=Down_config[2][1])
        #self.attn3 = Attention(Down_config[2][1])

        self.conv4 = nn.Conv2d(Down_config[3][0], Down_config[3][1], Down_config[3][2], Down_config[3][3], bias=False)
        self.dense4 = Dense(embed_dim, Down_config[3][1])
        self.gnorm4 = nn.GroupNorm(32, num_channels=Down_config[3][1])
        self.attn4 = Attention(Down_config[3][1])

        ### mid layers
        if len(Mid_config)>0:
            self.mconv1 = nn.Conv2d(Mid_config[0][0], Mid_config[0][1], Mid_config[0][2], Mid_config[0][3], bias=False)
            self.mdense1 = Dense(embed_dim, Mid_config[0][1])
            self.mgnorm1 = nn.GroupNorm(32, num_channels=Mid_config[0][1])
            self.mattn1 = Attention(Mid_config[0][1])
        else:
            self.mconv1 = None

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(Up_config[0][0], Up_config[0][1], Up_config[0][2], Up_config[0][3], bias=False)
        self.dense5 = Dense(embed_dim, Up_config[0][1])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=Up_config[0][1])
        self.attn5 = Attention(Up_config[0][1])

        self.tconv3 = nn.ConvTranspose2d(Up_config[1][0], Up_config[1][1], Up_config[1][2], Up_config[1][3], bias=False)
        self.dense6 = Dense(embed_dim, Up_config[1][1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=Up_config[1][1])
        #self.attn6 = Attention(Up_config[1][1])

        self.tconv2 = nn.ConvTranspose2d(Up_config[2][0], Up_config[2][1], Up_config[2][2], Up_config[2][3], bias=False)
        self.dense7 = Dense(embed_dim, Up_config[2][1])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=Up_config[2][1])
        #self.attn7 = Attention(Up_config[2][1])

        self.tconv1 = nn.ConvTranspose2d(Up_config[3][0], Up_config[3][1], Up_config[3][2], Up_config[3][3])

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, x_c, a, t):
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.embed(t/self.Nt))
        x = torch.cat([x, x_c, a], dim=-1)
        x = x.permute(0, 3, 1, 2)

        #x = self.dropout(x)

        # Encoding path
        h1 = self.conv1(x)
        ## Incorporate information from t

        h1 += self.dense1(embed)
        #h1 += self.attn1(h1)
        ## Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)

        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        #h2 += self.attn2(h2)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)

        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        #h3 += self.attn3(h3)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)

        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = h4 + self.attn4(h4)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        ### mid ###
        if self.mconv1:
            h4 = self.mconv1(h4)
            h4 += self.mdense1(embed)
            h4 = h4 + self.mattn1(h4)
            h4 = self.mgnorm1(h4)
            h4 = self.act(h4)

        # Decoding path
        h = self.tconv4(h4)
        ## Skip connection from the encoding path
        h += self.dense5(embed)
        h = h + self.attn5(h)
        h = self.tgnorm4(h)
        h = self.act(h)

        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        #h += self.attn6(h)
        h = self.tgnorm3(h)
        h = self.act(h)

        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        #h += self.attn7(h)
        h = self.tgnorm2(h)
        h = self.act(h)

        h = self.tconv1(torch.cat([h, h1], dim=1))
        return h.permute(0, 2, 3, 1)

