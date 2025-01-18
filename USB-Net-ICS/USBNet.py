import sys
import torch
from torch import nn
import numpy as np
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from einops.layers.torch import Rearrange
from utils.utils import LayerNorm
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BasicBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            LayerNorm(dim, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dim, 4*dim, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(4*dim, dim, kernel_size=1, padding=0),
        )
        self.se = SELayer(dim, reduction=16)
    
    def forward(self, x):
        short_cut = x
        x = self.block(x)
        x = self.se(x)
        x = x + short_cut
        return x


class IterationModule(nn.Module):
    def __init__(self, block_size, dim):
        super(IterationModule, self).__init__()
        self.block_size = block_size
        self.step = nn.Parameter(torch.tensor(1e-3), requires_grad=True)
        self.lamd = nn.Parameter(torch.tensor(1e-3), requires_grad=True)
        self.mu = nn.Parameter(torch.tensor(1e-0), requires_grad=True)
        self.unit1 = (nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=1, padding=0),
            BasicBlock(dim),
            BasicBlock(dim),
            nn.Conv2d(dim, 1, kernel_size=1, padding=0),
        ))
        self.unit2 = (nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=1, padding=0),
            BasicBlock(dim),
            BasicBlock(dim),
            nn.Conv2d(dim, 1, kernel_size=1, padding=0),
        ))

    def softshrink(self, x, mu):
        z = torch.zeros_like(x)
        x = torch.sign(x) * torch.maximum(torch.abs(x) - mu, z)
        return x

    def forward(self, A, xk, dk, bk, y):
        ##########################STEP 1############################
        xk_t1 = F.conv2d(xk, A, stride=self.block_size,
                         padding=0, bias=None) - y
        xk_t1 = F.conv_transpose2d(xk_t1, A, stride=self.block_size)
        xk_t2 = self.unit1(xk)
        xk_t2 = self.unit2(xk_t2 - (dk - bk))
        xk = xk - self.step * xk_t1 - self.step * self.mu * xk_t2

        ##########################STEP 2############################
        xk_c = self.unit1(xk)
        dk = self.softshrink(xk_c + bk, self.lamd / (self.mu + 1e-6))

        ##########################STEP 3############################
        bk = bk + xk_c - dk
        return xk, dk, bk


class UBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.down1 = nn.Sequential(
            BasicBlock(dim),
            nn.Conv2d(dim, dim*2, kernel_size=2, stride=2),
            LayerNorm(2*dim, eps=1e-6, data_format="channels_first"),
        )
        self.down2 = nn.Sequential(
            BasicBlock(2*dim),
            nn.Conv2d(2*dim, 4*dim, kernel_size=2, stride=2),
            LayerNorm(4*dim, eps=1e-6, data_format="channels_first"),
        )
        self.down3 = nn.Sequential(
            BasicBlock(4*dim),
            nn.Conv2d(4*dim, 8*dim, kernel_size=2, stride=2),
            LayerNorm(8*dim, eps=1e-6, data_format="channels_first"),
        )
        self.mid = nn.Sequential(
            BasicBlock(8*dim),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8*dim, out_channels=4*dim, kernel_size=2, stride=2),
            LayerNorm(4*dim, eps=1e-6, data_format="channels_first"),
            BasicBlock(4*dim),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=4*dim, out_channels=2*dim, kernel_size=2, stride=2),
            LayerNorm(2*dim, eps=1e-6, data_format="channels_first"),
            BasicBlock(2*dim),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2*dim, out_channels=dim, kernel_size=2, stride=2),
            LayerNorm(dim, eps=1e-6, data_format="channels_first"),
            BasicBlock(dim),
        )

    def forward(self, x):
        xk1 = self.down1(x)
        xk2 = self.down2(xk1)
        xk3 = self.down3(xk2)
        xk4 = self.mid(xk3)
        xk5 = self.up1(xk4) + xk2 
        xk6 = self.up2(xk5) + xk1
        x = self.up3(xk6) + x
        return x


class USBNet(torch.nn.Module):
    def __init__(self, ratio, block_size=32, dim=32, depth=8):
        super(USBNet, self).__init__()
        self.ratio = ratio
        self.depth = depth
        self.block_size = block_size
        A = torch.from_numpy(self.load_sampling_matrix()).float()
        self.A = nn.Parameter(Rearrange('m (1 b1 b2) -> m 1 b1 b2', b1=self.block_size)(A), requires_grad=True)

        self.pre = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=3, padding=1),
            BasicBlock(dim),
        )
        self.iters = nn.ModuleList()
        self.blocks0 = nn.ModuleList()
        self.blocks1 = nn.ModuleList()
        self.blocks2 = nn.ModuleList()
        for i in range(self.depth):
            self.iters.append(IterationModule(self.block_size, dim=dim))
            self.blocks0.append(nn.Sequential(
                nn.Conv2d(1, dim, kernel_size=3, padding=1),
                BasicBlock(dim),
            ))
            self.blocks1.append(nn.Sequential(
                UBlock(dim),
            ))
            self.blocks2.append(nn.Sequential(
                LayerNorm(dim, eps=1e-6, data_format="channels_first"),
                BasicBlock(dim),
                nn.Conv2d(dim, 1, kernel_size=3, padding=1),
            ))

        self.mid = nn.Sequential(
            nn.Conv2d(self.depth, dim, kernel_size=3, padding=1),
            LayerNorm(dim, eps=1e-6, data_format="channels_first"),
            BasicBlock(dim),
        )
        self.post = nn.Sequential(
            UBlock(2*dim),
            LayerNorm(2*dim, eps=1e-6, data_format="channels_first"),
            BasicBlock(2*dim),
            nn.Conv2d(2*dim, dim, kernel_size=3, padding=1),
            BasicBlock(dim),
            nn.Conv2d(dim, 1, kernel_size=3, padding=1),
        )
        self.apply(self._init_weights)

    def forward(self, x):
        # Sampling
        y = F.conv2d(x, self.A, stride=self.block_size, padding=0, bias=None)

        # Init
        x_init = F.conv_transpose2d(y, self.A, stride=self.block_size)
        dk, bk = torch.zeros_like(x_init), torch.zeros_like(x_init)
        xk_i = x_init
        inter_xk = []
        xk = self.pre(x_init)

        # Recon
        for i in range(self.depth):
            xk_i, dk, bk = self.iters[i](self.A, xk_i, dk, bk, y)

            xk = xk + self.blocks0[i](xk_i)
            xk = self.blocks1[i](xk)
            xk_i = self.blocks2[i](xk)
            inter_xk.append(xk_i)


        inter_xk = torch.cat(inter_xk, dim=1)
        inter_xk = self.mid(inter_xk)
        xk = torch.cat([xk, inter_xk], dim=1)
        xk = self.post(xk)

        return xk

    def load_sampling_matrix(self):
        path = "../data/sampling_matrix"
        data = np.load(f'{path}/{self.ratio}_{self.block_size}.npy')
        return data

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def load_pretrained_model(ratio, model_name):
    path = f"./model/checkpoint-{model_name}-{ratio}-best.pth"
    checkpoint = torch.load(path, map_location='cpu')
    return checkpoint


@register_model
def usbnet(ratio, pretrained=False, **kwargs):
    model = USBNet(ratio, block_size=32, dim=32, depth=8)
    if pretrained:
        checkpoint = load_pretrained_model(ratio, sys._getframe().f_code.co_name)
        model.load_state_dict(checkpoint['model'])
    return model

