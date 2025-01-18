import torch
from torch import nn
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def A(x, Phi):
    temp = x * Phi
    y = torch.sum(temp, 1)
    return y


def At(y, Phi):
    temp = torch.unsqueeze(y, 1).repeat(1, Phi.shape[1], 1, 1)
    x = temp * Phi
    return x


def shift_3d(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    outputs = inputs.clone()
    for i in range(nC):
        outputs[:, i, :, :] = torch.roll(inputs[:, i, :, :], shifts=step * i, dims=2)
    return outputs


def shift_back_3d(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    outputs = inputs.clone()
    for i in range(nC):
        outputs[:, i, :, :] = torch.roll(
            inputs[:, i, :, :], shifts=(-1) * step * i, dims=2
        )
    return outputs


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
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
            nn.Conv2d(dim, 4 * dim, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(4 * dim, dim, kernel_size=1, padding=0),
        )
        self.se = SELayer(dim, reduction=16)

    def forward(self, x):
        short_cut = x
        x = self.block(x)
        x = self.se(x)
        x = x + short_cut
        return x


class IterationModule(nn.Module):
    def __init__(self, dim):
        super(IterationModule, self).__init__()
        self.basic_dim = 28
        self.step = nn.Parameter(
            torch.zeros((1, self.basic_dim, 1, 1)) + 1e-3, requires_grad=True
        )
        self.lamd = nn.Parameter(
            torch.zeros((1, self.basic_dim, 1, 1)) + 1e-3, requires_grad=True
        )
        self.mu = nn.Parameter(
            torch.zeros((1, self.basic_dim, 1, 1)) + 1e-0, requires_grad=True
        )
        self.unit1 = nn.Sequential(
            nn.Conv2d(self.basic_dim, dim, kernel_size=1, padding=0),
            BasicBlock(dim),
            BasicBlock(dim),
            nn.Conv2d(dim, self.basic_dim, kernel_size=1, padding=0),
        )
        self.unit2 = nn.Sequential(
            nn.Conv2d(self.basic_dim, dim, kernel_size=1, padding=0),
            BasicBlock(dim),
            BasicBlock(dim),
            nn.Conv2d(dim, self.basic_dim, kernel_size=1, padding=0),
        )

    def softshrink(self, x, mu):
        z = torch.zeros_like(x)
        x = torch.sign(x) * torch.maximum(torch.abs(x) - mu, z)
        return x

    def forward(self, xk, dk, bk, y, Phi, Phi_s):
        ##########################STEP 1############################
        xk_t1 = shift_3d(xk)
        xk_t1 = A(xk_t1, Phi) - y
        xk_t1 = At(xk_t1, Phi)
        xk_t1 = shift_back_3d(xk_t1)

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
            nn.Conv2d(dim, dim * 2, kernel_size=2, stride=2),
            LayerNorm(2 * dim, eps=1e-6, data_format="channels_first"),
        )
        self.down2 = nn.Sequential(
            BasicBlock(2 * dim),
            nn.Conv2d(2 * dim, 4 * dim, kernel_size=2, stride=2),
            LayerNorm(4 * dim, eps=1e-6, data_format="channels_first"),
        )
        self.down3 = nn.Sequential(
            BasicBlock(4 * dim),
            nn.Conv2d(4 * dim, 8 * dim, kernel_size=2, stride=2),
            LayerNorm(8 * dim, eps=1e-6, data_format="channels_first"),
        )
        self.mid = nn.Sequential(
            BasicBlock(8 * dim),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=8 * dim, out_channels=4 * dim, kernel_size=2, stride=2
            ),
            LayerNorm(4 * dim, eps=1e-6, data_format="channels_first"),
            BasicBlock(4 * dim),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=4 * dim, out_channels=2 * dim, kernel_size=2, stride=2
            ),
            LayerNorm(2 * dim, eps=1e-6, data_format="channels_first"),
            BasicBlock(2 * dim),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=2 * dim, out_channels=dim, kernel_size=2, stride=2
            ),
            LayerNorm(dim, eps=1e-6, data_format="channels_first"),
            BasicBlock(dim),
        )

    def forward(self, x):
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode="reflect")

        xk1 = self.down1(x)
        xk2 = self.down2(xk1)
        xk3 = self.down3(xk2)
        xk4 = self.mid(xk3)
        xk5 = self.up1(xk4) + xk2
        xk6 = self.up2(xk5) + xk1
        x = self.up3(xk6) + x

        return x[:, :, :h_inp, :w_inp]


class USBNet(torch.nn.Module):
    def __init__(self, dim=32, depth=8):
        super(USBNet, self).__init__()
        self.depth = depth
        self.basic_dim = 28

        self.fution = nn.Conv2d(
            self.basic_dim * 2, self.basic_dim, kernel_size=1, padding=0
        )
        self.pre = nn.Sequential(
            nn.Conv2d(self.basic_dim, dim, kernel_size=3, padding=1),
            BasicBlock(dim),
        )
        self.iters = nn.ModuleList()
        self.blocks0 = nn.ModuleList()
        self.blocks1 = nn.ModuleList()
        self.blocks2 = nn.ModuleList()
        for i in range(self.depth):
            self.iters.append(IterationModule(dim=dim))
            self.blocks0.append(
                nn.Sequential(
                    nn.Conv2d(self.basic_dim, dim, kernel_size=3, padding=1),
                    BasicBlock(dim),
                )
            )
            self.blocks1.append(
                nn.Sequential(
                    UBlock(dim),
                )
            )
            self.blocks2.append(
                nn.Sequential(
                    LayerNorm(dim, eps=1e-6, data_format="channels_first"),
                    BasicBlock(dim),
                    nn.Conv2d(dim, self.basic_dim, kernel_size=3, padding=1),
                )
            )
        self.mid = nn.Sequential(
            nn.Conv2d(self.depth * self.basic_dim, dim, kernel_size=3, padding=1),
            LayerNorm(dim, eps=1e-6, data_format="channels_first"),
            BasicBlock(dim),
        )
        self.post = nn.Sequential(
            UBlock(2 * dim),
            LayerNorm(2 * dim, eps=1e-6, data_format="channels_first"),
            BasicBlock(2 * dim),
            nn.Conv2d(2 * dim, dim, kernel_size=3, padding=1),
            BasicBlock(dim),
            nn.Conv2d(dim, self.basic_dim, kernel_size=3, padding=1),
        )
        self.apply(self._init_weights)

    def forward(self, y, input_mask):
        # Init
        Phi, Phi_s = input_mask
        xk_i = self.initial(y, Phi)
        xk_i = shift_back_3d(xk_i)
        dk, bk = torch.zeros_like(xk_i), torch.zeros_like(xk_i)
        xk = self.pre(xk_i)
        inter_xk = []

        # Recon
        for i in range(self.depth):
            xk_i, dk, bk = self.iters[i](xk_i, dk, bk, y, Phi, Phi_s)
            xk = xk + self.blocks0[i](xk_i)
            xk = self.blocks1[i](xk)
            xk_i = self.blocks2[i](xk)
            inter_xk.append(xk_i)

        inter_xk = torch.cat(inter_xk, dim=1)
        inter_xk = self.mid(inter_xk)
        xk = torch.cat([xk, inter_xk], dim=1)
        xk = self.post(xk)

        return xk[:, :, :, : y.shape[1]]

    def initial(self, y, Phi):
        """
        :param y: [b,256,310]
        :param Phi: [b,28,256,310]
        :return: [b,28,256,310]
        """
        nC, step = 28, 2
        y = y / nC * 2
        bs, row, col = y.shape
        y_shift = torch.zeros(bs, nC, row, col).float().to(y.device)
        for i in range(nC):
            y_shift[:, i, :, step * i : step * i + col - (nC - 1) * step] = y[
                :, :, step * i : step * i + col - (nC - 1) * step
            ]
        x_init = self.fution(torch.cat([y_shift, Phi], dim=1))
        return x_init

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def load_pretrained_model():
    path = f"./usbnet_real.pth"
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint


@register_model
def usbnet(pretrained=False, **kwargs):
    model = USBNet(dim=32, depth=8)
    if pretrained:
        checkpoint = load_pretrained_model()
        model.load_state_dict(checkpoint["model"])
    return model
