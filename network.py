""" Main NoiseDF network. """

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange
from network_ridnet import RIDNET


class SeparableConv2d(nn.Module):
    """ Separable Convolutional Layer. """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class SiameseNet(nn.Module):
    """ Siamese Noise Feature Extraction Module. """

    def __init__(self, in_nc=3, out_c=64, kernel_size=7, stride=4, padding=2):
        super(SiameseNet, self).__init__()
        self.ridnet = RIDNET()
        self.separableConv2d = SeparableConv2d(in_nc, out_c, kernel_size, stride, padding)
        self.Rearrange = Rearrange('b c h w -> b (h w) c')
        self.norm = nn.LayerNorm(out_c)

    def forward(self, x):
        output = self.ridnet(x)
        # The clean picture minus the original picture is the noise
        noise = output - x
        feature = self.separableConv2d(noise)
        feature = self.Rearrange(feature)
        out = self.norm(feature)
        return out


class RelativeInteraction(nn.Module):
    """ Multi-Head Relative-Interaction. """

    def __init__(self, img_size, in_c, kernel=3, num_heads=8, head_dim=64, stride=1, padding=1, dilation=1, dropout=0):
        super(RelativeInteraction, self).__init__()
        self.img_size = img_size // 4
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.energy_dim = head_dim * num_heads

        # Face projection.
        self.face_proj = nn.Sequential(
            SeparableConv2d(in_c, in_c, kernel_size=kernel, stride=stride, padding=padding, dilation=dilation),
            torch.nn.Conv2d(in_c, self.energy_dim, kernel_size=1),
            nn.BatchNorm2d(self.energy_dim)
        )
        # Background projection.
        self.background_proj = nn.Sequential(
            SeparableConv2d(in_c, in_c, kernel_size=kernel, stride=stride, padding=padding, dilation=dilation),
            torch.nn.Conv2d(in_c, self.energy_dim, kernel_size=1),
            nn.BatchNorm2d(self.energy_dim)
        )
        self.fc_energy = nn.Sequential(
            nn.Linear(self.img_size ** 2, in_c),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.BatchNorm2d(self.num_heads)
        )
        self.fc_out = nn.Sequential(
            nn.Linear(self.energy_dim, in_c),
            nn.Dropout(dropout)
        )

    def forward(self, face, background):
        face = rearrange(face, 'b (h w) c -> b c h w', h=self.img_size, w=self.img_size)
        background = rearrange(background, 'b (h w) c -> b c h w', h=self.img_size, w=self.img_size)
        face_out = self.face_proj(face)
        face_out = rearrange(face_out, 'b (n d) h w -> b n (h w) d', n=self.num_heads)
        background_out = self.background_proj(background)
        background_out = rearrange(background_out, 'b (n d) h w -> b n (h w) d', n=self.num_heads)
        # Multi-Head Relative-Interaction.
        energy = torch.einsum('b n j d, b h k d -> b n j k', face_out, background_out)
        interaction = F.softmax(energy / ((self.head_dim * self.num_heads) ** (1 / 2)), dim=-1)
        out = self.fc_energy(interaction)
        out = rearrange(out, 'b n l d -> b l (n d)')
        out = self.fc_out(out)
        return out


class NoiseDF(nn.Module):
    """ NoiseDF Main Network. """

    def __init__(self, img_size, in_nc=3, out_c=64, kernel_size=7, stride=4,
                 padding=2, dropout=0):
        super(NoiseDF, self).__init__()
        self.img_size = img_size
        self.siamese = SiameseNet(in_nc, out_c, kernel_size, stride)
        self.interaction = RelativeInteraction(img_size, out_c)
        self.refine_dim = ((img_size - kernel_size + 2 * padding) // stride + 1) ** 2
        self.fc_refine = nn.Sequential(
            nn.Linear(self.refine_dim, out_c),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.fc_out = nn.Sequential(
            nn.Linear(out_c * out_c, out_c),
            nn.GELU(),
            nn.Linear(out_c, 1)
        )

    def forward(self, face, background):
        face = self.siamese(face)
        background = self.siamese(background)
        interaction = self.interaction(face, background)
        interaction = interaction.permute(0, 2, 1)
        refine = self.fc_refine(interaction)
        b_size = refine.shape[0]
        refine = refine.view(b_size, -1)
        out = self.fc_out(refine)
        return out
