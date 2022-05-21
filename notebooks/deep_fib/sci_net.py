"""
From the repository of the paper
https://github.com/cure-lab/SCINet
"""
from dataclasses import dataclass
import math
from typing import Tuple
import torch.nn.functional as F
from torch import nn
from torch import Tensor
import torch
import numpy as np


@dataclass(frozen=True)
class SCIBlockCfg:
    input_dim: int = 0
    hidden_size: int = 1
    kernel_size: int = 3
    groups: int = 1
    dilation: int = 1
    dropout: float = 0.5
    modified: bool = True

    @property
    def pad_l(self) -> int:
        return self._pad(2)

    @property
    def pad_r(self):
        return self._pad(0)

    def _pad(self, pad) -> int:
        # by default: stride==1, else we fix the kernel size of the second layer as 3.
        pad = pad if self.kernel_size % 2 == 0 else 1
        return self.dilation * (self.kernel_size - pad) // 2 + 1


class Splitting(nn.Module):
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns the odd and even part"""
        return x[:, ::2, :], x[:, 1::2, :]


class Interactor(nn.Sequential):
    def __init__(self, prev_size: int, cfg: SCIBlockCfg) -> None:
        super().__init__(
            nn.ReplicationPad1d((cfg.pad_l, cfg.pad_r)),
            nn.Conv1d(
                cfg.input_dim * prev_size,
                int(cfg.input_dim * cfg.hidden_size),
                kernel_size=cfg.kernel_size,
                dilation=cfg.dilation,
                stride=1,
                groups=cfg.groups,
            ),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Conv1d(
                int(cfg.input_dim * cfg.hidden_size),
                cfg.input_dim,
                kernel_size=3,
                stride=1,
                groups=cfg.groups,
            ),
            nn.Tanh(),
        )


class SCINetBlock(nn.Module):
    def __init__(self, cfg: SCIBlockCfg) -> None:
        super().__init__()
        self.modified = cfg.modified

        self.split = Splitting()

        prev_size = 1
        self.P = Interactor(prev_size, cfg)
        self.U = Interactor(prev_size, cfg)
        self.phi = Interactor(prev_size, cfg)
        self.psi = Interactor(prev_size, cfg)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x_even, x_odd = self.split(x)

        x_even: Tensor = x_even.permute(0, 2, 1)
        x_odd: Tensor = x_odd.permute(0, 2, 1)

        if self.modified:
            d = x_odd.mul(torch.exp(self.phi(x_even)))
            c = x_even.mul(torch.exp(self.psi(x_odd)))

            x_even_update = c + self.U(d)
            x_odd_update = d - self.P(c)
            x_even, x_odd = x_even_update, x_odd_update

        else:
            d = x_odd - self.P(x_even)
            c = x_even + self.U(d)
            x_even, x_odd = c, d

        x_even = x_even.permute(0, 2, 1)
        x_odd = x_odd.permute(0, 2, 1)

        return x_even, x_odd


class EncoderTree(nn.Module):
    def __init__(self, num_levels: int, cfg: SCIBlockCfg) -> None:
        super().__init__()
        self.current_level = num_levels - 1
        self.workingblock = SCINetBlock(cfg)

        if self.current_level != 0:
            self.SCINet_Tree_odd = EncoderTree(self.current_level, cfg)
            self.SCINet_Tree_even = EncoderTree(self.current_level, cfg)

    def zip_up_the_pants(self, even: Tensor, odd: Tensor) -> Tensor:
        # We recursively reordered these sub-series.
        even = even.permute(1, 0, 2)
        odd = odd.permute(1, 0, 2)  # L, B, D
        even_len = even.shape[0]
        odd_len = odd.shape[0]
        mlen = min(odd_len, even_len)
        pants = []
        for i in range(mlen):
            pants.append(even[i].unsqueeze(0))
            pants.append(odd[i].unsqueeze(0))
        if odd_len < even_len:
            pants.append(even[-1].unsqueeze(0))
        return torch.cat(pants, 0).permute(1, 0, 2)  # B, L, D

    def forward(self, x):
        x_even_update, x_odd_update = self.workingblock(x)
        if self.current_level > 0:
            x_even_update = self.SCINet_Tree_even(x_even_update)
            x_odd_update = self.SCINet_Tree_odd(x_odd_update)

        return self.zip_up_the_pants(x_even_update, x_odd_update)


class PositionalEncoding(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        # For positional encoding
        self.pe_hidden_size = input_dim
        if self.pe_hidden_size % 2 == 1:
            self.pe_hidden_size += 1

        num_timescales = self.pe_hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0

        log_timescale_increment = math.log(
            float(max_timescale) / float(min_timescale)
        ) / max(num_timescales - 1, 1)
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment
        )
        self.inv_timescales: torch.Tensor
        self.register_buffer("inv_timescales", inv_timescales)

    def forward(self, x: Tensor) -> Tensor:
        max_length = x.size()[1]
        position = torch.arange(
            max_length, dtype=torch.float32, device=x.device
        )  # tensor([0., 1., 2., 3., 4.], device='cuda:0')
        # temp1 = position.unsqueeze(1)  # 5 1
        # temp2 = self.inv_timescales.unsqueeze(0)  # 1 256
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)  # 5 256
        signal = torch.cat(
            [torch.sin(scaled_time), torch.cos(scaled_time)], dim=1
        )  # [T, C]
        signal = F.pad(signal, (0, 0, 0, self.pe_hidden_size % 2))
        signal = signal.view(1, max_length, self.pe_hidden_size)

        return signal


class Decoder(nn.Module):
    def __init__(self, input_len, output_len, num_layer) -> None:
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.num_layer = num_layer
        self.projection1 = nn.Conv1d(
            input_len, self.output_len, kernel_size=1, stride=1, bias=False
        )

        self.overlap_len = input_len // 4
        self.div_len = input_len // 6

        self.div_projection = nn.ModuleList()

        if self.num_layer > 1:
            self.projection1 = nn.Linear(input_len, self.output_len)
            for _ in range(self.num_layer - 1):
                div_projection = nn.ModuleList()
                for i in range(6):
                    lens = (
                        min(i * self.div_len + self.overlap_len, input_len)
                        - i * self.div_len
                    )
                    div_projection.append(nn.Linear(lens, self.div_len))
                self.div_projection.append(div_projection)

    def forward(self, x: Tensor) -> Tensor:
        if self.num_layer == 1:
            return self.projection1(x)

        x = x.permute(0, 2, 1)
        for div_projection in self.div_projection:
            output = torch.zeros(x.shape, dtype=x.dtype).cuda()
            for i, div_layer in enumerate(div_projection):  # type: ignore
                a, b = i * self.div_len, (i + 1) * self.div_len
                mlen = min(i * self.div_len + self.overlap_len, self.input_len)
                div_x = x[:, :, a:mlen]
                output[:, :, a:b] = div_layer(div_x)
            x = output
        x = self.projection1(x)
        return x.permute(0, 2, 1)


class SCINet(nn.Module):
    def __init__(
        self,
        output_len: int,
        input_len: int,
        cfg: SCIBlockCfg,
        num_levels: int = 3,
        num_decoder_layer: int = 1,
        concat_len: int = 0,
        pos_enc: bool = False,
        RIN: bool = False,
    ):
        super(SCINet, self).__init__()

        assert (
            input_len % (np.power(2, num_levels)) == 0
        ), "evenly divided the input length into two parts. (e.g., 32 -> 16 -> 8 -> 4 for 3 levels)"

        self.concat_len = concat_len
        self.RIN = RIN

        self.pos_enc = PositionalEncoding(cfg.input_dim) if pos_enc else None
        self.encoder = EncoderTree(num_levels, cfg)
        self.decoder = Decoder(input_len, output_len, num_decoder_layer)

        ### RIN Parameters ###
        if self.RIN:
            self.affine_weight = nn.parameter.Parameter(torch.ones(1, 1, cfg.input_dim))
            self.affine_bias = nn.parameter.Parameter(torch.zeros(1, 1, cfg.input_dim))

    def forward(self, x: Tensor) -> Tensor:
        if self.pos_enc is not None:
            pe: Tensor = self.pos_enc(x)
            if pe.shape[2] > x.shape[2]:
                x += pe[:, :, :-1]
            else:
                x += pe

        ### activated when RIN flag is set ###
        if self.RIN:
            print("/// RIN ACTIVATED ///\r", end="")
            means = x.mean(1, keepdim=True).detach()
            # mean
            x = x - means
            # var
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
            # affine
            # print(x.shape,self.affine_weight.shape,self.affine_bias.shape)
            x = x * self.affine_weight + self.affine_bias
        else:
            stdev = 0
            means = 0

        # the first stack
        res1 = x
        x = self.encoder(x)
        x += res1
        x = self.decoder(x)

        if self.RIN:
            x = x - self.affine_bias
            x = x / (self.affine_weight + 1e-10)
            x = x * stdev
            x = x + means

        return x
