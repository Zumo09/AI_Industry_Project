"""
From the repository of the paper
https://github.com/cure-lab/SCINet
"""
from typing import Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

import numpy as np


@dataclass(frozen=True)
class SCIBlockCfg:
    input_dim: int = 0
    hidden_size: int = 1
    kernel_size: int = 3
    groups: int = 1
    dilation: int = 1
    dropout: float = 0.5

    @property
    def pad_l(self) -> int:
        return self._pad(2)

    @property
    def pad_r(self) -> int:
        return self._pad(0)

    def _pad(self, pad) -> int:
        # by default: stride==1, else we fix the kernel size of the second layer as 3.
        pad = pad if self.kernel_size % 2 == 0 else 1
        return self.dilation * (self.kernel_size - pad) // 2 + 1


class Interactor(nn.Sequential):
    def __init__(self, cfg: SCIBlockCfg) -> None:
        super().__init__(
            nn.ReplicationPad1d((cfg.pad_l, cfg.pad_r)),
            nn.Conv1d(
                cfg.input_dim,
                cfg.input_dim * cfg.hidden_size,
                kernel_size=cfg.kernel_size,
                dilation=cfg.dilation,
                stride=1,
                groups=cfg.groups,
            ),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Conv1d(
                cfg.input_dim * cfg.hidden_size,
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
        self.rho = Interactor(cfg)
        self.eta = Interactor(cfg)
        self.phi = Interactor(cfg)
        self.psi = Interactor(cfg)

    @staticmethod
    def split(x: Tensor) -> Tuple[Tensor, Tensor]:
        return x[:, ::2, :], x[:, 1::2, :]

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        eve, odd = self.split(x)

        eve = eve.permute(0, 2, 1)
        odd = odd.permute(0, 2, 1)

        e_s = eve.mul(torch.exp(self.psi(odd)))
        o_s = odd.mul(torch.exp(self.phi(eve)))

        eve: Tensor = e_s + self.eta(o_s)
        odd: Tensor = o_s - self.rho(e_s)

        eve = eve.permute(0, 2, 1)
        odd = odd.permute(0, 2, 1)

        return eve, odd


class EncoderTree(nn.Module):
    def __init__(self, num_levels: int, cfg: SCIBlockCfg) -> None:
        super().__init__()
        self.current_level = num_levels - 1
        self.workingblock = SCINetBlock(cfg)

        if self.current_level != 0:
            self.sub_tree_odd = EncoderTree(self.current_level, cfg)
            self.sub_tree_eve = EncoderTree(self.current_level, cfg)

    def zip_up_the_pants(self, even: Tensor, odd: Tensor) -> Tensor:
        # We recursively reordered these sub-series.
        even = even.permute(1, 0, 2)
        odd = odd.permute(1, 0, 2)  # L, B, D
        pants = []
        for e, o in zip(even, odd):
            pants.append(e.unsqueeze(0))
            pants.append(o.unsqueeze(0))
        if odd.shape[0] < even.shape[0]:
            pants.append(even[-1].unsqueeze(0))
        return torch.cat(pants, 0).permute(1, 0, 2)  # B, L, D

    def forward(self, x: Tensor) -> Tensor:
        eve, odd = self.workingblock(x)
        if self.current_level > 0:
            eve = self.sub_tree_eve(eve)
            odd = self.sub_tree_odd(odd)

        return self.zip_up_the_pants(eve, odd)


class Decoder(nn.Module):
    def __init__(self, input_len: int, output_len: int, num_layer: int) -> None:
        super().__init__()
        self.input_len = input_len
        self.overlap_len = input_len // 4
        self.div_len = input_len // 6
        self.num_layer = num_layer

        self.projection1 = nn.Conv1d(
            input_len, output_len, kernel_size=1, stride=1, bias=False
        )

        self.div_projection = nn.ModuleList()

        if self.num_layer > 1:
            self.projection1 = nn.Linear(input_len, output_len)
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

        log_timescale_increment = np.log(max_timescale / min_timescale) / max(
            num_timescales - 1, 1
        )
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment
        )
        self.inv_timescales: torch.Tensor
        self.register_buffer("inv_timescales", inv_timescales)

    def get_encoding(self, x: Tensor) -> Tensor:
        max_length = x.size()[1]
        position = torch.arange(
            max_length, dtype=torch.float32, device=x.device
        )  # tensor([0., 1., 2., 3., 4.], device='cuda:0')
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)  # 5 256
        signal = torch.cat(
            [torch.sin(scaled_time), torch.cos(scaled_time)], dim=1
        )  # [T, C]
        signal = F.pad(signal, (0, 0, 0, self.pe_hidden_size % 2))
        signal = signal.view(1, max_length, self.pe_hidden_size)

        return signal

    def forward(self, x: Tensor) -> Tensor:
        signal = self.get_encoding(x)
        if signal.shape[2] > x.shape[2]:
            return x + signal[:, :, :-1]
        else:
            return x + signal


class SCINet(nn.Module):
    def __init__(
        self,
        *,
        input_len: int,
        num_encoder_levels: int,
        num_decoder_layer: int,
        output_len: int,
        pos_enc: bool = False,
        block_config: SCIBlockCfg,
    ):
        super().__init__()

        assert (
            input_len % (np.power(2, num_encoder_levels)) == 0
        ), "evenly divided the input length into two parts. (e.g., 32 -> 16 -> 8 -> 4 for 3 levels)"

        print("Initializing Scinet...")
        self.pos_enc = PositionalEncoding(block_config.input_dim) if pos_enc else None
        print("Initializing Encoder Tree...")
        self.encoder = EncoderTree(num_encoder_levels, block_config)
        print("Initializing Decoder...")
        self.decoder = Decoder(input_len, output_len, num_decoder_layer)
        print("Ready")

    def pad(self, x: Tensor) -> Tensor:
        bottom = self.decoder.input_len - x.shape[1]
        return F.pad(x, (0, 0, 0, bottom), "replicate")

    def forward(self, x: Tensor) -> Tensor:
        print(x.shape, x.dtype)
        x = self.pad(x)
        if self.pos_enc is not None:
            x = self.pos_enc(x)

        print(x.shape, x.dtype)
        x += self.encoder(x)
        print(x.shape, x.dtype)
        x = self.decoder(x)

        return x
