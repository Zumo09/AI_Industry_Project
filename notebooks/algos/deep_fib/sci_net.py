"""
From the repository of the paper
https://github.com/cure-lab/SCINet
Simplified to the level as descibed in the paper
"""
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch import Tensor

import numpy as np


def _pad(kernel_size: int, pad: int) -> int:
    pad = pad if kernel_size % 2 == 0 else 1
    return (kernel_size - pad) // 2 + 1


class Interactor(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,  # hidden_multiplier
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__(
            nn.ReplicationPad1d((_pad(kernel_size, 2), _pad(kernel_size, 0))),
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=input_dim * hidden_size,
                kernel_size=kernel_size,
            ),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(
                in_channels=input_dim * hidden_size,
                out_channels=input_dim,
                kernel_size=3,
            ),
            nn.Tanh(),
        )


class SCINetBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.rho = Interactor(input_dim, hidden_size, kernel_size, dropout)
        self.eta = Interactor(input_dim, hidden_size, kernel_size, dropout)
        self.phi = Interactor(input_dim, hidden_size, kernel_size, dropout)
        self.psi = Interactor(input_dim, hidden_size, kernel_size, dropout)

    @staticmethod
    def split(x: Tensor) -> Tuple[Tensor, Tensor]:
        return x[:, ::2, :].clone(), x[:, 1::2, :].clone()

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
    def __init__(
        self,
        num_levels: int,
        input_dim: int,
        hidden_size: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.current_level = num_levels - 1
        self.workingblock = SCINetBlock(input_dim, hidden_size, kernel_size, dropout)

        if self.current_level > 0:
            self.sub_tree_odd = EncoderTree(
                self.current_level, input_dim, hidden_size, kernel_size, dropout
            )
            self.sub_tree_eve = EncoderTree(
                self.current_level, input_dim, hidden_size, kernel_size, dropout
            )

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
    def __init__(
        self, input_len: int, output_len: int, hidden_sizes: Optional[List[int]] = None
    ) -> None:
        super().__init__()
        lens = [input_len] + (hidden_sizes or []) + [output_len]
        layers = []
        for i, o in zip(lens[:-1], lens[1:]):
            layers += [nn.Linear(i, o), nn.ReLU()]
        layers[-1] = nn.Sigmoid()
        self.fcn = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 1)
        x = self.fcn(x)
        return x.permute(0, 2, 1)


class SCINet(nn.Module):
    def __init__(
        self,
        *,
        input_len: int,
        output_len: int,
        num_encoder_levels: int,
        hidden_decoder_sizes: Optional[List[int]] = None,
        input_dim: int = 0,
        hidden_size: int = 1,
        kernel_size: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()

        if input_len % np.power(2, num_encoder_levels) != 0:
            raise ValueError(
                f"Input of len {input_len} cannot be evenly divided in {num_encoder_levels} levels"
            )

        self.encoder = EncoderTree(
            num_encoder_levels, input_dim, hidden_size, kernel_size, dropout
        )
        self.decoder = Decoder(input_len, output_len, hidden_decoder_sizes)

    def forward(self, x: Tensor) -> Tensor:
        x += self.encoder(x)
        return self.decoder(x)
