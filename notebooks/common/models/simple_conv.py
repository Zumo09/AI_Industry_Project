import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Optional, Tuple


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv1d:
    """3x3 convolution with padding"""
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        # bias=False,
        dilation=dilation,
        padding_mode="reflect",
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    """1x1 convolution"""
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        # bias=False,
    )


class Encoder(nn.Sequential):
    def __init__(
        self,
        in_feats: int,
        down_layers: int,
        base_width: int = 512,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        assert down_layers > 0, "at least 1 layer"
        norm_layer = norm_layer or nn.BatchNorm1d

        layers = [in_feats] + [2**i * base_width for i in range(down_layers)]

        dilations = replace_stride_with_dilation or [False for _ in layers[:-1]]
        assert len(layers[:-1]) == len(dilations), "specify dilation for each layer"

        modules = []
        _dilation = 1
        self.reduction = 0
        self.out_ch = layers[-1]

        def block(in_f, out_f, stride, dilation):
            return nn.Sequential(
                conv3x3(in_f, out_f, stride, 1, dilation),
                norm_layer(out_f),
                nn.ReLU(inplace=True),
            )

        for in_f, out_f, dilate in zip(layers[:-1], layers[1:], dilations):
            stride = 2
            prev_dilation = _dilation
            if dilate:
                _dilation *= stride
                stride = 1
            else:
                self.reduction += 1
            modules.append(block(in_f, out_f, stride, prev_dilation))
        super().__init__(*modules)


class SimpleSegConv(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        out_channels: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        norm_layer = norm_layer or nn.BatchNorm1d

        layers = [encoder.out_ch // 2**i for i in range(encoder.reduction + 1)]

        self.encoder = encoder
        self.blocks = nn.ModuleList()

        def block(in_f, out_f):
            return nn.Sequential(
                conv3x3(in_f, out_f),
                norm_layer(out_f),
                nn.ReLU(inplace=True),
            )

        for in_f, out_f in zip(layers[:-1], layers[1:]):
            self.blocks.append(block(in_f, out_f))

        self.out_layer = conv1x1(layers[-1], out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = inputs.permute(0, 2, 1)
        x = self.encoder(x)

        for module in self.blocks:
            x = F.interpolate(x, scale_factor=2, mode="linear")
            x = module(x)

        x = self.out_layer(x)
        x = x.permute(0, 2, 1)

        return x


class SimpleConv(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        out_channels: int,
        out_size: int,
    ) -> None:
        super().__init__()
        self.out_size = out_size
        self.encoder = encoder
        self.out_layer = conv1x1(encoder.out_ch, out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = inputs.permute(0, 2, 1)
        x = self.encoder(x)
        x = self.out_layer(x)
        x = F.interpolate(x, size=self.out_size, mode="linear")
        x = x.permute(0, 2, 1)

        return x
