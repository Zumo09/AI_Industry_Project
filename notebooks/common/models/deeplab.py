import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .resnet import ResNetFeatures


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__(
            nn.Conv1d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="linear")


class ASPP(nn.Module):
    def __init__(
        self, in_channels: int, atrous_rates: List[int], out_channels: int = 256
    ) -> None:
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            )
        )

        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv1d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))

        res = torch.cat(_res, dim=1)
        return self.project(res)


class DeepLabHeadV3Plus(nn.Module):
    def __init__(
        self,
        low_level_channels: int,
        out_channels: int,
        num_classes: int,
        low_level_name: str,
        out_name: str,
        aspp_dilate=[12, 24, 36],
    ):
        super().__init__()
        llc_prj = low_level_channels // 2
        self.project = nn.Sequential(
            nn.Conv1d(low_level_channels, llc_prj, 1, bias=False),
            nn.BatchNorm1d(llc_prj),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(out_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv1d(llc_prj + 256, 256, 3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, num_classes, 1),
        )

        self.low_level_name = low_level_name
        self.out_name = out_name

    def forward(self, features):
        low_level_feature = self.project(features[self.low_level_name])
        # print("low", features[self.low_level_name].size())
        # print("prj", low_level_feature.size())
        output_feature = self.aspp(features[self.out_name])
        # print("high", features[self.out_name].size())
        # print("feat", output_feature.size())
        output_feature = F.interpolate(
            output_feature,
            size=low_level_feature.shape[2:],
            mode="linear",
            align_corners=False,
        )
        # print("up1", output_feature.size())
        cat = torch.cat([low_level_feature, output_feature], dim=1)
        # print("cat", cat.size())
        outs = self.classifier(cat)
        # print("outs", outs.size())
        return outs


class DeepLabNet(nn.Module):
    def __init__(
        self,
        backbone: ResNetFeatures,
        backbone_channels: List[int],
        out_feats: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = DeepLabHeadV3Plus(
            backbone_channels[0],
            backbone_channels[1],
            out_feats,
            backbone.nodes[0],
            backbone.nodes[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        input_shape = x.shape[2:]

        features = self.backbone(x)

        out: torch.Tensor = self.head(features)
        out = F.interpolate(out, size=input_shape, mode="linear")

        return out.permute(0, 2, 1)
