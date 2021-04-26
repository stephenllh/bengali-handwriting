import torch
from torch import nn
from mish_activation import Mish


class Head(nn.Module):
    def __init__(self, nc, n, ps=0.5):
        super().__init__()
        layers = (
            [AdaptiveConcatPool2d(), Mish(), Flatten()]
            + bn_drop_lin(nc * 2, 512, True, ps, Mish())
            + bn_drop_lin(512, n, True, ps)
        )
        self.fc = nn.Sequential(*layers)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(self, x):
        return self.fc(x)


class DensenetOneChannel(nn.Module):
    def __init__(self, arch, n, pretrained=True, ps=0.5):
        super().__init__()
        m = arch(True) if pretrained else arch()

        # change the first conv to accept 1 chanel input
        conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        w = (m.features.conv0.weight.sum(1)).unsqueeze(1)
        conv.weight = nn.Parameter(w)

        self.layer0 = nn.Sequential(conv, m.features.norm0, nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
            ),
            m.features.denseblock1,
        )
        self.layer2 = nn.Sequential(m.features.transition1, m.features.denseblock2)
        self.layer3 = nn.Sequential(m.features.transition2, m.features.denseblock3)
        self.layer4 = nn.Sequential(
            m.features.transition3, m.features.denseblock4, m.features.norm5
        )

        nc = self.layer4[-1].weight.shape[0]
        self.head1 = Head(nc, n[0])
        self.head2 = Head(nc, n[1])
        self.head3 = Head(nc, n[2])

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x1 = self.head1(x)
        x2 = self.head2(x)
        x3 = self.head3(x)

        return x1, x2, x3


class Flatten(nn.Module):
    "Flatten `x` to a single dimension, often used at the end of a model. `full` for rank-1 tensor"

    def __init__(self, full: bool = False):
        self.full = full

    def forward(self, x):
        return x.view(-1) if self.full else x.view(x.size(0), -1)


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."

    def __init__(self, sz: int = None):
        "Output will be 2*sz or 2 if sz is None"
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


def bn_drop_lin(
    n_in: int,
    n_out: int,
    bn: bool = True,
    p: float = 0.0,
    actn: nn.Module = None,
):
    "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0:
        layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None:
        layers.append(actn)
    return layers
