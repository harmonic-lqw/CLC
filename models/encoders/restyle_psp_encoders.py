import torch
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential, Module
from torchvision.models.resnet import resnet34

from models.encoders.helpers import get_blocks, bottleneck_IR, bottleneck_IR_SE
from models.encoders.map2style import GradualStyleBlock


class BackboneEncoder(Module):
    """
    The simpler backbone architecture used by ReStyle where all style vectors are extracted from the final 16x16 feature
    map of the encoder. This classes uses the simplified architecture applied over an ResNet IRSE-50 backbone.
    Note this class is designed to be used for the human facial domain.
    """
    def __init__(self, num_layers, mode='ir', n_styles=18, opts=None):
        super(BackboneEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE

        # Hin = Hout
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = n_styles
        for i in range(self.style_count):
            style = GradualStyleBlock(512, 512, 16)
            self.styles.append(style)

    def forward(self, x, layers=[], encode_only=False):
        if len(layers) > 0:
            feat = x
            feats = []
            # (N, 3, 256, 256)
            feats.append(feat)
            # (N, 64, 256, 256)
            feat = self.input_layer(feat)
            # feats.append(feat)

            modulelist = list(self.body._modules.values())
            for i, l in enumerate(modulelist):
                feat = l(feat)
                if i in layers:   # (第一个）0/(N, 64, 128, 128)  (第一个）3/(N, 128, 64, 64)  7/(N, 256, 32, 32)  20/(N, 256, 32, 32)   21/(N, 512, 16, 16)  
                    feats.append(feat)
                if i == layers[-1] and encode_only:
                    return feats
                    
        x = self.input_layer(x)
        x = self.body(x)
        latents = []
        for j in range(self.style_count):
            latents.append(self.styles[j](x))
        out = torch.stack(latents, dim=1)
        return out


class ResNetBackboneEncoder(Module):
    """
    The simpler backbone architecture used by ReStyle where all style vectors are extracted from the final 16x16 feature
    map of the encoder. This classes uses the simplified architecture applied over an ResNet34 backbone.
    """
    def __init__(self, n_styles=18, opts=None):
        super(ResNetBackboneEncoder, self).__init__()

        self.conv1 = nn.Conv2d(opts.input_nc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = PReLU(64)

        resnet_basenet = resnet34(pretrained=True)
        blocks = [
            resnet_basenet.layer1,
            resnet_basenet.layer2,
            resnet_basenet.layer3,
            resnet_basenet.layer4
        ]
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck)
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = n_styles
        for i in range(self.style_count):
            style = GradualStyleBlock(512, 512, 16)
            self.styles.append(style)

    def forward(self, x, layers=[], encode_only=False):
        if len(layers) > 0:
            feat = x
            feats = []
            feats.append(feat)
            feat = self.conv1(feat)
            feat = self.bn1(feat)
            feat = self.relu(feat)
            feats.append(feat)

            modulelist = list(self.body._modules.values())
            for i, l in enumerate(modulelist):
                feat = l(feat)
                if i in layers:
                    feats.append(feat)
                if i == layers[-1] and encode_only:
                    return feats

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.body(x)
        latents = []
        for j in range(self.style_count):
            latents.append(self.styles[j](x))
        out = torch.stack(latents, dim=1)
        return out
