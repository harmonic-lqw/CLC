import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import init_net

# (3, mlp_sample, instance, False, xavier, 0.02, 使用2stride卷积, 0, opt)
def define_F(input_nc, netF, norm='batch', use_dropout=False, init_type='normal',
             init_gain=0.02, no_antialias=False, gpu_ids=[], opt=None):
    if netF == 'global_pool':
        net = PoolingF()
    elif netF == 'reshape':
        net = ReshapeF()
    elif netF == 'sample':
        net = PatchSampleF(use_mlp=False, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc)
    elif netF == 'mlp_sample':
        net = PatchSampleF(use_mlp=True, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc)
    elif netF == 'strided_conv':
        net = StridedConvF(init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('projection model name [%s] is not recognized' % netF)
    return init_net(net, init_type, init_gain, gpu_ids)


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


class PoolingF(nn.Module):
    def __init__(self):
        super(PoolingF, self).__init__()
        model = [nn.AdaptiveMaxPool2d(1)]
        self.model = nn.Sequential(*model)
        self.l2norm = Normalize(2)

    def forward(self, x):
        return self.l2norm(self.model(x))


class ReshapeF(nn.Module):
    def __init__(self):
        super(ReshapeF, self).__init__()
        model = [nn.AdaptiveAvgPool2d(4)]
        self.model = nn.Sequential(*model)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = self.model(x)
        x_reshape = x.permute(0, 2, 3, 1).flatten(0, 2)
        return self.l2norm(x_reshape)


class StridedConvF(nn.Module):
    def __init__(self, init_type='normal', init_gain=0.02, gpu_ids=[]):
        super().__init__()
        # self.conv1 = nn.Conv2d(256, 128, 3, stride=2)
        # self.conv2 = nn.Conv2d(128, 64, 3, stride=1)
        self.l2_norm = Normalize(2)
        self.mlps = {}
        self.moving_averages = {}
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, x):
        C, H = x.shape[1], x.shape[2]
        n_down = int(np.rint(np.log2(H / 32)))
        mlp = []
        for i in range(n_down):
            mlp.append(nn.Conv2d(C, max(C // 2, 64), 3, stride=2))
            mlp.append(nn.ReLU())
            C = max(C // 2, 64)
        mlp.append(nn.Conv2d(C, 64, 3))
        mlp = nn.Sequential(*mlp)
        init_net(mlp, self.init_type, self.init_gain, self.gpu_ids)
        return mlp

    def update_moving_average(self, key, x):
        if key not in self.moving_averages:
            self.moving_averages[key] = x.detach()

        self.moving_averages[key] = self.moving_averages[key] * 0.999 + x.detach() * 0.001

    def forward(self, x, use_instance_norm=False):
        C, H = x.shape[1], x.shape[2]
        key = '%d_%d' % (C, H)
        if key not in self.mlps:
            self.mlps[key] = self.create_mlp(x)
            self.add_module("child_%s" % key, self.mlps[key])
        mlp = self.mlps[key]
        x = mlp(x)
        self.update_moving_average(key, x)
        x = x - self.moving_averages[key]
        if use_instance_norm:
            x = F.instance_norm(x)
        return self.l2_norm(x)

# PatchSampleF(use_mlp=True, init_type=xavier, init_gain=0.02, gpu_ids=0, nc=256)
class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feats):
        # mlp_id:因为对于每一层提取出来的feat，都要有一个mlp
        for mlp_id, feat in enumerate(feats):
            # 取出输入通道
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        # 初始化网络权重
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    # num_patches = 256
    def forward(self, feats, num_patches=64, patch_ids=None, return_all=False):
        return_ids = []
        return_feats = []
        if return_all:
            return_feats_all = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            # (N, C, H, W) -> (N, H*W, C)
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    # 取出对应层的patch_id(row.168)
                    patch_id = patch_ids[feat_id]
                else:
                    # torch.randperm(n):将0~n-1（包括0和n-1）随机打乱后获得的数字序列
                    # 随机生成打乱得H*W个整数，范围0 ~ H*W-1

                    # print("*"*30)
                    # print('feat_q[0].device: ', feats[0].device)
                    # print('feat_reshape.shape[1]: ', feat_reshape.shape[1])
                    # print('feat_reshape.size(): ', feat_reshape.size())
                    # print('type(feat_reshape): ', type(feat_reshape))
                    # print("*"*30)

                    # patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = torch.randperm(feat_reshape.shape[1]).to(feats[0].device)
                    # 取前min(256, H*W)个
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)

                if return_all:
                    x_sample = feat_reshape.flatten(0, 1)
                else:
                    # sample方法：取出patch_id个像素
                    # (N, H*W, C) -> (N*256, C) 每一个N有256随机sample的像素(附带通道c)
                    x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                # num_patches == 0: 每张样本的全部像素送入H
                x_sample = feat_reshape.flatten(0, 1)
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                # (N*256, C) -> (N*256, 256)
                x_sample = mlp(x_sample)
            # 每个不同大小的特征图patch_id不同
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if return_all:
                x_sample = x_sample.view(B, -1, self.nc)
                x_sample_all = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
                x_sample = x_sample[:, patch_id, :].flatten(0, 1)
                return_feats.append(x_sample)
                return_feats_all.append(x_sample_all)
            else:
                if num_patches == 0:
                    # contiguous: https://www.jianshu.com/p/51678ea7a959
                    # (N*H*W, 256) -> (N, H*W, 256)
                    x_sample = x_sample.reshape([B, -1, x_sample.shape[-1]]).contiguous()
                    # (N, H*W, 256) -> (N, 256, H*W) -> (N, 256, H, W)
                    x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
                return_feats.append(x_sample)
        if return_all:
            return return_feats, return_feats_all, return_ids
        else:
            # num_patches == 0: [5, N, 256, H, W]
            # [5, N*256, 256], [5, 256]
            return return_feats, return_ids
