import torch
import numpy as np
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F

# from .hyvit import *
from .hyp_crossvit import *
from .mobilefacenet import MobileFaceNet
from .ir50 import Backbone
# from .hrnet_face_landmark import *
# from .hrnet_config import config as hrnet_config


def load_pretrained_weights(model, checkpoint):
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    # new_state_dict.requires_grad = False
    model_dict.update(new_state_dict)

    model.load_state_dict(model_dict)
    print('load_weight', len(matched_layers))
    return model


class HRnetHead(nn.Module):
    def __init__(self, input_dim: int, target_dim: int):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(input_dim, 1, 1)
        self.linear = torch.nn.Linear(64 * 64, target_dim)
        self.dropout = nn.Dropout()

    def forward(self, x):
        B_ = x.shape[0]
        x = self.conv1(x)
        x = x.view(B_, -1)
        y_hat = self.linear(x)
        return y_hat


class SE_block(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.linear2 = torch.nn.Linear(input_dim, input_dim)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.relu(x1)
        x1 = self.linear2(x1)
        x1 = self.sigmod(x1)
        x = x * x1
        return x


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, target_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, target_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y_hat = self.linear(x)
        return y_hat


class pyramid_trans_expr(nn.Module):
    def __init__(self, img_size=224, num_classes=7, type="large"):
        super().__init__()
        depth = 8
        if type == "small":
            depth = 4
        if type == "base":
            depth = 6
        if type == "large":
            depth = 8

        self.img_size = img_size
        self.num_classes = num_classes

        self.face_landback = MobileFaceNet([112, 112],136)
        face_landback_checkpoint = torch.load('./models/pretrain/mobilefacenet_model_best.pth.tar', map_location=lambda storage, loc: storage)
        self.face_landback.load_state_dict(face_landback_checkpoint['state_dict'])


        for param in self.face_landback.parameters():
            param.requires_grad = False

        # self.face_layer = nn.Linear(512, 512)
        ###########################################################################333


        self.ir_back = Backbone(50, 0.0, 'ir')
        ir_checkpoint = torch.load('./models/pretrain/ir50.pth', map_location=lambda storage, loc: storage)
        # ir_checkpoint = ir_checkpoint["model"]
        self.ir_back = load_pretrained_weights(self.ir_back, ir_checkpoint)

        self.ir_layer = nn.Linear(1024,512)
        self.face_layer = nn.Linear(512,1024)


        #############################################################3
        ### for swin_large , in chas = 19+

        # self.land_downsample = nn.Conv2d(19, 19, kernel_size=2, stride=2, padding=0, )
        self.pyramid_fuse = HyVisionTransformer(in_chans=49, q_chanel = 49, embed_dim=1024,
                                             depth=depth, num_heads=8, mlp_ratio=2.,
                                             drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1)

        ####  drop_path_rate=0.1     90.84

        # self.drop = torch.nn.Dropout2d(p=0.2)
        self.se_block = SE_block(input_dim=1024)
        self.head = ClassificationHead(input_dim=1024, target_dim=self.num_classes)


    def forward(self, x):
        B_ = x.shape[0]
        x_face = F.interpolate(x, size=112)
        _, x_face = self.face_landback(x_face)
        x_face = x_face.view(B_, -1, 49).transpose(1,2)
        x_face = self.face_layer(x_face)
        ###############  x_hrland ([B, 19, 64, 64])
        # x_hrland = self.land_downsample(x_hrland)


        x_ir = self.ir_back(x)
        ###############  x_ir ([B, 49, 1024])
        # x_ir = self.ir_layer(x_ir)

        # print("x_swin ", x_swin.shape)
        # y_hat = torch.cat((x_face.view(B_,-1,64,64), x_ir.view(B_,-1,64,64)),dim=1)

        # y_hat = self.drop(y_hat)
        y_hat = self.pyramid_fuse(x_ir, x_face)
        y_hat = self.se_block(y_hat)
        # y_feat = nn.functional.normalize(y_hat)
        y_feat = y_hat
        out = self.head(y_hat)
        # out = self.head(y_feat)

        return out, y_feat


