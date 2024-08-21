import jittor.models as models
import jittor as jt
from jittor import nn
import os
from jittor.models import resnet101
from jittor.models.resnet import *
from jittor.models.densenet import *

def load_moco_resnext():
    print("=> creating model")
    model = Resnext101_32x8d(pretrained=True)

    #  计算模型参数量
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    total_params_M = total_params / (1000 * 1000)
    print("aux模型参数量为：", str(total_params_M), "M")
    model.fc = nn.Identity()
    return model, 2048


def load_moco_resnet():
    print("=> creating model")
    model = resnet101(pretrained=True)

    #  计算模型参数量
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    total_params_M = total_params / (1000 * 1000)
    print("aux模型参数量为：", str(total_params_M), "M")
    model.fc = nn.Identity()
    return model, 2048





