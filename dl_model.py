import torch
import torchvision
from utils import *
import torch.nn as nn

def res_model_34(num_classes, frozen=False, pretrained=True):
    model_ft = torchvision.models.resnet34(pretrained=pretrained)
    if frozen:
        model_freeze(model_ft)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(
        nn.Linear(in_features=num_ftrs, out_features=num_classes)
    )
    return model_ft

def resnext_model_50(num_classes, frozen=False, pretrained=True):
    model_ft = torchvision.models.resnext50_32x4d(pretrained=pretrained)
    if frozen:
        model_freeze(model_ft)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(
        nn.Linear(in_features=num_ftrs, out_features=num_classes)
    )
    return model_ft