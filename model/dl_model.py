import torch
import torchvision
from model.utils import *
import torch.nn as nn
from resnest.torch import resnest50

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

def resnest_model_50(num_classes, frozen=False, pretrained=True):
    model_ft = resnest50(pretrained=pretrained)
    if frozen:
        model_freeze(model_ft)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(
        nn.Linear(in_features=num_ftrs, out_features=num_classes)
    )
    return model_ft

def densenet_161(num_classes, frozen=False, pretrained=True):
    model_ft = torchvision.models.densenet161(pretrained=pretrained)
    if frozen:
        model_freeze(model_ft)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Sequential(
        nn.Linear(in_features=num_ftrs, out_features=num_classes)
    )
    return model_ft