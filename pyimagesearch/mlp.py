#!/usr/bin/env python3
"""
-*- coding:utf-8 -*-
demo : mlp.py
@author : Jedrek.LiuYX
Date : 2023/2/18 16:37
        Build A simple neural network.
"""
# import the necessary packages
from collections import OrderedDict
import torch.nn as nn


# define the model function
def get_training_model(inFeatures=4, hiddenDim=8, nbClasses=3):
    # construct a shallow, sequential neural network
    mlpModel = nn.Sequential(OrderedDict([
        ("hidden_layer_1", nn.Linear(inFeatures, hiddenDim)),
        ("activation_1", nn.ReLU()),
        ("output_layer", nn.Linear(hiddenDim, nbClasses))
    ]))

    # return the sequential model
    return mlpModel
