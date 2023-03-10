#!/usr/bin/env python3
"""
-*- coding:utf-8 -*-
demo : hubconf.py
@author : Jedrek.LiuYX
Date : 2023/2/18 16:26

"""
# import the necessary packages
import torch
from pyimagesearch import mlp


# define entry point/callable function  to initialize and return model
def custom_model():
    """
    This docstring shows up in hub.help()
    Initializes the MLP model instance
    Loads weights from path and returns the model
    """
    # initialize the model, load weights from path and returns model
    model = mlp.get_training_model()
    model.load_state_dict(torch.load("output/model_wt.pth"))
    return model
