# Proj_kagg

# Pytorch Image Models (timm)

`timm` is a deep-learning library created by Ross Wightman and is a collection of SOTA computer vision models, layers, utilities, optimizers, schedulers, data-loaders, augmentations and also training/validating scripts with ability to reproduce ImageNet training results.

> Create a model

import timm 

import torch

model = timm.create_model('resnet34')

x     = torch.randn(1, 3, 224, 224)

model(x).shape

It is that simple to create a model using timm. The `create_model` function is a factory method that can be used to create over 300 models that are part of the timm library.

>To create a pretrained model, simply pass in `pretrained=True`.

ie    ---->   pretrained_resnet_34 = timm.create_model('resnet34', pretrained=True)

>To create a model with a custom number of classes, simply pass in `num_classes=<number_of_classes>`.

>List Models with Pretrained Weights

`timm.list_models()` returns a complete list of available models in timm. To have a look at a complete list of pretrained models, pass in `pretrained=True` in list_models.

There are a total of 271 models with pretrained weights currently available in timm!

`avail_pretrained_models = timm.list_models(pretrained=True)`

>Fine-tune timm model in fastai

The fastai library has support for fine-tuning models from timm: `from fastai.vision.all import *`
