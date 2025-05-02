# Homebrew Transformers

## Overview

This project recreates the model and processes used in the foundational [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper, and applies these techniques to the task of machine translation, using the europarl dataset. Also included is a vision transformer, which uses a transformer encoder on the task of image classification using the [cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

The project uses pytorch to recreate the transformer models from scratch comprehensively from the top-level encoder-decoder model, down to the core dot product attention function, these models are then trained using a custom trainer class. The results of the training are viewable in the ```plots``` folder, and via tensorboard for the base and image classifcation models respectively.

The ```data.py``` contains the code used to both massage, prune, and parse the data to be used in the model training.

## Instructions

### Custom Model

Preset model parameters are available in ```modelDict.py```. A user can simply create a new entry in the dictionary with the parameters of their choosing, then train their desired model by setting the ```modelName``` variable in ```train.py```.
