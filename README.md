# TODO: LINK TO REPORT, add rest of results

# LookaheadOptimizer
Reproduction of CIFAR-10/CIFAR-100 and Penn Treebank experiments to test claims in "LookaheadOptimizer: k steps forward, 1 step back" https://arxiv.org/abs/1907.08610

-----------------------------------------------------------------

## Introduction 
Stochastic gradient descent (SGD) is a popular method for training neural networks, using “minibatches” of data to update the network’s weights.  Improvements are often made upon SGD by either using acceleration/momentum or by altering the learning rate over time. Zhang et al. propose a novel improvement in "LookaheadOptimizer: k steps forward, 1 step back". They showed that Lookahead consistantly outperformed other optimizers on popular language modelling, machine translantion and image classification tasks. 

This project aims to test these findings by reimplementing the main CIFAR-10/100 and Penn Treebank experiments. 

See PAPER FOR MORE INFO


## Team Members 

Connah Romano-Scott - crs1u19@soton.ac.uk
John Joyce - jvjj1u19@soton.ac.uk
Ilias Kazantzidis - ik3n19@soton.ac.uk

## Code References 

All experiments used the PyTorch implementation of Lookahead (written by Zhang et al.), available at: https://github.com/michaelrzhang/lookahead

The ResNet-18 implementation in this work (and Zhang et al.'s work) is availble at: https://github.com/uoguelph-mlrg/Cutout/blob/master/model/resnet.py

The Penn Treebank training setup in this work (and Zhang et al.'s work) is modified from: https://github.com/salesforce/awd-lstm-lm
See [PTB README](https://github.com/COMP6248-Reproducability-Challenge/LookaheadOptimizer/blob/master/PTB/README.MD) for more information on modifications of this code. 
