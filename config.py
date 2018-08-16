#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This Module contains the configurations
   and macros for the project.
"""

import os

#set cuda environment variables
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#deep network's parameters
BIN, OVERLAP = 2, 0.1
W = 1.
ALPHA = 1.
MAX_JIT = 3
BATCH_SIZE = 8

#dataset parameters
VEHICLE_CLASSES = ['Car', 'Truck', 'Van', 'Tram']
NORM_H, NORM_W = 224, 224

#training data directories
label_dir = '/home/vkvigneshram/disk/monodepth_wb/custom/dataset/labels/training/label_2/'
image_dir = '/home/vkvigneshram/disk/monodepth_wb/custom/dataset/images/training/image_2/'

#example images and label directories
ex_image_dir = '/home/vkvigneshram/disk/monodepth_wb/custom/example_data/images/'
ex_label_dir = '/home/vkvigneshram/disk/monodepth_wb/custom/example_data/labels/'


#predicted 3d box output file directory
detection3d_dir = '/home/vkvigneshram/disk/monodepth_wb/custom/example_data/output3d/'
output_dir = '/home/vkvigneshram/disk/monodepth_wb/custom/example_data/output_predi/'

detection2d_dir = ex_label_dir

#calibration data directory
calib_dir = '/home/vkvigneshram/disk/monodepth_wb/custom/dataset/calib/training/calib/'
