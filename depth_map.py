# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import matplotlib.pyplot as plt

from monodepth.monodepth_model import *
from monodepth.monodepth_dataloader import *
from monodepth.average_gradients import *


ENCODER = 'vgg'
IMAGE_PATH = '002233.png'
CHECKPOINT_PATH = 'model/model_kitti'
DEFAULT_HEIGHT = 256
DEFAULT_WIDTH = 512

def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def test_simple(params):
    import tensorflow as tf
    """Test function."""

    left  = tf.placeholder(tf.float32, [2, DEFAULT_HEIGHT, DEFAULT_WIDTH, 3])

    input_image = scipy.misc.imread(IMAGE_PATH, mode="RGB")
    original_height, original_width, num_channels = input_image.shape
    
    
    model = MonodepthModel(params, "test", left, None, reuse_variables=tf.AUTO_REUSE)
    input_image = scipy.misc.imresize(input_image, [DEFAULT_HEIGHT, DEFAULT_WIDTH], interp='lanczos')
    input_image = input_image.astype(np.float32) / 255
    input_images = np.stack((input_image, np.fliplr(input_image)), 0)

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    restore_path = CHECKPOINT_PATH.split(".")[0]
    train_saver.restore(sess, restore_path)

    disp = sess.run(model.disp_left_est[0], feed_dict={left: input_images})
    disp_pp = post_process_disparity(disp.squeeze()).astype(np.float32)

    output_directory = os.path.dirname(IMAGE_PATH)
    output_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]

    np.save(os.path.join(output_directory, "{}_disp.npy".format(output_name)), disp_pp)
    disp_to_img = scipy.misc.imresize(disp_pp.squeeze(), [original_height, original_width])
    plt.imsave(os.path.join(output_directory, "{}_disp.png".format(output_name)), disp_to_img, cmap='plasma')
    sess.close()
    return disp_pp

def get_world_coordinates(image_path, x_px, y_px):
    params = monodepth_parameters(
        encoder=ENCODER,
        height=DEFAULT_HEIGHT,
        width=DEFAULT_WIDTH,
        batch_size=2,
        num_threads=1,
        num_epochs=1,
        do_stereo=False,
        wrap_mode="border",
        use_deconv=False,
        alpha_image_loss=0,
        disp_gradient_loss_weight=0,
        lr_loss_weight=0,
        full_summary=False)

    disparity_matrix = test_simple(params)

    import cv2
    original_image = cv2.imread(image_path)
    original_height, original_width, num_channels = original_image.shape
    mod_disp = original_width * cv2.resize(disparity_matrix, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

    #depth = focal_length * baseline / disparity
    mod_depth = 721.5377 * 0.54 / mod_disp

    #x,y in world coordinates = xy_pixel * depth / focal_length
    x_real_world = (x_px - (original_height/2.)) * mod_depth[int(x_px)][int(y_px)] / 721.5377
    y_real_world = (y_px - (original_width/2.)) * mod_depth[int(x_px)][int(y_px)] / 721.5377

    return x_real_world, y_real_world, mod_depth[int(x_px)][int(y_px)]


if __name__ == '__main__':
    tx = 215
    ty = 723
