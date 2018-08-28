#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Main.py
"""

import argparse as ap
import multiprocessing as mp

import predict
import evaluation
import depth_map

def get_3d_center():
    import tensorflow as tf
    print(depth_map.get_world_coordinates("002233.png",215,723))


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--mode", help="train or predict", default='predict')
    args = parser.parse_args()
    if "predict" in str(args):
        predict.predict_images()
        # Using a seperate process for depth prediction model
        #p1 = mp.Process(target=get_3d_center)
        #p1.start()
        #p1.join()
        evaluation.evaluate3d_detection()
    elif "train" in str(args):
        import train
        train.train_model()
    else:
        print("Please enter valid argument.")
