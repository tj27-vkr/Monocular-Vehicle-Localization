#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Main.py
"""

import argparse as ap
import multiprocessing as mp

#import predict
#import evaluation
import depth_map

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--mode", help="train or predict", default='predict')
    args = parser.parse_args()
    if "predict" in str(args):
        #predict.predict_images()
        #evaluation.evaluate3d_detection()
	print("run:\npython predict.py\npython evaluation.py")
	#Need to implement multiple tensorflow session to be run in the
	#same instance
    elif "train" in str(args):
        import train
        train.train_model()
    else:
        print("Please enter a valid argument.")
