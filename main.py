#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Main.py
"""

import argparse as ap

import predict
import train
import evaluation

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--mode", help="train or predict", default='predict')
    args = parser.parse_args()
    if "predict" in str(args):
        predict.predict_images()
	evaluation.evaluate3d_detection()
    elif "train" in str(args):
        train.train_model()
    else:
        print("Please enter valid argument.")
