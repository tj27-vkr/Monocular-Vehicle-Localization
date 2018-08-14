import argparse as ap

import predict
import train

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--mode", help="train or predict")
    args = parser.parse_args()
    if "predict" in str(args):
        predict.predict_images()
    elif "train" in str(args):
        train.train_model()
    else:
        print("Please enter valid argument.")
