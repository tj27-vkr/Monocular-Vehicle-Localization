# Vehicle-Pose-Estimation-Monocular-Vision (In Progress)

Introduction

  Estimating the orientation and the size of vehicle using a single 2d image. The output is a 3d bounding box around the detected vehicle.
    
Prerequisites
  + Tensorflow
  + Keras
  + Numpy
  + OpenCV2
  + [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
 
Usage
```
#export current working directory
export PYTHONPATH='.'
python train.py --model weights.hdf5
python predict.py --image image1.png
```
  
