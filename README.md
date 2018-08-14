# Vehicle-Pose-Estimation-Monocular-Vision (In Progress)

## Introduction

  Estimating the orientation and the size of vehicle using a single 2d image. The output is a 3d bounding box around the detected vehicle.
    
Prerequisites
  + Tensorflow
  + Keras
  + Numpy
  + OpenCV2
  + [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
 
## Usage

Edit the necessary configurations and paths in `config.py`

```
#export current working directory
export PYTHONPATH='.'

#train the model
python main.py --mode=train

#evaluate example images
python main.py --mode=predict
```

## Ouput

The below images are the ouput from the partially trained model on 8 vCPUs, 52 GB memory machine (no GPU). The output will be more precise after complete training (currently trained only for 1 out of 500 Epochs).

   <img src="./example_data/output_predi/001012.png" width = "700" height = "250" align=center />

   <img src="./example_data/output_predi/001016.png" width = "700" height = "250" align=center />

## References

+ "3D Bounding Box Estimation Using Deep Learning and Geometry" [link to paper](https://arxiv.org/abs/1612.00496)
+ [experiencor/image-to-3d-bbox](https://github.com/experiencor/image-to-3d-bbox)
