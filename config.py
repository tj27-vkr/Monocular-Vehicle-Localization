import os 

#set cuda environment variables
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#deep network's parameters
BIN, OVERLAP = 2, 0.1
W = 1.
ALPHA = 1.
MAX_JIT = 3
NORM_H, NORM_W = 224, 224
VEHICLES = ['Car', 'Truck', 'Van', 'Tram']
BATCH_SIZE = 8

#training data directories
label_dir = '/home/vkvigneshram/disk/monodepth_wb/custom/dataset/labels/training/label_2/'
image_dir = '/home/vkvigneshram/disk/monodepth_wb/custom/dataset/images/training/image_2/'

#example images and label directories
ex_image_dir = '/home/vkvigneshram/disk/monodepth_wb/custom/example_data/images/'
ex_label_dir = '/home/vkvigneshram/disk/monodepth_wb/custom/example_data/labels/'
box2d_loc = ex_label_dir

#predicted 3d box output file directory
box3d_loc = '/home/vkvigneshram/disk/monodepth_wb/custom/example_data/output3d/'
predi_dir = '/home/vkvigneshram/disk/monodepth_wb/custom/example_data/output_predi/'

box2d_dir = box2d_loc
box3d_dir = box3d_loc

#calibration data directory
calib_dir = '/home/vkvigneshram/disk/monodepth_wb/custom/dataset/calib/training/calib/'
