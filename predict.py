#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Process the image and predict output
"""

import cv2, os
import numpy as np
import tensorflow as tf

from config import *
import dn_model
import depth_map
import ssd_detection
#from depth_map import *

dims_avg = AVERAGE_DIMENSIONS

def get_camera_param(calib_file, cam_id):
    calib_data = {}
    with open(calib_file, 'r') as fp:
        for line in fp.readlines():
            key, value = line.split(":", 1)
            value = value.strip()
            try:
                calib_data[key] = np.array(map(float, value.split(' ')))
            except:
                print("Skipping calib data key:{} value:{}".format(key, value))
                pass

    P2_rect = calib_data['P_rect_02'].reshape(3,4)
    P3_rect = calib_data['P_rect_03'].reshape(3,4)

    b2 = P2_rect[0,3] / -P2_rect[0,0]
    b3 = P3_rect[0,3] / -P3_rect[0,0]
    baseline = b3-b2
    
    if cam_id == 2:
        focal_length = P2_rect[0,0]
    elif cam_id == 3:
        focal_length = P3_rect[0,0]

    return baseline, focal_length


def predict_images():

    model_o = dn_model.network_arch()
    print("Loading weights...")
    model_o.load_weights('model/weights.hdf5')
    print("Done...")

    all_image = sorted(os.listdir(ex_image_dir))

    for f in all_image:
        image_file = ex_image_dir + f
        box2d_file = detection2d_dir + f.replace('png', 'txt')
        box3d_file = detection3d_dir + f.replace('png', 'txt')
        
        with open(box3d_file, 'w') as box3d:
            img = cv2.imread(image_file)
            img = img.astype(np.float32, copy=False)

	    ssd_boxes_2d = ssd_detection.get_2d_box(image_file) 

            for ssd_box in ssd_boxes_2d:
	        line = []
		line.append(0)
		line.append(0)
		line.append(0)
                
		obj = {'xmin':int(float(ssd_box[0])),
                       'ymin':int(float(ssd_box[1])),
                       'xmax':int(float(ssd_box[2])),
                       'ymax':int(float(ssd_box[3])),
                      }
        

                cneter_2d = np.asarray([(obj['xmin']+obj['xmax'])/2., (obj['ymin'] + obj['ymax'])/2.])


                #cropping the image based on the 2d prediction
                patch = img[obj['ymin']:obj['ymax'],obj['xmin']:obj['xmax']]
                patch = cv2.resize(patch, (NORM_H, NORM_W))

                #normalizing the dataset by subtracting the mean pixel value
                patch = patch - np.array([[[103.939, 116.779, 123.68]]])
                patch = np.expand_dims(patch, 0)
                
                #run the model for 3d prediction
                
                prediction = model_o.predict(patch)

                # Transform regressed angle
                max_anc = np.argmax(prediction[2][0])
                anchors = prediction[1][0][max_anc]

                if anchors[1] > 0:
                    angle_offset = np.arccos(anchors[0])
                else:
                    angle_offset = -np.arccos(anchors[0])

                wedge = 2.*np.pi/BIN
                angle_offset = angle_offset + max_anc*wedge
                angle_offset = angle_offset % (2.*np.pi)

                angle_offset = angle_offset - np.pi/2
                if angle_offset > np.pi:
                    angle_offset = angle_offset - (2.*np.pi)

                line.append(str(angle_offset))
                for t_i in ssd_box:
		    line.append(t_i)


                # Transform regressed dimension
                dims = dims_avg['Car'] + prediction[0][0]

                line = line + list(dims)

                # Write regressed 3D dim and oritent to file
                line = ' '.join([str(item) for item in line]) + '\n'
                box3d.write(line)

            cv2.rectangle(img, (obj['xmin'],obj['ymin']), (obj['xmax'],obj['ymax']), (255,0,0), 3)
            cv2.imwrite("example_data/output3d/{}".format(f),img)
        
        print("Output generated for image {}".format(f))


if __name__ == "__main__":
    predict_images()
