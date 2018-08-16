#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Process the image and predict output
"""

import cv2, os
import numpy as np

from config import *
import dn_model

def labels_parse(label_dir, image_dir):
    all_objs = []
    dims_avg = {key:np.array([0, 0, 0]) for key in VEHICLE_CLASSES}
    dims_cnt = {key:0 for key in VEHICLE_CLASSES}
        
    for label_file in os.listdir(label_dir):
        image_file = label_file.replace('txt', 'png')

        for line in open(label_dir + label_file).readlines():
            line = line.strip().split(' ')
            truncated = np.abs(float(line[1]))
            occluded  = np.abs(float(line[2]))

            if line[0] in VEHICLE_CLASSES and truncated < 0.1 and occluded < 0.1:
                new_alpha = float(line[3]) + np.pi/2.
                if new_alpha < 0:
                    new_alpha = new_alpha + 2.*np.pi
                new_alpha = new_alpha - int(new_alpha/(2.*np.pi))*(2.*np.pi)

                obj = {'name':line[0],
                       'image':image_file,
                       'xmin':int(float(line[4])),
                       'ymin':int(float(line[5])),
                       'xmax':int(float(line[6])),
                       'ymax':int(float(line[7])),
                       'dims':np.array([float(number) for number in line[8:11]]),
                       'new_alpha': new_alpha
                      }
                
                dims_avg[obj['name']]  = dims_cnt[obj['name']]*dims_avg[obj['name']] + obj['dims']
                dims_cnt[obj['name']] += 1
                dims_avg[obj['name']] /= dims_cnt[obj['name']]

                all_objs.append(obj)
            
    return all_objs, dims_avg


all_objs, dims_avg = labels_parse(label_dir, image_dir)





def predict_images():

	model = dn_model.network_arch()
	print("Loading weights...")
	model.load_weights('model/weights.hdf5')
	print("Done...")

	all_image = sorted(os.listdir(ex_image_dir))

	for f in all_image:
	    image_file = ex_image_dir + f
	    box2d_file = detection2d_dir + f.replace('png', 'txt')
	    box3d_file = detection3d_dir + f.replace('png', 'txt')
	    
	    with open(box3d_file, 'w') as box3d:
	        img = cv2.imread(image_file)
	        img = img.astype(np.float32, copy=False)

	        for line in open(box2d_file):
		    line = line.strip().split(' ')
		    truncated = np.abs(float(line[1]))
		    occluded  = np.abs(float(line[2]))

		    obj = {'xmin':int(float(line[4])),
			   'ymin':int(float(line[5])),
			   'xmax':int(float(line[6])),
			   'ymax':int(float(line[7])),
			  }

		    #cropping the image based on the 2d prediction
		    patch = img[obj['ymin']:obj['ymax'],obj['xmin']:obj['xmax']]
		    patch = cv2.resize(patch, (NORM_H, NORM_W))

		    #normalizing the dataset by subtracting the mean pixel value
		    patch = patch - np.array([[[103.939, 116.779, 123.68]]])
		    patch = np.expand_dims(patch, 0)
		    
		    #run the model for 3d prediction
		    prediction = model.predict(patch)
		    #print("The prediction: {}".format(prediction))

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

		    line[3] = str(angle_offset)

		    # Transform regressed dimension
		    dims = dims_avg['Car'] + prediction[0][0]

		    line = line + list(dims)
		    #print("$$$$${}".format(line))

		    # Write regressed 3D dim and oritent to file
		    line = ' '.join([str(item) for item in line]) + '\n'
		    box3d.write(line)

		    cv2.rectangle(img, (obj['xmin'],obj['ymin']), (obj['xmax'],obj['ymax']), (255,0,0), 3)
		    cv2.imwrite("example_data/output3d/{}".format(f),img)
	    
	    print("Output generated for image {}".format(f))



if __name__ == "__main__":
    predict_images()
