#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module to draw 3d bounding boxes from the prediction
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from config import *


def evaluate3d_detection():
	all_image = sorted(os.listdir(ex_image_dir))

	for f in all_image:
	    image_file = ex_image_dir + f
	    box3d_file = detection3d_dir + f.replace('png', 'txt')
	    label_file = ex_label_dir + f.replace('png', 'txt')
	    calib_file = calib_dir + f.replace('png', 'txt')
	    output_file = output_dir + f.replace('png', 'txt')
	    
	    with open(output_file, 'w') as prediction:
		# Construct list of all candidate centers
		centers_2d = []
		centers_3d = []

		for line in open(calib_file):
		    if 'P2:' in line:
			cam_to_img = line.strip().split(' ')
			cam_to_img = np.asarray([float(number) for number in cam_to_img[1:]])
			cam_to_img = np.reshape(cam_to_img, (3,4))

		for line in open(label_file):
		    line = line.strip().split(' ')

		    center = np.asarray([float(number) for number in line[11:14]])
		    center = np.append(center, 1)
#		    print("################ camtoimg : {}, center: {}".format(cam_to_img, center))
		    center = np.dot(cam_to_img, center)
#		    print("center from the file: {}".format(center))
		    center = center[:2]/center[2]
		    x_c  = (float(line[4]) + float(line[6]))/2.
		    y_c  = (float(line[5]) + float(line[7]))/2.
#		    print("## {} \n## {}\n\n".format((x_c, y_c),center[:2]))
		    center = center.astype(np.int16)

		    centers_2d.append(center)
		    centers_3d.append(np.asarray([float(number) for number in line[11:14]]))

		# Find the nearest centres among the candidates
		for line in open(box3d_file):
		    line = line.strip().split(' ')

		    obj = {'xmin':int(float(line[4])),
			   'ymin':int(float(line[5])),
			   'xmax':int(float(line[6])),
			   'ymax':int(float(line[7])),}

		    center = np.asarray([(obj['xmin']+obj['xmax'])/2., (obj['ymin'] + obj['ymax'])/2.])

		    nearest_index = -1
		    last_distance = 1000000000.
		    #print ("the center for 2d is {}".format(centers_2d))

		    for i in xrange(len(centers_2d)):
			candidate = centers_2d[i]
			distance = np.sum(np.square(center - candidate))
			#print("the distance is {} and last is {}".format(distance, last_distance))
			if distance < 2000 and distance < last_distance:
			    #print ("nearest index hit")
			    nearest_index = i
			    last_distance = distance

		    if nearest_index > -1:
			line += list(centers_3d[nearest_index])
			del centers_2d[nearest_index]
			del centers_3d[nearest_index]
			
			# Write regressed 3D dim and oritent to file
			line = ' '.join([str(item) for item in line]) + '\n'
			print("prediction writen.")
			prediction.write(line)


	print("Done !! ")


	all_image = sorted(os.listdir(ex_image_dir))

	for f in all_image:
	    image_file = ex_image_dir + f
	    calib_file = calib_dir + f.replace('png', 'txt')
	    output_file = output_dir + f.replace('png', 'txt')

	    # read calibration data
	    for line in open(calib_file):
		if 'P2:' in line:
		    cam_to_img = line.strip().split(' ')
		    cam_to_img = np.asarray([float(number) for number in cam_to_img[1:]])
		    cam_to_img = np.reshape(cam_to_img, (3,4))
		
	    image = cv2.imread(image_file)
	    cars = []
	    
	    # Draw 3D Bounding Box
	    for line in open(output_file):
		line = line.strip().split(' ')

		dims   = np.asarray([float(number) for number in line[8:11]])
		center = np.asarray([float(number) for number in line[11:14]])
		rot_y  = float(line[3]) + np.arctan(center[0]/center[2])#float(line[14])

		box_3d = []

		for i in [1,-1]:
		    for j in [1,-1]:
			for k in [0,1]:
			    point = np.copy(center)
			    point[0] = center[0] + i * dims[1]/2 * np.cos(-rot_y+np.pi/2) + (j*i) * dims[2]/2 * np.cos(-rot_y)
			    point[2] = center[2] + i * dims[1]/2 * np.sin(-rot_y+np.pi/2) + (j*i) * dims[2]/2 * np.sin(-rot_y)                  
			    point[1] = center[1] - k * dims[0]

			    point = np.append(point, 1)
			    point = np.dot(cam_to_img, point)
			    point = point[:2]/point[2]
			    point = point.astype(np.int16)
			    box_3d.append(point)

		for i in xrange(4):
		    point_1_ = box_3d[2*i]
		    point_2_ = box_3d[2*i+1]
		    cv2.line(image, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (0,255,0), 2)

		for i in xrange(8):
		    point_1_ = box_3d[i]
		    point_2_ = box_3d[(i+2)%8]
		    cv2.line(image, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (0,255,0), 2)
			
	    #video_writer.write(np.uint8(image))
	    cv2.imwrite("example_data/output_predi/{}".format(f),image)
	    print ("{} generated.".format(f))

	print("Done.")

if __name__ == "__main__":
    evaluate3d_detection()
