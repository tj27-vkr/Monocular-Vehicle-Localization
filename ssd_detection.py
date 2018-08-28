#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tensorflow object detection model.
   Pretrained SSD COCO mobilenet detector is used.
"""

import numpy as np
import os
import cv2

import tensorflow as tf

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

CHECKPNT_PATH = "./model/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb"
LABELS_PATH = "../../models/research/object_detection/data/mscoco_label_map.pbtxt"


label_map = label_map_util.load_labelmap(LABELS_PATH)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def get_2d_box(image_path):
	image_np = cv2.imread('000011.png')
	im_w, im_h, _ = image_np.shape

	detection_graph = tf.Graph()
	with detection_graph.as_default():
	  od_graph_def = tf.GraphDef()
	  with tf.gfile.GFile(CHECKPNT_PATH, 'rb') as fid:
	    serialized_graph = fid.read()
	    od_graph_def.ParseFromString(serialized_graph)
	    tf.import_graph_def(od_graph_def, name='')

	with detection_graph.as_default():
	  with tf.Session(graph=detection_graph) as sess:
		image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
		# Each box represents a part of the image where a particular object was detected.
		detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
		# Each score represent how level of confidence for each of the objects.
		# Score is shown on the result image, together with the class label.
		detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
		detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
		num_detections = detection_graph.get_tensor_by_name('num_detections:0')
	       
		image_np_expanded = np.expand_dims(image_np, axis=0)
		(boxes, scores, classes, num) = sess.run(
		      [detection_boxes, detection_scores, detection_classes, num_detections],
		      feed_dict={image_tensor: image_np_expanded})

		vis_util.visualize_boxes_and_labels_on_image_array(
		      image_np,
		      np.squeeze(boxes),
		      np.squeeze(classes).astype(np.int32),
		      np.squeeze(scores),
		      category_index,
		      use_normalized_coordinates=True,
		      line_thickness=8)
		cv2.imwrite("2d_out.png",image_np)
		#print("IMage written.")
		out_box = []
		for i, iter in enumerate(classes[0]):
		    if iter == 3:
			#print("Is a Car. Score: {}".format(scores[0][i]))
			#print("Box: {}".format(boxes[0][i]))
			ymin_2d = boxes[0][i][0] * im_w
			xmin_2d = boxes[0][i][1] * im_h
			ymax_2d = boxes[0][i][2] * im_w
			xmax_2d = boxes[0][i][3] * im_h
			#print("#"*70)
			out_box.append((xmin_2d, ymin_2d, xmax_2d, ymax_2d))
		#print("#"*70)

	return out_box

if __name__ == "__main__":
    image_file = '002233.png'
    output_box = get_2d_box(image_file)
    for i in output_box:
       print(i)
    print("Done.")
