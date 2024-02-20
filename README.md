# Cone-Detection-for-FSAE-Driverless
Hey there! 👋 Welcome to our FSAE Driverless Cone Detection project on GitHub! We're on a mission to make self-driving cars ace the Formula Student competition. This repo focuses on spotting and tracking cones using cool tech like computer vision and machine learning.

## Table of Contents:
- [Datasets](#datasets)
	- [The FSOCO Dataset](#fsoco)

# Datasets
All the datasets used in the project is added to this section
## The FSOCO Dataset
- Website: [www.fsoco-dataset.com](https://www.fsoco-dataset.com)

- The FSOCO dataset helps Formula Student / FSAE teams to get started with their visual perception system for driverless disciplines.

- FSOCO contains bounding box and segmentation annotations from multiple teams and continues to grow thanks to numerous contributions from the Formula Student community.

- i downloaded segmentation data because bouding box was 24gb :)

  # CODE
  - Resizing Code, first you need to resize all the data in 416-to-416 ratio , coz yolov2 works best with this size
  -     % Set your input and output folder paths
		inputFolderPath = 'C:\Users\mahen\OneDrive\Documents\DRVERLESS FORMULA BHARAT\Cone Detection\fsoco_segmentation_train\train';
		outputFolderPath = 'C:\Users\mahen\OneDrive\Documents\DRVERLESS FORMULA BHARAT\Cone Detection\resize\trainResize';

		% Set the desired new size (width, height)
		newSize = [416, 416];  % Adjust the size as needed

		% Create output folder if it doesn't exist
		if ~exist(outputFolderPath, 'dir')
		    mkdir(outputFolderPath);
		end

		% List all files in the input folder	
		files = dir(fullfile(inputFolderPath, '*.jpg'));  % change the image format according to your dataset

		% Loop through each file
		for i = 1:length(files)
    		% Read the image
    		imagePath = fullfile(inputFolderPath, files(i).name);
    		img = imread(imagePath);
    
    		% Resize the image
 		   resizedImg = imresize(img, newSize);
    
	    % Save the resized image to the output folder
	    [~, fileName, fileExt] = fileparts(files(i).name);
	    outputImagePath = fullfile(outputFolderPath, [fileName, '_resized', fileExt]);
	    imwrite(resizedImg, outputImagePath);
		end
- cone detection code
- 

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import cv2
import argparse
import torch
from models import Darknet
from utils.utils import xywh2xyxy, calculate_padding
from utils.nms import nms
from PIL import ImageDraw
import torchvision.transforms.functional as TF
from PIL import Image as PILImage
import numpy as np

detection_tmp_path = "/tmp/detect/"

class ObjectDetectorROS:
    def __init__(self):
        rospy.init_node('object_detector')

        self.bridge = CvBridge()
        self.model = self.load_model()
        self.conf_thres = rospy.get_param('~conf_thres', 0.8)
        self.nms_thres = rospy.get_param('~nms_thres', 0.25)
        self.output_path = rospy.get_param('~output_path', 'outputs/visualization/')

        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)

    def load_model(self):
        # Load your model here
        model_cfg = rospy.get_param('~model_cfg', 'model_cfg/yolo_baseline.cfg')
        weights_path = rospy.get_param('~weights_path', 'path/to/weights')
        xy_loss = rospy.get_param('~xy_loss', 2)
        wh_loss = rospy.get_param('~wh_loss', 1.6)
        no_object_loss = rospy.get_param('~no_object_loss', 25)
        object_loss = rospy.get_param('~object_loss', 0.1)
        vanilla_anchor = rospy.get_param('~vanilla_anchor', False)

        model = Darknet(config_path=model_cfg, xy_loss=xy_loss, wh_loss=wh_loss, no_object_loss=no_object_loss,
                        object_loss=object_loss, vanilla_anchor=vanilla_anchor)

        if torch.cuda.is_available():
            model.cuda()
        model.load_weights(weights_path, model.get_start_weight_dim())
        model.eval()
        return model

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        detections = self.detect(cv_image)
        # Process detections further if needed
        rospy.loginfo("Detections: {}".format(detections))

    def detect(self, image):
        img = PILImage.fromarray(image)
        w, h = img.size
        new_width, new_height = self.model.img_size()
        pad_h, pad_w, ratio = calculate_padding(h, w, new_height, new_width)
        img = TF.pad(img, padding=(pad_w, pad_h, pad_w, pad_h), fill=(127, 127, 127), padding_mode="constant")
        img = TF.resize(img, (new_height, new_width))

        img = TF.to_tensor(img)
        img = img.unsqueeze(0)

        with torch.no_grad():
            img = img.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            output = self.model(img)

            for detections in output:
                detections = detections[detections[:, 4] > self.conf_thres]
                box_corner = torch.zeros((detections.shape[0], 4), device=detections.device)
                xy = detections[:, 0:2]
                wh = detections[:, 2:4] / 2
                box_corner[:, 0:2] = xy - wh
                box_corner[:, 2:4] = xy + wh
                probabilities = detections[:, 4]
                nms_indices = nms(box_corner, probabilities, self.nms_thres)
                main_box_corner = box_corner[nms_indices]
                if nms_indices.shape[0] == 0:
                    continue

            img_with_boxes = Image.open(image)
            draw = ImageDraw.Draw(img_with_boxes)

            for i in range(len(main_box_corner)):
                x0 = main_box_corner[i, 0].cpu().item() / ratio - pad_w
                y0 = main_box_corner[i, 1].cpu().item() / ratio - pad_h
                x1 = main_box_corner[i, 2].cpu().item() / ratio - pad_w
                y1 = main_box_corner[i, 3].cpu().item() / ratio - pad_h
                draw.rectangle((x0, y0, x1, y1), outline="red")

            img_with_boxes.show()

            return main_box_corner

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        detector = ObjectDetectorROS()
        detector.run()
    except rospy.ROSInterruptException:
        pass


