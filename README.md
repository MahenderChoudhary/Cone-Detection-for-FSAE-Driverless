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
  - Resizing Code, first you need to resize all the data in 416-to-418 ratio
  - 	% Set your input and output folder paths
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

