### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

You need to download the mobile-net ssd model from zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). other than that you need to install
	1. cv2 (pip install opencv-python)
	2. pyttsx (pip install pyttsx3)
	and some other installation like imutilus, numpy, argparser and time

## Project Motivation<a name="motivation"></a>

this project was our first attempt to provide a basic navigation system to a visually impaired person in order to help them in there day to day life style. altough it was a small step but our team is working continously on this fields. we need some fundings too if you are interested you can contact us.

## File Descriptions <a name="files"></a>

	1.There are a pre trained model for object detection.
	2. a python file consist of codes to run the project.
	3. a prototxt file having info of objects that can be detected

## Results<a name="results"></a>

(python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel)
use the above mentioned command to run the code.
The result of this is a model that can also run on a low powered device which can detect commonly used objects like television bottles etc and convet the result to an audio and spell through the speakers on real time.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

this project is prepared while the ideathon conducted in patna this project use mobile-net model for object detection provided by model zoo.

## Authors copy<a name="licensing"></a>

the modified code is the work of Prince Amit and for using the mentioned work you can mention this part of readme.