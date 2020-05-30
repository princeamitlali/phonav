# use code below to run the script in terminal
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# neccesary package that are used in the module import them
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import pyttsx3
c = 0

counter = 0                #counter to speack on desired frame

# pyttsx is a offline text to speech synthesizer that spell out the detected object
engine = pyttsx3.init()
rate = engine.getProperty('rate')   
rate = 200
voices = engine.getProperty('voices')
engine.setProperty('rate',rate)
voice_id = 'tamil'
engine.setProperty('voices',voice_id)



# parse the arguments  by creating argument parse
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")       #caffe is a framework of deep learning
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")           #load the pre trained model for object detection
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")    # here confidence is the probability of object to fall in a certain level
args = vars(ap.parse_args())

# make the list that can be detected by the mobile ssd net model and assign them index values
# detect the object and then assign a square or rectangle along with the detected objectprimarly we can detect 22 diffrent objects
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "table",
	"dog", "horse", "bike", "person", "plant", "sheep",
	"sofa", "train", "tv"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
#later on devlopment of project we will try to get a better pretrained model or can combine diffrent pretrained models to detect more objects

# the model is stored in the same folder with name mobile net ssd 
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# start the video from the camera of the laptop for now we use only laptop camera but later on we use stereo camera the code then slightly,
# change we use two input compare them and the common of the both are produced as the output for current we set video speed 2-3 fps
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(1.0)
fps = FPS().start()

# put the video in the loop so as the stream is not affected until manually killed
while True:
	# vs here is video stream we take video stream as an input and resize using imutils imutils is a image processing package
	# here for the sack of presentation we display the image so as it can be verified that what is detected object resized to 900 px
	frame = vs.read()
	frame = imutils.resize(frame, width=900)

	# we convert the image received into blob i.e binary large object
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 600)),
		0.007843, (300, 300), 127.5)

	# now pass the blob to the netframe work to get the predictions it is able to detect many objects together with there percentage 
	# but sack of simplicity we are not displaying the percentage of the predicted object
	net.setInput(blob)
	detections = net.forward()

	# as there can be more than a single object in the blob file so we go through each of then to get there confidence score
	for i in np.arange(0, detections.shape[2]):
		# we can also compare the weak predictions to remove them here the minimum criterial for selection is .2 of confidence
		confidence = detections[0, 0, i, 2]

	#the predicted object is selected and there dimesion in the picture is noted to mark rectangle around them

		idx = int(detections[0, 0, i, 1])
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

			# based on prediction the index is found and feeded to label them according to the list classes
		label = CLASSES[idx]
		counter = counter +1   #a counter is provided to repeat at a certain time in furthur devlopment we deploy a tracker to
		if counter == 8:      # to the detected object so as any large movment in object is predicted and can be informed
			v =label
			engine.say(v)
			engine.runAndWait()
			counter = 0
			   
		cv2.rectangle(frame, (startX, startY), (endX, endY),  #this is for the marking on the image
			COLORS[idx], 2)
		y = startY - 15 if startY - 15 > 15 else startY + 15
		cv2.putText(frame, label, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
	
		
			

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(200) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()