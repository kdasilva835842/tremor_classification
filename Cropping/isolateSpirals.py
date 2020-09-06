import numpy as np
import argparse
import cv2
import imutils
from imutils import contours
from imutils.object_detection import non_max_suppression
import time
import utilities
import processingutils as putils
from skimage.filters import threshold_local
from scipy import ndimage
import math
import glob
import os	
import re
import tempfile

### Read in and convert to jpg
image_array = []
image_names = []
counter = 0
rawCount = 0
sizeErrorCountA = 0
sizeErrorCountB = 0

with tempfile.TemporaryDirectory() as tempDir:
	print('created temporary directory', tempDir)
	#for filename in glob.glob('/Users/KJ/Google Drive/patientData/SortedData/Before/Non-Dominant/Control/*.pdf'):
	for filename in glob.glob('/Users/kelvi/Documents/Masters/patientData/RegressionDatabase/Control/non/*.pdf'):
		arrayName= os.path.basename(filename)
		arrayName=arrayName.replace(".pdf","")
		arrayName = arrayName + "_non_Cont"
		newName = filename.replace(".pdf","")
		newName = str(tempDir) + "/"+str(arrayName)
		# print(newName)
		# print(arrayName)
		utilities.convertPDFtoJGP(filename,newName+'.jpg',150)
		image = cv2.imread(newName+'.jpg')
		image_array.append(image)
		image_names.append(arrayName)
		#print(arrayName)
		# counter = counter + 1
		rawCount = rawCount + 1

imagePathA = "/Users/kelvi/Desktop/DATA/nonDominantControl/DrawingA/"
imagePathB = "/Users/kelvi/Desktop/DATA/nonDominantControl/DrawingB/"
cropToleranceA = 0.9
cropToleranceB = 0.95
MidPlaneFractionX = 0.4
MidPlaneFractionY = 0.5

#Desired square dimension of each isolated spiral
squareDimension = 300

for counter,image in enumerate(image_array):
	print("Starting image ", image_names[counter])
	copied = image.copy()

	orientedImage = utilities.fixOrientation(copied)
	# orig = orientedImage.copy()
	finalA = orientedImage.copy()
	finalB = orientedImage.copy()
	image = orientedImage.copy()

	(h,w) = image.shape[:2]
	mask = np.zeros((h,w), np.uint8)
	cv2.rectangle(mask, (0,int(0.245*h)), (w, int(0.6*h)), 255, -1)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.bitwise_and(image,mask)
	kernel = np.ones((3,3),np.uint8)
	image = cv2.erode(image,kernel,iterations = 1)
	image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2BGR)

	newW = 704
	newH = 704
	minConf = 0.96
	(H,W) = image.shape[:2]
	rW = W / float(newW)
	rH = H / float(newH)
	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]

	# define the two output layer names for the EAST detector model that
	# we are interested -- the first is the output probabilities and the
	# second can be used to derive the bounding box coordinates of text
	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]

	# load the pre-trained EAST text detector
	#print("[INFO] loading EAST text detector...")
	net = cv2.dnn.readNet("frozen_east_text_detection.pb")

	# construct a blob from the image and then perform a forward pass of
	# the model to obtain the two output layer sets
	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	#start = time.time()
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)
	#end = time.time()

	# show timing information on text prediction
	#print("[INFO] text detection took {:.6f} seconds".format(end - start))

	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []
	
	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the geometrical
		# data used to derive potential bounding box coordinates that
		# surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

	# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability, ignore it
			if scoresData[x] < minConf:
				continue
	
			# compute the offset factor as our resulting feature maps will
			# be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)
	
			# extract the rotation angle for the prediction and then
			# compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
	
			# use the geometry volume to derive the width and height of
			# the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]
	
			# compute both the starting and ending (x, y)-coordinates for
			# the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)
	
			# add the bounding box coordinates and probability score to
			# our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])   

	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	boxes = non_max_suppression(np.array(rects), probs=confidences)
	
	xCoordStart = []
	xCoordEnd = []
	yCoordStart = []
	yCoordEnd = []
	# loop over the bounding boxes
	for (startX, startY, endX, endY) in boxes:
		# scale the bounding box coordinates based on the respective
		# ratios
		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)
	
		# draw the bounding box on the image
		# cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
		xCoordStart.append(startX) # vector of x coords
		yCoordStart.append(startY)
		xCoordEnd.append(endX)
		yCoordEnd.append(endY)

	(height,width) = finalA.shape[:2]
	orientation = 0

	## Sorting coords
	xStartOrdered = np.full([3], 1000)
	yStartOrdered = np.full([3], 1000)
	xEndOrdered = np.full([3], 1000)
	yEndOrdered = np.full([3], 1000)

	(height,width) = finalA.shape[:2]

	for i in range(0,len(xCoordStart)):
		if xCoordStart[i] < width * MidPlaneFractionX and yCoordStart[i] < height * MidPlaneFractionY:
			xStartOrdered[0]=xCoordStart[i]
			yStartOrdered[0]=yCoordStart[i]
			xEndOrdered[0]=xCoordEnd[i]
			yEndOrdered[0]=yCoordEnd[i]
		elif xCoordStart[i] < width * MidPlaneFractionX and yCoordStart[i] > height * MidPlaneFractionY:
			xStartOrdered[1]=xCoordStart[i]
			yStartOrdered[1]=yCoordStart[i]
			xEndOrdered[1]=xCoordEnd[i]
			yEndOrdered[1]=yCoordEnd[i]
		elif xCoordStart[i] > width * MidPlaneFractionX and yCoordStart[i] < height * MidPlaneFractionY and yCoordEnd[i] < yEndOrdered[2]:
			xStartOrdered[2]=xCoordStart[i]
			yStartOrdered[2]=yCoordStart[i]
			xEndOrdered[2]=xCoordEnd[i]
			yEndOrdered[2]=yCoordEnd[i]
		else:
			print("does not match conditions crop ",image_names[counter], ".jpg")
			print(xCoordStart[i], yCoordStart[i], width * MidPlaneFractionX, height * MidPlaneFractionY)

	### Cropping out spiral A
	width = (xStartOrdered[2]*cropToleranceA-xStartOrdered[0])
	finalA = cv2.cvtColor(finalA, cv2.COLOR_BGR2GRAY)
	finalA = finalA[int(yEndOrdered[0]):int(yEndOrdered[0]+width), int(xStartOrdered[0]):int(xStartOrdered[2]*cropToleranceA)]

	try:
		(h,widthA) = finalA.shape[:2]

		finalA = imutils.resize(finalA, width=squareDimension)
		(hFinalA,wFinalA) = finalA.shape[:2]
		if hFinalA != wFinalA:
			sizeErrorCountA = sizeErrorCountA + 1


		#finalA = finalA[0:squareDimension,0:squareDimension]
		cv2.imwrite(str(imagePathA)+str(image_names[counter])+"_A_"+".jpg",finalA)

		drawingWidth = xStartOrdered[2] - xStartOrdered[0]
		leftOfB = xStartOrdered[0] + cropToleranceB*drawingWidth
		### Cropping out spiral B
		finalB = cv2.cvtColor(finalB, cv2.COLOR_BGR2GRAY)
		finalB = finalB[ int(yEndOrdered[2]) : int(yEndOrdered[2]+widthA), int(leftOfB): int(leftOfB+widthA)]

		(heightB,widthB) = finalB.shape[:2]

		finalB = imutils.resize(finalB, width=squareDimension)
		(hFinalB,wFinalB) = finalB.shape[:2]
		if hFinalB != wFinalB:
			sizeErrorCountB = sizeErrorCountB + 1
		#finalB = finalB[0:squareDimension,0:squareDimension]
		cv2.imwrite(str(imagePathB)+str(image_names[counter])+"_B_"+".jpg",finalB)

		print("Finished image ", image_names[counter])

	except Exception as e:
		print("Unable to save to file as image size is empty: ",str(e))

countSpiralA = glob.glob(str(imagePathA)+ "*.jpg")
countSpiralB = glob.glob(str(imagePathB)+ "*.jpg")
countSpiralA = len(countSpiralA)
countSpiralB = len(countSpiralB)

print("Percentage of Spiral A processed:", (float(countSpiralA-sizeErrorCountA)/float(rawCount))*100)
print("Percentage of Spiral B processed:", (float(countSpiralB-sizeErrorCountB)/float(rawCount))*100)