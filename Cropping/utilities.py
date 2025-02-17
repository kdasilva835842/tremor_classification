import numpy as np
import cv2
from pdf2image import convert_from_path
import time
from imutils import contours
from imutils.object_detection import non_max_suppression

def convertPDFtoJGP(originalImage,finalImage,dpi=200):
	pages = convert_from_path(originalImage, dpi)
	for page in pages:
		page.save(finalImage, 'JPEG')

def fixOrientation(image):
	cropToleranceA = 0.9
	cropToleranceB = 0.95
	MidPlaneFractionX = 0.4
	MidPlaneFractionY = 0.5

	finalA = image.copy()

	(h,w) = image.shape[:2]
	mask = np.zeros((h,w), np.uint8)
	cv2.rectangle(mask, (0,int(0.25*h)), (w, int(0.75*h)), 255, -1)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.bitwise_and(image,mask)

	kernel = np.ones((3,3),np.uint8)
	image = cv2.erode(image,kernel,iterations = 1)

	image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2BGR)

	newW = 704
	newH = 704
	minConf = 0.8
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

	## If there are 2 words in the top half, its the corretc orientation. Otherwise, flip it
	for i in range(0,len(xCoordStart)):
		if yCoordStart[i] < height * MidPlaneFractionY:
			orientation = orientation + 1
		elif yCoordStart[i] > height * MidPlaneFractionY:
			orientation = orientation -1
		else:
			print("does not match conditions")

	#print(orientation)

	if orientation < 0:
		finalA = cv2.flip(finalA, -1)
	else:
		finalA = finalA

	return finalA