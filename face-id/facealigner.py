
# from: https://github.com/jrosebr1/imutils/tree/master/imutils/face_utils

from collections import OrderedDict
import numpy as np
import cv2

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])


def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)

	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords


class FaceAligner:
	def __init__(self, predictor, desiredLeftEye=(0.32, 0.32),
		desiredFaceWidth=256, desiredFaceHeight=None):
		# store the facial landmark predictor, desired output left
		# eye position, and desired output face width + height
		self.predictor = predictor
		self.desiredLeftEye = desiredLeftEye
		self.desiredFaceWidth = desiredFaceWidth
		self.desiredFaceHeight = desiredFaceHeight

		# https://github.com/AlfredXiangWu/face_verification_experiment/blob/master/code/face_db_align.m
		# ec_mc_y=48, ec_y =40
		# self.ec_mc_y = ec_mc_y
		# self.ec_y =ec_y

		# if the desired face height is None, set it to be the
		# desired face width (normal behavior)
		if self.desiredFaceHeight is None:
			self.desiredFaceHeight = self.desiredFaceWidth


	def align(self, image, gray, rect): # align with eyes and mouth center, as in: 
		# https://github.com/AlfredXiangWu/face_verification_experiment/blob/master/code/face_db_align.m
		
		# convert the landmark (x, y)-coordinates to a NumPy array
		shape = self.predictor(gray, rect)
		shape = shape_to_np(shape)

		# extract the left and right eye (x, y)-coordinates
		(lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
		(rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
		# (mStart, mEnd) = FACIAL_LANDMARKS_IDXS["mouth"]
		leftEyePts = shape[lStart:lEnd]
		rightEyePts = shape[rStart:rEnd]
		# mouthPts = shape[mStart:mEnd]

		# compute the center of mass for each eye
		leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
		rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
		# mounthCenter = mouthPts.mean(axis=0).astype("int")

		# compute the angle between the eye centroids
		dY = rightEyeCenter[1] - leftEyeCenter[1]
		dX = rightEyeCenter[0] - leftEyeCenter[0]
		angle = np.degrees(np.arctan2(dY, dX)) - 180

		# compute the desired right eye x-coordinate based on the
		# desired x-coordinate of the left eye
		desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

		# compute center (x, y)-coordinates (i.e., the median point)
		# between the two eyes in the input image
		eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
			(leftEyeCenter[1] + rightEyeCenter[1]) // 2)

		# determine the scale of the new resulting image
		dist = np.sqrt((dX ** 2) + (dY ** 2))
		desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
		desiredDist *= self.desiredFaceWidth
		scale = desiredDist / dist
		# using: https://github.com/AlfredXiangWu/face_verification_experiment/blob/5142b9db314557ba1fea974dee74496f5ec4d2c3/README.md
		# scale = self.ec_mc_y/(mounthCenter[1]-eyesCenter[1])

		# grab the rotation matrix for rotating and scaling the face
		M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

		# update the translation component of the matrix
		tX = self.desiredFaceWidth * 0.5
		tY = self.desiredFaceHeight * self.desiredLeftEye[1]
		M[0, 2] += (tX - eyesCenter[0])
		M[1, 2] += (tY - eyesCenter[1])

		# apply the affine transformation
		(w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
		output = cv2.warpAffine(image, M, (w, h),
			flags=cv2.INTER_CUBIC)

		# test save:
		# cv2.imwrite('test-align.png', output)

		# return the aligned face
		return output, (rightEyeCenter, leftEyeCenter)
