#python eyeballcursor_v5.py

from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import time
import dlib
import cv2
import math
import pyautogui
import win32gui, win32con
import webbrowser

pyautogui.FAILSAFE = False

Minimize = win32gui.GetForegroundWindow()
win32gui.ShowWindow(Minimize, win32con.SW_MINIMIZE)

screen_rez = pyautogui.size()
rez_x = screen_rez[0]
rez_y = screen_rez[1]
calib_rez_x = int(0.75*rez_x)
calib_rez_y = int(0.75*rez_y)
frame_x = int(rez_x/2)
frame_y = int(rez_y/2)

def sound_alarm(path):
	# play an alarm sound
	playsound.playsound(path)

def eye_aspect_ratio(eye):

	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	EYE = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	cnter=(eye[0]+eye[3])/2;
	return EYE,cnter


EYE_AR_CONSEC_FRAMES = 3

COUNTER = 0
ALARM_ON = False

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor

cam = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
time.sleep(1.0)


muldiffcentX=0
muldiffcentY=0
CentInitX=rez_x/2
CentInitY=rez_y/2
FrameCentX=frame_x/2
FrameCentY=frame_y/2
lp=0



def CalibrationHelper(displaypointX, displaypointY, displayString):
	
	blank_image = 255 * np.ones(shape=[calib_rez_y, calib_rez_x, 3], dtype=np.uint8)
	cv2.circle(blank_image,(displaypointX,displaypointY),15,(0,255,0),-1)
	sttime = time.time()
	cv2.putText(blank_image, displayString, (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	while time.time()-sttime < 2:
		cv2.imshow("Calibration Frame", blank_image)
		cv2.moveWindow("Calibration Frame", int(rez_x/2) - int(calib_rez_x/2), int(rez_y/2) - int(calib_rez_y/2) - 35)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
	Xarr = []
	Yarr = []
	EAR_arr = []
	blank_image = 255 * np.ones(shape=[calib_rez_y, calib_rez_x, 3], dtype=np.uint8)
	cv2.circle(blank_image,(displaypointX,displaypointY),15,(0,255,0),-1)
	sttime = time.time()
	while time.time()-sttime < 3:
				
		__,frame = cam.read()
		
		frame = cv2.resize(frame,(frame_x, frame_y), interpolation = cv2.INTER_CUBIC)

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		rects = detector(gray, 0)

		for rect in rects:
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR,cnter = eye_aspect_ratio(leftEye)
			rightEAR,cnterr = eye_aspect_ratio(rightEye)
			xl = int(math.ceil(cnter[0]));
			yl = int(math.ceil(cnter[1]));
			xr = int(math.ceil(cnterr[0]));
			yr = int(math.ceil(cnterr[1]));
			x = int((xl+xr)/2)
			y = int((yl+yr)/2)
			cv2.circle(frame,(x,y),5,255,-1);
			EAR_arr.append((leftEAR+rightEAR)/2)
			Xarr.append(x)
			Yarr.append(y)
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		
		cv2.imshow("Frame", frame)
		cv2.imshow("Calibration Frame", blank_image)
		cv2.moveWindow("Calibration Frame", int(rez_x/2) - int(calib_rez_x/2), int(rez_y/2) - int(calib_rez_y/2) - 35)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break	
	cv2.destroyWindow("Calibration Frame")
	return Xarr,Yarr,EAR_arr

def Calibration():	
	x0_arr, y0_arr, EAR_arr = CalibrationHelper(int(calib_rez_x/2),int(calib_rez_y/2),"Please Look at the appearing dot and keep \n both your eyes blinked for 3 seconds")
	x0 = int(sum(x0_arr)/len(x0_arr))
	y0 = int(sum(y0_arr)/len(y0_arr))
	EYE_AR_THRESH = sum(EAR_arr)/len(EAR_arr)
	x1_arr, y1_arr, __ = CalibrationHelper(int(calib_rez_x/2),0,"Please Look at the appearing dot for 3 seconds")
	x1 = int(sum(x1_arr)/len(x1_arr))
	y1 = int(sum(y1_arr)/len(y1_arr))
	mul_y1 = (calib_rez_y/2)/abs(y0-y1)
	x2_arr, y2_arr, __ = CalibrationHelper(int(calib_rez_x/2),calib_rez_y,"Please Look at the appearing dot for 3 seconds")
	x2 = int(sum(x2_arr)/len(x2_arr))
	y2 = int(sum(y2_arr)/len(y2_arr))
	mul_y2 = (calib_rez_y/2)/abs(y0-y2)
	x3_arr, y3_arr, __ = CalibrationHelper(0,int(calib_rez_y/2),"Please Look at the appearing dot for 3 seconds")
	x3 = int(sum(x3_arr)/len(x3_arr))
	y3 = int(sum(y3_arr)/len(y3_arr))
	mul_x3 = (calib_rez_x/2)/abs(x0-x3)
	x4_arr, y4_arr, __ = CalibrationHelper(calib_rez_x,int(calib_rez_y/2),"Please Look at the appearing dot for 3 seconds")
	x4 = int(sum(x4_arr)/len(x4_arr))
	y4 = int(sum(y4_arr)/len(y4_arr))
	mul_x4 = (calib_rez_x/2)/abs(x0-x4)
	return x0, y0, mul_y1, mul_y2, mul_x3, mul_x4, EYE_AR_THRESH

x0, y0, mul_y1, mul_y2, mul_x3, mul_x4, EYE_AR_THRESH = Calibration()
mul_y1 = 0.75 * mul_y1
mul_y2 = 0.75 * mul_y2
mul_x3 = 0.75 * mul_x3
mul_x4 = 0.75 * mul_x4
#EYE_AR_THRESH = 1.2 * EYE_AR_THRESH
while (mul_y1>30 or mul_y2>30 or mul_x3>30 or mul_x4>30):
	x0, y0, mul_y1, mul_y2, mul_x3, mul_x4, EYE_AR_THRESH = Calibration()
	mul_y1 = 0.75 * mul_y1
	mul_y2 = 0.75 * mul_y2
	mul_x3 = 0.75 * mul_x3
	mul_x4 = 0.75 * mul_x4
	#EYE_AR_THRESH = 1.2 * EYE_AR_THRESH

print(x0, y0, mul_y1, mul_y2, mul_x3, mul_x4, EYE_AR_THRESH)

#pyautogui.hotkey('win', 'ctrl', 'o')

webbrowser.open_new('https://eye-tracker-d8638.web.app/')

while True:
	
	__,frame = cam.read()	
	frame = cv2.resize(frame,(frame_x, frame_y), interpolation = cv2.INTER_CUBIC)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)

	for rect in rects:

		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR,cnter = eye_aspect_ratio(leftEye)
		rightEAR,cnterr = eye_aspect_ratio(rightEye)
		xl = int(math.ceil(cnter[0]));
		yl = int(math.ceil(cnter[1]));
		xr = int(math.ceil(cnterr[0]));
		yr = int(math.ceil(cnterr[1]));
		x = int((xl+xr)/2)
		y = int((yl+yr)/2)
		cv2.circle(frame,(int(FrameCentX),int(FrameCentY)),5,(0,255,0),-1)
		cv2.circle(frame,(x,y),5,255,-1)
		if lp>0:
			diffcentX=x0-x;
			diffcentY=y0-y;
			#10,15   12,18    P20,25    15,20
			if(y<y0):
				muldiffcentY=diffcentY*mul_y1
			elif(y>=y0):
				muldiffcentY=diffcentY*mul_y2
			if(x<x0):
				muldiffcentX=diffcentX*mul_x3
			elif(x>=x0):
				muldiffcentX=diffcentX*mul_x4
		moveCordX = CentInitX+muldiffcentX
		moveCordY = CentInitY-muldiffcentY
		#print(moveCordX, moveCordY)
		if moveCordX>rez_x:
			moveCordX = rez_x
		elif moveCordX<0:
			moveCordX = 0
		if moveCordY>rez_y:
			moveCordY = rez_y
		elif moveCordY<0:
			moveCordY = 0
		if COUNTER == 0 and leftEAR > EYE_AR_THRESH and rightEAR > EYE_AR_THRESH:
			pyautogui.moveTo(moveCordX, moveCordY)
		lp=1

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if (leftEAR < EYE_AR_THRESH) or (leftEAR < EYE_AR_THRESH and rightEAR < EYE_AR_THRESH):
			#pyautogui.doubleClick() 
			COUNTER += 1

			# if the eyes were closed for a sufficient number of
			# then sound the alarm
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				pyautogui.click(button='left')
				COUNTER = 0
				# if the alarm is not on, turn it on
				if not ALARM_ON:
					ALARM_ON = True

				# draw an alarm on the frame
				cv2.putText(frame, "LEFT CLICK PRESSED!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		elif rightEAR < EYE_AR_THRESH:
			COUNTER += 1
			
			# if the eyes were closed for a sufficient number of
			# then sound the alarm
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				pyautogui.click(button='right')
				COUNTER = 0
				# if the alarm is not on, turn it on
				if not ALARM_ON:
					ALARM_ON = True


				# draw an alarm on the frame
				cv2.putText(frame, "RIGHT CLICK PRESSED!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
		else:
			COUNTER = 0
			ALARM_ON = False

		cv2.putText(frame, "Left Eye: {:.2f}, Right Eye: {:.2f}".format(leftEAR, rightEAR), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()
