print"\tCode for analysing Spectrogram colour \n\t Author: Usama Saqib \n\t website: BitBytelab.co"
import cv2
import numpy as np;
import matplotlib.pyplot as plt

# Read image
im = cv2.imread("img_keypo.png")

lower_red = np.array([0,0,160])
upper_red = np.array([50,50,255])
mask = cv2.inRange(im, lower_red, upper_red)
mask = cv2.dilate(mask, None, iterations=5)


params = cv2.SimpleBlobDetector_Params()
 

params.filterByColor = True
params.blobColor = 255
   
detector = cv2.SimpleBlobDetector_create(params)

reversemask=255-mask
keypoints = detector.detect(mask)
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('original',im)
cv2.imshow('Blob',mask)
cv2.imshow('Keypoint', im_with_keypoints)

cv2.waitKey(0)
