#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os.path
import cv2.cv as cv
import sys
import scipy as sp
import pylab as plt

# BGRをRGBに変更
def bgr2rbg(im):

	b,g,r = cv2.split(im)       # get b,g,r
	im = cv2.merge([r,g,b])     # switch it to rgb
	return im

if __name__ == "__main__" :
	# カメラ映像の取得
	capture = cv2.VideoCapture(0)
	
	# 検出器の読み込み
	face_cascade = cv2.CascadeClassifier("./haarcascade/haarcascade_frontalface_alt.xml")
	eye_cascade = cv2.CascadeClassifier("./haarcascade/haarcascade_eye_tree_eyeglasses.xml")
	eye_cascade2 = cv2.CascadeClassifier("./haarcascade/haarcascade_mcs_lefteye.xml")
	nose_cascade = cv2.CascadeClassifier("./haarcascade/haarcascade_mcs_nose.xml")
	
	scale = 1
	delta = 0
	ddepth = cv2.CV_16S
	
	while True :
		ret, frame = capture.read()
		# グレースケール
		im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		im_gray2 = np.copy(im_gray)
		
		
		im_gray = cv2.GaussianBlur(im_gray,(3,3),0)
		# 認識して、領域を保存(複数ある場合は配列になる)
		facerect = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
		kernel = np.ones((5,5), np.uint8)
		
		kernel2 = np.array([[0,-1,0],
							[-1,5,-1],
							[0,-1,0] ],np.float32)
		im_gray2 = cv2.filter2D(im_gray2,-1,kernel2) # -1:入力画像と同じ深さ
		
		#im_gray2 = bgr2rbg(im_gray2)
		
		#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
		erosion = cv2.erode(im_gray,kernel,iterations = 1)
		
		
		if len(facerect) > 0 :
			facecolor = np.copy(frame)
			face = np.copy(im_gray)
			facepaste = np.copy(facecolor)
			
			for x, y, w, h in facerect :
				face = im_gray[y:y+h+20 if y+h+20 else y+h, x:x+w]
				facecolor = frame[y:y+h+20 if y+h+20 else y+h, x:x+w]
				facepaste = np.copy(facecolor)
				
			# 鼻の位置
			noserect = mouserect = nose_cascade.detectMultiScale(face, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
			#eyerect = eye_cascade.detectMultiScale(face, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
			#eyerect2 = eye_cascade2.detectMultiScale(face, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
			
			#face_hsv = cv2.cvtColor(facecolor, cv2.COLOR_BGR2HSV)
			#skin_color = face_hsv[facecolor.shape[0]/2, facecolor.shape[1]/2]
			#skin_min = np.array([skin_color[0] - 5 if (skin_color[0] - 5 >= 0) else 0, skin_color[1] - 40 if (skin_color[1] - 40 >= 0) else 0, skin_color[2] - 60 if (skin_color[2] - 60 >= 0) else 0])
			#skin_max = np.array([skin_color[0] + 5 , 255, 255])
			#skin_mask = cv2.inRange(face_hsv, skin_min, skin_max)
			# ゴマ塩ノイズ除去
			#skin_mask = cv2.medianBlur(skin_mask, 7)
			#skin_mask = cv2.Canny(skin_mask, 50, 200)
			
			#cv2.imshow("mask", skin_mask)
			
			#face = cv2.Canny(face, 50, 200)
			
#			if len(eyerect) > 0 :
#				for x, y, w, h in eyerect :
#					cv2.rectangle(facecolor, (x,y), (x+w, y+h), (255, 0, 255), 1)
#			
#			left_eye_x = 10000
#			left_eye_y = 10000
#			left_eye_h = 0
#			left_eye_w = 0
#			if len(eyerect2) > 0 :
#				for x, y, w, h in eyerect2 :
#					if x <= face.shape[1]/2 and x <= face.shape[0]/2 :
#						if left_eye_h <= h and left_eye_w <= w :
#							print "a"
#							left_eye_x = x
#							left_eye_y = y
#							left_eye_h = h
#							left_eye_w = w
#				cv2.rectangle(facecolor, (left_eye_x,left_eye_y), (left_eye_x+left_eye_w, left_eye_y+left_eye_h), (255, 0, 0), 1)
			
			if len(noserect) > 0:
				for x, y, w, h in noserect :
					if face.shape[0] / 2 >= y and face.shape[1] / 2 >= x and face.shape[0] / 2 <= y+h and face.shape[1] / 2 <= x+w :
						cv2.rectangle(facecolor, (x, y), (x+w, y+h), (255, 255, 0), 1)
			#cv2.imshow("face", facecolor)
			

		#im_gray = cv2.morphologyEx(im_gray, cv2.MORPH_OPEN, kernel)
		#im_gray = cv2.dilate(im_gray, kernel, iterations = 1)
		cv2.imshow("frame", frame)
		
		#print2 = cv2.Laplacian(im_gray2, cv2.CV_64F)
		
		#cv2.imshow("laplacian", print2)
		
		# sobelフィルタ
		sobelx = cv2.Sobel(im_gray,cv2.CV_64F,1,0,ksize = 3, scale = scale, delta = delta,borderType = cv2.BORDER_DEFAULT)
		sobely = cv2.Sobel(im_gray,cv2.CV_64F,0,1,ksize = 3, scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)
		
		abs_grad_x = cv2.convertScaleAbs(sobelx)   # converting back to uint8
		abs_grad_y = cv2.convertScaleAbs(sobely)
		dst = cv2.addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0)
		
		
		cv2.imshow('dst',dst)
		
		# sobelフィルタ  2
		sobelx = cv2.Sobel(im_gray2,cv2.CV_16S,1,0,ksize = 3, scale = scale, delta = delta,borderType = cv2.BORDER_DEFAULT)
		sobely = cv2.Sobel(im_gray2,cv2.CV_16S,0,1,ksize = 3, scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)
		
		abs_grad_x = cv2.convertScaleAbs(sobelx)   # converting back to uint8
		abs_grad_y = cv2.convertScaleAbs(sobely)
		dst = cv2.addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0)
		
		cv2.imshow("as", dst)
		
		#for y in range(dst.shape[0]) :
		#	for x in range(dst.shape[1]) :
		#		print dst[y][x]
		#dst = dst * 2
		#mask = dst > 30
		#mask2 = dst != 0
		#dst[mask] = 255
		# ゴマ塩ノイズ除去
		#dst = cv2.medianBlur(dst, 7)
		#dst = cv2.Canny(dst, 10, 40)
		#cv2.imshow("a", dst)
		
		#dilation = cv2.dilate(dst,kernel,iterations = 1)
		#tophat = cv2.morphologyEx(dst, cv2.MORPH_TOPHAT, kernel)
		#closing = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
		#gradient = cv2.morphologyEx(dst, cv2.MORPH_GRADIENT, kernel)
		#opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)
		#closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
		#erosion = cv2.erode(opening,kernel,iterations = 1)
		
		#cv2.imshow("dilation", erosion)
		
		# キー入力待機
		key = cv2.waitKey(10)
		# 空白キー押したときのフレームを背景画像として生成
		if key == 32:
			cv2.imwrite('./cap/bg.jpg',frame)
		elif key > 0 :
			break
			
	# キャプチャー解放
	capture.release()
	# ウィンドウ破棄
	cv2.destroyAllWindows()