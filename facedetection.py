#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os.path
import cv2.cv as cv
import sys
import scipy as sp

if __name__ == "__main__" :
	# カメラ映像の取得
	capture = cv2.VideoCapture(0)
	
	# 検出器の読み込み
	face_cascade = cv2.CascadeClassifier("./haarcascade/haarcascade_frontalface_alt.xml")
	eye_cascade = cv2.CascadeClassifier("./haarcascade/haarcascade_mcs_eyepair_big.xml")
	nose_cascade = cv2.CascadeClassifier("./haarcascade/haarcascade_mcs_nose.xml")
	
	detector = cv2.FeatureDetector_create("SIFT")
	descriptor = cv2.DescriptorExtractor_create("SIFT")
	
	
	while True :
		ret, frame = capture.read()
		# グレースケール
		im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# 認識して、領域を保存(複数ある場合は配列になる)
		facerect = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
		
		captmpl = detector.detect(im_gray)
		capkeypoints, capdescriptors = descriptor.compute(im_gray, captmpl)
		
		if len(facerect) > 0 :
			
			facecolor = np.copy(frame)
			face = np.copy(im_gray)
			facepaste = np.copy(facecolor)
			
			for x, y, w, h in facerect :
				face = im_gray[y:y+h+20 if y+h+20 else y+h, x:x+w]
				facecolor = frame[y:y+h+20 if y+h+20 else y+h, x:x+w]
				facepaste = np.copy(facecolor)
				
			face_bi = np.zeros((face.shape[0], face.shape[1]), np.uint8)
			
			
			# 目、鼻の領域を取得
			eyerect = eye_cascade.detectMultiScale(face, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
			noserect = mouserect = nose_cascade.detectMultiScale(face, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
			if len(eyerect) > 0 :
				eyeleft = np.copy(eyerect[0])
				eyeright = np.copy(eyerect[0])
				for x, y, w, h in eyerect :
					# 目の領域は、顔の半分より上なので調べる範囲を見る
					if face_bi.shape[0]/2 >= y :
						# 両目の領域を切り抜き
						eyeleft = np.copy(facecolor[y:y+h, x:x+w/2])
						eyeright = np.copy(facecolor[y:y+h, x+w/2:x+w])
						
						# 確認用画像
						cv2.rectangle(facepaste, (x, y), (x+w, y+h), (255, 255, 0), 1)
						cv2.line(facepaste, (x + w/2, y), (x + w/2, y+h), (255, 255, 0), 1)
						# 両名のエッジの画像を作る
						eyeleft = cv2.Canny(eyeleft, 100, 200)
						eyeright = cv2.Canny(eyeright, 100, 200)
						
						# 目の特徴点描画（左目）
						left_left = (1000, 1000)	# 左目左端
						left_right = (0, 0)		 	# 左目右端
						left_up = (1000, 1000)		# 左目上
						left_down = (0, 0)			# 左目下
						# エッジの立っている部分を検索(左右の特徴点)
						for yy in range(eyeleft.shape[0]) :
							for xx in range(eyeleft.shape[1]) :
								if eyeleft[yy][xx] == 255 :
									#cv2.circle(facepasete)
									if xx <= left_left[0] :
										left_left = (xx, yy)
									if xx >= left_right[0]:
										left_right = (xx, yy)
						
						# 上下の特徴点を調べる
						for yy in range(eyeleft.shape[0]) :
							if yy < 0 or yy >= eyeleft.shape[0]:
								continue
							if eyeleft[yy][(left_left[0]+left_right[0])/2] == 255 :
								if yy <= left_up[1] :
									left_up = ((left_left[0]+left_right[0])/2, yy)
								if yy >= left_down[1] :
									left_down = ((left_left[0]+left_right[0])/2, yy)
						# 目の特徴点を描画
						cv2.circle(facepaste, (x+left_left[0], y+left_left[1]), 5, (255, 0, 255))
						cv2.circle(facepaste, (x+left_right[0], y+left_right[1]), 5, (255, 0, 255))
						cv2.circle(facepaste, (x+left_up[0], y+left_up[1]), 5, (255, 0, 255))
						cv2.circle(facepaste, (x+left_down[0], y+left_down[1]), 5, (255, 0, 255))
						
						# 目の特徴点描画（右目）
						right_left = (1000, 1000)	# 右目左端
						right_right = (0, 0)		# 右目右端
						right_up = (1000, 1000)		# 右目上
						right_down = (0, 0)			# 右目下
						# エッジの立っている部分を検索(左右の特徴点)
						for yy in range(eyeright.shape[0]) :
							for xx in range(eyeright.shape[1]) :
								if eyeright[yy][xx] == 255 :
									#cv2.circle(facepasete)
									if xx <= right_left[0] :
										right_left = (xx, yy)
									if xx >= right_right[0]:
										right_right = (xx, yy)
						
						# 上下の特徴点を調べる
						for yy in range(eyeright.shape[0]) :
							if yy < 0 or yy >= eyeright.shape[0]:
								continue
							if eyeright[yy][(right_left[0]+right_right[0])/2] == 255 :
								if yy <= right_up[1] :
									right_up = ((right_left[0]+right_right[0])/2, yy)
								if yy >= right_down[1] :
									right_down = ((right_left[0]+right_right[0])/2, yy)
						# 目の特徴点を描画(右目)
						cv2.circle(facepaste, (x+w/2+right_left[0], y+right_left[1]), 5, (255, 0, 255))
						cv2.circle(facepaste, (x+w/2+right_right[0], y+right_right[1]), 5, (255, 0, 255))
						cv2.circle(facepaste, (x+w/2+right_up[0], y+right_up[1]), 5, (255, 0, 255))
						cv2.circle(facepaste, (x+w/2+right_down[0], y+right_down[1]), 5, (255, 0, 255))
				cv2.imshow("left", eyeleft)
				cv2.imshow("right", eyeright)
			
			
			# 鼻が存在するときに見る
			if len(noserect) > 0 :
				# 口の領域もここで作る
				# 鼻の下に口は存在するので、鼻の位置がわかれば口はわかる
				# 口の検出器だと、誤検出が多発するため
				mousex = 0
				mousew = 0
				mousey = 0
				nose = np.copy(frame)
				#print nose.shape
				mouse = np.copy(noserect[0])
				
				# 鼻の領域を切り抜くと同時に口の領域を切り抜く
				nosex = 0
				nosey = 0
				nosew = 0
				for x, y, w, h in noserect :
					if face_bi.shape[0]/2 >= y and face_bi.shape[0]/2 <= y+h:
						nosex = x
						nosey = y
						nosew = w
						nose = np.copy(facecolor[y:y+h, x:x+w])
						#cv2.rectangle(facecolor, (x, y), (x+w, y+h), (0, 255, 0), 1)
						mousex = x - 10
						mousew = x+w+10
						mousey = y+h
				
				#print mousex, "  ", mousey
				if mousex != 0 and mousey != 0 :
					mouse = np.copy(facecolor[mousey:face_bi.shape[0], mousex:mousew])
					#cv2.rectangle(facecolor, (mousex, mousey), (mousew, face_bi.shape[0]), (0, 0, 255))
				
				# 鼻のエッジをとる
				nose = cv2.Canny(nose, 100, 200)
				#mouse = cv2.Canny(mouse, 0, 200)
				# 唇の色抽出
				mouse_hsv = cv2.cvtColor(mouse, cv2.COLOR_BGR2HSV)
				skin_color = mouse_hsv[mouse.shape[0] / 2][mouse.shape[1] / 2]
				skin_min = np.array([skin_color[0] - 5 if (skin_color[0] - 5 >= 0) else 0, skin_color[1] - 60 if (skin_color[1] - 60 >= 0) else 0, skin_color[2] - 60 if (skin_color[2] - 60 >= 0) else 0])
				skin_max = np.array([skin_color[0] + 5 , 255, 255])
				skin_mask = cv2.inRange(mouse_hsv, skin_min, skin_max)
				# ゴマ塩ノイズ除去
				skin_mask = cv2.medianBlur(skin_mask, 7)
				skin_mask = cv2.Canny(skin_mask, 100, 200)
				
				# 鼻の特徴点を取る
				# 鼻の特徴点描画（左目）
				nose_left = (1000, 1000)	# 鼻左端
				nose_right = (0, 0)			# 鼻右端
				nose_up = (1000, 1000)		# 鼻上
				nose_down = (0, 0)			# 鼻下
				for y in range(nose.shape[0]) :
					for x in range(nose.shape[1]) :
						# エッジの立っている部分を検索(左右の特徴点)
						if nose[y][x] == 255 :
							#cv2.circle(facepasete)
							if x <= nose_left[0] :
								nose_left = (x, y)
							if x >= nose_right[0]:
								nose_right = (x, y)
				# 鼻の特徴点を描画
				cv2.circle(facepaste, (nosex+nose_left[0], nosey+nose_left[1]), 5, (255, 0, 255))
				cv2.circle(facepaste, (nosex+nose_right[0], nosey+nose_right[1]), 5, (255, 0, 255))
				# 鼻の中間値
				nose_center = ((nose_left[0]+nose_right[0])/2, (nose_left[1]+nose_right[1])/2)
				print nose_center
				cv2.circle(facepaste, (nosex+nose_center[0], nosey+nose_center[1]), 5, (255, 0, 255))
				
				
				# 口の特徴点を取る
				# 口の特徴点描画
				mouse_left = (1000, 1000)	# 口左端
				mouse_right = (0, 0)		# 口右端
				mouse_up = (1000, 1000)		# 口上
				mouse_down = (0, 0)			# 口下
				# エッジの立っている部分を検索(左右の特徴点)
				for y in range(skin_mask.shape[0]) :
					for x in range(skin_mask.shape[1]) :
						if skin_mask[y][x] == 255 :
							#cv2.circle(facepasete)
							if x <= mouse_left[0] :
								mouse_left = (x, y)
							if x >= mouse_right[0]:
								mouse_right = (x, y)
				for y in range(skin_mask.shape[0]) :
					# 上下の特徴点を調べる
					if y < 0 or y >= skin_mask.shape[0] :
						continue
					if skin_mask[y][(mouse_left[0]+mouse_right[0])/2] == 255 :
						if y <= mouse_up[1] :
							mouse_up = ((mouse_left[0]+mouse_right[0])/2, y)
						if y >= mouse_down[1] :
							mouse_down = ((mouse_left[0]+mouse_right[0])/2, y)
				# 口の特徴点を描画
				cv2.circle(facepaste, (mousex+mouse_left[0], mousey+mouse_left[1]), 5, (255, 0, 255))
				cv2.circle(facepaste, (mousex+mouse_right[0], mousey+mouse_right[1]), 5, (255, 0, 255))
				cv2.circle(facepaste, (mousex+mouse_up[0], mousey+mouse_up[1]), 5, (255, 0, 255))
				cv2.circle(facepaste, (mousex+mouse_down[0], mousey+mouse_down[1]), 5, (255, 0, 255))
				
				# エッジの画像
				cv2.imshow("nose", nose)
				cv2.imshow("mouse", skin_mask)
				
				
			cv2.line(facepaste, (0, face_bi.shape[0]/2), (face_bi.shape[1], face_bi.shape[0]/2), (255, 255, 255))
			cv2.imshow("face", facepaste)
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