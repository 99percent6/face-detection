import cv2 

img = cv2.imread("./img/chris.jpg")
face_recog = cv2.CascadeClassifier('./filters/haarcascade_frontalface_default.xml')
face_result = face_recog.detectMultiScale(img, scaleFactor=2, minNeighbors=5) 

if len(face_result) != 0:
	for index, (x,y,w,h) in enumerate(face_result):
		img = img[x:y+h]
		cv2.imwrite(f'./img/chris_{index}.jpg', img)
