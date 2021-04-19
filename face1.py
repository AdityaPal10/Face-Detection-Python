import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

id=input('Enter user id')
sampleNum=0
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        sampleNum=sampleNum+1
        cv2.imwrite('FaceDataBase/User.'+str(id)+'.'+str(sampleNum)+'.jpg',gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.waitKey(100)
    cv2.imshow('img',img)
    cv2.waitKey(1)
    if(sampleNum>20):
        break


cap.release()
cv2.destroyAllWindows()
