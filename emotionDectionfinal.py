import cv2 
from deepface import DeepFace 
import numpy as np
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,     
        scaleFactor=1.1,
        minNeighbors=5,     
    )
    

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
        
        try:
            result= DeepFace.analyze(frame,actions=['emotion'])
            #print(analyze[0]['dominant_emotion'])
            print(result[0]["dominant_emotion"][:])
        except:
            print("No face")
    cv2.imshow('cap',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()
        