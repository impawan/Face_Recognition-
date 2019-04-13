
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 21:16:30 2018

@author: ACER
"""

import cv2
import numpy as np


def id_to_name(id):
    config_file = open("config.config",'r+')
    lines = config_file.readlines()
    TempDict = {}
    for line in lines:
        key,username = line.split(":")
        TempDict[key] = username
    usertemp = TempDict[str(id)]
    return usertemp.strip()   

face_cascade = cv2.CascadeClassifier("classfier.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
recgnr = cv2.face.LBPHFaceRecognizer_create()

recgnr.read("Face_Recognizer_trained_data/traindata.yml")
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX
while True :
    ret , frame = cap.read()
    faceCount = 0
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        faceCount=int(faces.shape[0])
        eyes = eye_cascade.detectMultiScale(gray[y:y+h, x:x+w])
        for (a,b,c,d) in eyes:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
            cv2.rectangle(frame[y:y+h, x:x+w],(a,b),(a+c,b+d),(0,255,0),1)
            id,confidence =recgnr.predict(gray[y:y+h,x:x+w])
            name = id_to_name(id)
            cv2.putText(frame,name+str(confidence),(x,y+h),font,2,(0,0,255),2,cv2.LINE_4)
#        if (confidence<70):  
#            name = id_to_name(id)
#            cv2.putText(frame,name+str(confidence),(x,y+h),font,2,(0,0,255),2,cv2.LINE_4)
#            cv2.rectangle(frame[y:y+h, x:x+w],(a,b),(a+c,b+d),(0,255,0),1)
    cv2.imshow('Face Detection',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
cap.release()
cv2.destroyAllWindows()   
        
        
 
    

    