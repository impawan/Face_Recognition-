# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 22:18:25 2018

@author: impawan
"""

import cv2
from PIL import Image

    
def config_manager(name):
    config_file = open("config.config",'r+')
    lines = config_file.readlines()
    TempDict = {}
    
    if (len(lines) > 0):
        for line in lines:
            key,username = line.split(":")
            TempDict[key] = username   
        if (name in str(TempDict.values()).strip()):
            print ("user face samples is already present in the face repositiroy")
            print ("please enter new user name")
            name = input()
            config_manager(name)
        else:
             max_key=max(TempDict, key=int)
             print (max_key)
             new_key = int(max_key)+1
             TempDict[new_key]=name
             temp = str(new_key)+":"+name
             config_file.write("\n"+str(temp))
             config_file.close()
             return new_key
    else:
        config_file.write("1:"+name)
        config_file.close()
        return 1
    
face_cascade = cv2.CascadeClassifier("classfier.xml")

cap = cv2.VideoCapture(0)

name = input("enter the name of the person, no space is allowed >> ") 
user_key = config_manager(name)
sampleNo = 0
while True :
    ret , frame = cap.read()
    faceCount = 0
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    print(faces)
    if (x,y,w,h) in faces:
        sampleNo= sampleNo+1
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(125,255,0),2)
        cv2.imwrite("Face_Repo/face_"+str(user_key)+"_"+str(sampleNo)+".jpg",gray[y:y+h,x:x+w]) 
        cv2.waitKey(100)
    cv2.imshow("mirror",frame)    
    if(sampleNo>29):
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()   
        
        
 
    

    