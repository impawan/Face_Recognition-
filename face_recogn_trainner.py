# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 11:58:12 2018

@author: impawan

"""

import os 
import cv2
import numpy as np
from PIL import Image

repo_path = "Face_Repo"

recgnr = cv2.face.LBPHFaceRecognizer_create()

def getImageLabel(path):
    Abslt_Image_paths = [os.path.join(path,img_name) for img_name in os.listdir(path)]
    labels = []
    training_faces=[]
    for Abslt_Image_path in Abslt_Image_paths:
        cur_face = Image.open(Abslt_Image_path)
        face_np = np.array(cur_face,'uint8')
        cur_label = int(str(Abslt_Image_path).split("_")[2])
        labels.append(cur_label)
        training_faces.append(face_np)
        cv2.imshow('scanning',face_np)
        cv2.waitKey(100)
        #np.array(labels)
    return labels , training_faces
    
labels , training_faces= getImageLabel(repo_path)    
recgnr.train(training_faces,np.array(labels))
recgnr.save("Face_Recognizer_trained_data/traindata.yml")
cv2.destroyAllWindows()