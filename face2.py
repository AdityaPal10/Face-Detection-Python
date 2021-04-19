import cv2,os
#we will need the os to accress the file list in out dataset folder
import numpy as np
from PIL import Image
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    #Load the training images from dataset folder
    #capture the faces and Id from the training images
    #Put them In a List of Ids and FaceSamples  and return it
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    #this will get the path of each images in the folder.
    faces=[]
    Ids=[]
    #create two lists for faces and Ids to store the faces and Ids
    for imagePath in imagePaths:
        pilImage=Image.open(imagePath).convert('L')
        #the image and converting it to gray scale
        imageNp=np.array(pilImage,'uint8')
        #PIL image converted to numpy array
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        #split the image name into ()user, ()id, ()samplenumber
        faces.append(imageNp)
        Ids.append(Id)
        #detector is extracting the faces and appending them in the faceSamples list with the Id
        cv2.imshow('training',imageNp)
        cv2.waitKey(10)
    return faces,Ids

faces,Ids = getImagesAndLabels('FaceDataBase')
recognizer.train(faces, np.array(Ids))
recognizer.write('trainner/trainner.yml')
