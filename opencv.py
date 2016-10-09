import numpy as np
import cv2
import os

images = []
labels = []

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def addFaces(name, dir):
    img = cv2.imread(dir)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        images.append(gray[y:y+h, x:x+w])
        labels.append(name)


for folders in os.walk('Pictures'):
    for folder in os.listdir('Pictures'):
        for file in os.walk('Pictures/' + folder):
            arr = file[2]
            for img in arr:
                if img != ".DS_Store":
                    print 'Pictures/' + folder + '/' + img
                    addFaces(folder, 'Pictures/' + folder + '/' + img)

recognizer = cv2.face.createLBPHFaceRecognizer()

recognizer.train(images, np.array(labels))

for file in os.walk('Pictures/Test'):
    print file[2]
    for name in arr:
        if name != ".DS_Store":
            img = cv2.imread('Pictures/Test/' + name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.imshow('img',img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                nbr_predicted, conf = recognizer.predict(gray[y: y + h, x: x + w])
                print "{} as {} with {} confidence".format(name, nbr_predicted, conf)
