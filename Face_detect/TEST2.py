import sys
import cv2

imagePath = r'./people.jpg' 
cascPath = r'./haarcascade_frontalface_default.xml'

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
image = cv2.imread("people.jpg",1)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


faces = faceCascade.detectMultiScale(
       gray,
       scaleFactor = 1.1,
       minNeighbors = 5,
       minSize = (5, 5),
       flags = cv2.CASCADE_SCALE_IMAGE
       )


print ("Found {0} faces!".format(len(faces)))
       
for (x,y,w,h) in faces:
       cv2.rectangle(image,(x,y),(x+w,y+w),(0,255,0),2)
      
cv2.imshow("Faces found",image)
cv2.waitKey(0)

