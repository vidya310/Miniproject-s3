import numpy as np
import face_recognition
import os

path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
#print(len(encodeListKnown))
print('Encoding complete')

cap = cv2.VideoCapture(0)

while True:
    success, img =cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)[0]
    encode=face_recognition.face_encodings(img)[0]

#faceLoc=face_recognition.face_locations(imgVidya)[0]
#encodeVidya=face_recognition.face_encodings(imgVidya)[0]
#cv2.rectangle(imgVidya,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

#imgVidya = face_recognition.load_image_file('ImagesBasic/my photo Test.jpg')
#imgVidya = cv2.cvtColor(imgVidya,cv2.COLOR_BGR2RGB)
#imgTest = face_recognition.load_image_file('ImagesBasic/my photo Test.jpg')
