import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('ImageSource/Elon_Musk.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('ImageSource/Elon_MuskTest.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
# rectaangle 시작점 마지막 좌표
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLoc = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
# rectaangle 시작점 마지막 좌표
cv2.rectangle(imgTest, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeElon], encodeTest)
# distance가 높은경우 false 낮을경우 true
faceDis = face_recognition.face_distance([encodeElon], encodeTest)

print(results, faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('elon musk', imgElon)
cv2.imshow('elon test', imgTest)
cv2.waitKey(0)
