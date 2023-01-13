# This project was done in google colab, because face recognition module was not getting installed in my system
# link to google colab page is given below
# https://drive.google.com/file/d/17fpoPlgqC7m4q7zTfRMiCx0qXFhW7LAt/view?usp=sharing

import face_recognition
import cv2
import os
from google.colab.patches import cv2_imshow

def read_img(path, width = 320):
  img = cv2.imread(path)
  (h, w) = img.shape[:2]
  height = int(h * (width / float(w)))
  return cv2.resize(img, (width, height))

known = []
names = []
known_dir = "known"
for file in os.listdir(known_dir):
  img = read_img(f"{known_dir}/{file}")
  known.append(face_recognition.face_encodings(img)[0])
  names.append(file.split('.')[0])

unknown_dir = "unknown"
for file in os.listdir(unknown_dir):
  img = read_img(f"{unknown_dir}/{file}")
  unknown = face_recognition.face_encodings(img)[0]
  results = face_recognition.compare_faces(known, unknown)
  for i in range(len(results)):
    if results[i]:
      name = names[i]
      (top,right,bottom,left) = face_recognition.face_locations(img)[0]
      cv2.rectangle(img, (left,top), (right,bottom), (0,0,255), 2)
      cv2.putText(img, name, (left+2,bottom+20), cv2.FONT_HERSHEY_PLAIN, 1,
                  (255,255,255), 1)
      cv2_imshow(img)