import cv2
from src.network import GestureNetwork
import sys
import numpy as np

net = GestureNetwork()
net.load(sys.argv[1], load_labels = True)
labels = net.labels
print(labels)

webcam = cv2.VideoCapture(0)
ok, frame = webcam.read()

x1,y1 = 400,50
p1 = (x1, y1)
p2 = (x1 + 200, y1 + 200)

while ok:
    cv2.rectangle(frame, p1, p2, (255,0,0))
    cv2.imshow('fr', frame)
    cv2.waitKey(1)
    thumb = cv2.resize(frame[p1[1]:p2[1]+1, p1[0]:p2[0]+1], (64, 64))
    print(net.predict(thumb))

    ok, frame = webcam.read()