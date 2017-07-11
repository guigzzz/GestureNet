import cv2
from skimage.io import imshow
import matplotlib.pyplot as plt
import os

webcam = cv2.VideoCapture(0)
ok, frame = webcam.read()


x1,y1 = 400,50

p1 = (x1, y1)
p2 = (x1 + 200, y1 + 200)
savedir = 'ims'
i = 0
while ok:
    cv2.rectangle(frame, p1, p2, (255,0,0))
    cv2.imshow('fr', frame)
    cv2.waitKey(1)
    cv2.imwrite(os.path.join(savedir, str(i) + '.png'), frame[p1[1]:p2[1]+1, p1[0]:p2[0]+1])
    i += 1
    ok, frame = webcam.read()


    