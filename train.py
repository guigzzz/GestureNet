num_classes = 4

from os.path import basename, dirname, join
import os
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from src.network import GestureNetwork
import sys

data_dict = {}
for root, dirs, files in os.walk(sys.argv[1]):
    if files != []:
        data_dict[basename(root)] = [join(root, f) for f in files]

x = []
y = []

classes = data_dict.keys()
input_res = 64
for i, (_, v) in enumerate(data_dict.items()):
    x += [cv2.resize(cv2.imread(p), (input_res, input_res)) for p in v]
    y += [i] * len(v) 


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3)

net = GestureNetwork()
net.build(num_classes, input_res)
hist = net.fit(X_train, Y_train)
print(net.evaluate(X_test, Y_test))
net.save('net3.hdf5', classes)