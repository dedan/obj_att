import cv
import numpy as np
import pickle
from my_classes import Obj
import pylab as plt

path = '/home/dedan/obj_att/out/'

p1 = pickle.load(open(path + 'conts_0.pickle'))
p2 = pickle.load(open(path + 'conts_1.pickle'))
p3 = pickle.load(open(path + 'conts_2.pickle'))


res = np.zeros((len(p1), len(p3)))

for i, o1 in enumerate(p1):
    for j, o2 in enumerate(p3):
        bla  = cv.CreateMat(480, 640, cv.CV_8UC1)
        blub = cv.CreateMat(480, 640, cv.CV_8UC1)
        cv.SetZero(bla)
        cv.SetZero(blub)
        cv.FillPoly(bla, [o1.cont], 1)
        cv.FillPoly(blub, [o2.cont], 1)
        area1 = np.sum(bla)
        area2 = np.sum(np.asarray(bla)*np.asarray(blub))
        res[i,j] = area2 > 0.5 * area1

plt.figure()
plt.imshow(res, interpolation='nearest')
plt.colorbar()
plt.show()
