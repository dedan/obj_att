
import numpy as np
import cv
from guppy import hpy
import pickle
h = hpy()


height = 500
width = 400

#mat = cv.CreateMat(height, width, cv.CV_8UC1)
#cv.SetZero(mat)

mat = pickle.load(open('/home/dedan/obj_att/out/30mask.m'))
print np.shape(mat)

for i in range(100000):
    l = []
    storage = cv.CreateMemStorage(0)
    conts = cv.FindContours(mat, storage, cv.CV_RETR_EXTERNAL, cv.CV_CHAIN_APPROX_SIMPLE)
#    del conts
#    while conts:
#        l.append(list(conts))
#        conts = conts.h_next()

print h.heap()