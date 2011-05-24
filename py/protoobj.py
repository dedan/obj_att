'''
Created on 18 May 2011

@author: dedan
'''

import random
import cv
import siftfastpy
import numpy as np
import threading
import time

def random_color():
    """
    Return a random color
    """
    icolor = random.randint (0, 0xFFFFFF)
    return cv.Scalar (icolor & 0xff, (icolor >> 8) & 0xff, (icolor >> 16) & 0xff)

class Obj(object):
    def __init__(self, cont):
        self.cont = cont
        self.color = random_color()
        self.count = -1
        self.id = random.randint(1, 2**31)
        self.frames = None
        self.desc = None
        box = cv.MinAreaRect2(cont)
        self.box_points = [(int(x), int(y)) for x, y in cv.BoxPoints(box)]
        
        # TODO: when contour changes also change the box_points
    @property
    def cont(self):
        return self._cont
    @cont.setter
    def cont(self, value):
        self._cont = value
        box = cv.MinAreaRect2(value)
        self.box_points = [(int(x), int(y)) for x, y in cv.BoxPoints(box)]
        
        
class SiftThread(threading.Thread):
    
    def __init__(self, image, obj):
        self.image = image
        self.obj = obj
        threading.Thread.__init__(self)
    
    def run(self):
        starttime = time.time()
        
        # TODO: die nur ein mal erzeugen bei init
        gray = cv.CreateMat(480, 640, cv.CV_8UC1)
        
        # TODO: vielleicht doch die minarearect benutzen, ausschneiden und drehen
        rect = cv.BoundingRect(self.obj.cont)
        siftimage = siftfastpy.Image(rect[2], rect[3])
        cv.CvtColor(self.image, gray, cv.CV_BGR2GRAY)
        gnp = np.asarray(cv.GetSubRect(gray, rect))
        siftimage.SetData(gnp)
        
        print 'initialization in: %fs' % (time.time()-starttime)
        
        frames,desc = siftfastpy.GetKeypoints(siftimage)
        
        # TODO: descriptoren wieder in frame coordinaten rechnen
        self.obj.frames = frames
        self.obj.desc = desc
        print '%d  keypoints found in %fs'%(frames.shape[0],time.time()-starttime)
    