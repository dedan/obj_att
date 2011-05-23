'''
Created on 18 May 2011

@author: dedan
'''

import random
import cv

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
    