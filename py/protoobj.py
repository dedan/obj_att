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

class Rect(object):
    def __init__(self, r):
        self.width = r[2]
        self.height = r[3]
        self.x1 = r[0]
        self.y1 = r[1]
        self.x2 = r[0] + self.width
        self.y2 = r[1] + self.height


class Obj(object):
    def __init__(self, cont):
        self.cont = cont
        self.color = random_color()
        self.count = -1
        self.id = random.randint(1, 2**31)
        self.frames = None
        self.desc = None
        self.ids = None
        box = cv.MinAreaRect2(cont)
        self.box_points = [(int(x), int(y)) for x, y in cv.BoxPoints(box)]
        
    @property
    def cont(self):
        return self._cont
    
    @cont.setter
    def cont(self, value):
        self._cont = value
        box = cv.MinAreaRect2(value)
        self.box_points = [(int(x), int(y)) for x, y in cv.BoxPoints(box)]
        
        
class SiftThread(threading.Thread):
    
    def __init__(self, width, height, q, stats, flann, meta):
        self.q = q
        self.stats = stats
        self.flann = flann
        self.meta = meta
        self.gray = cv.CreateMat(height, width, cv.CV_8UC1)
        self._stop = threading.Event()
        threading.Thread.__init__(self)
        
    def stop(self):
        self._stop.set()
    
    def run(self):
  
        while not self._stop.isSet():
            task  = self.q.get()
            
            if task != None:
                obj, image = task
        
                # TODO: vielleicht doch die minarearect benutzen, ausschneiden und drehen
                # or maybe not: http://stackoverflow.com/questions/4230572/cvbox2d-processing/4535459#4535459
                rect = cv.BoundingRect(obj.cont)
                siftimage = siftfastpy.Image(rect[2], rect[3])
                cv.CvtColor(image, self.gray, cv.CV_BGR2GRAY)
                gnp = np.asarray(cv.GetSubRect(self.gray, rect))
                siftimage.SetData(gnp)
#                t0 = time.time()

                # compute keypoints and time how long it takes
                frames,desc = siftfastpy.GetKeypoints(siftimage)
#                self.stats.append((rect[2]*rect[3], time.time() - t0))

                # compute feature vector
                tmp = np.concatenate((frames[:,2:4], desc), axis=1).astype('float64')
                
                # search in the flann tree for the feature vectors
                n = 2
                thr = 1.5
                result, dists = self.flann.nn_index(tmp, n, checks=32)
                
                # ismatch contains the indices in the testset for which a match is found
                ismatch = dists[:,1] > thr * dists[:,0]
                               
                # meta contains the index to object-ID mapping
                obj.ids = []
                for i, res in enumerate(result):
                    if ismatch[i]:
                        obj.ids.append(self.meta[res[0]][0])
                print obj.ids
#                obj.ids = [self.meta[res][0] for i, res in enumerate(result) if ismatch[i]]

                
                # transfer keypoints back to full frame coordinates
                frames[:,0] += rect[0]
                frames[:,1] += rect[1] 
                obj.frames = frames
                obj.desc = desc
