
import time
import siftfastpy
import numpy as np
import threading
from my_classes import Obj
import pickle
import cv

class SiftThread(threading.Thread):
    
    def __init__(self, image, obj):
        self.image = image
        self.obj = obj
        threading.Thread.__init__(self)
    
    def run(self):
        starttime = time.time()
        
        gray = cv.CreateMat(480, 640, cv.CV_8UC1)
        im = cv.LoadImage(self.image)    
        rect = cv.BoundingRect(self.obj.cont)
        siftimage = siftfastpy.Image(rect[2], rect[3])
        cv.CvtColor(im, gray, cv.CV_BGR2GRAY)
        gnp = np.asarray(cv.GetSubRect(gray, rect))
        siftimage.SetData(gnp)
        
        print 'initialization in: %fs' % (time.time()-starttime)
        
        frames,desc = siftfastpy.GetKeypoints(siftimage)
        self.obj.frames = frames
        self.obj.desc = desc
        print '%d  keypoints found in %fs'%(frames.shape[0],time.time()-starttime)


objects = pickle.load(open('../out/objects_0.pickle'))
tmp = None
for obj in objects:
    obj.frames = None
    obj.desc = None
    if obj.count > 20:
        tmp = obj


print [bla.frames for bla in objects]
t = SiftThread('../out/image_0.jpg', tmp)
t.start()
#objects.remove(tmp)

#for i in range(3):
print "fertig"
t.join()

print [bla.frames for bla in objects]

#print '%d %d'%(desc.shape[0],desc.shape[1])
#for i in xrange(frames.shape[0]):
#    print '%d %d %f %f'%(frames[i,1],frames[i,0],frames[i,3],frames[i,2])
#    s = ''
#    for j,d in enumerate(desc[i]):
#        s += str(min(255,int(d*512.0))) + ' '
#        if np.mod(j,16) == 15:
#            s += '\n'
#    print s