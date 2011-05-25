

import cv

bla = cv.CreateMat(100, 100, cv.CV_8UC1)

data = {'pt1': None, 'pt2': None, 'draw': True}

def mouse_call(event, x, y, flags, param):
    
    if param['draw']:
        param['pt2'] = (x, y)
    
    if event == cv.CV_EVENT_LBUTTONDOWN:
        param['pt1'] = (x, y)
        param['pt2'] = None
        param['draw'] = True

    if event == cv.CV_EVENT_LBUTTONUP:
        param['draw'] = False


cv.NamedWindow('test')
cv.SetMouseCallback('test', mouse_call, data)

   
while True:
    cv.SetZero(bla)
    if data['pt1'] != None and data['pt2'] != None:
        cv.Rectangle(bla, data['pt1'], data['pt2'], cv.Scalar(255))

    cv.ShowImage('test', bla)

    current_key = cv.WaitKey( 5 ) % 0x100
    if current_key == 27:
        break




#import pyflann
#from numpy import *
#from numpy.random import *
#dataset = rand(10000, 128)
#testset = rand(1000, 128)
#flann = pyflann.FLANN()
#result,dists = flann.nn(dataset,testset,5,algorithm="kmeans", branching=32, iterations=7, checks=16);
#print result
#print dists
