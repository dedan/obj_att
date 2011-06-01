from protoobj import random_color
import cv
import math
import numpy as np
import pickle
import siftfastpy
import pyflann

n_colors = 100
n = 2
thr = 1.5

karton = pickle.load(open('../out/karton.pickle'))

# go from most distant to closest picture
karton['image'].reverse()
karton['depth'].reverse()

# create a list of random colors
color_tab = [random_color() for i in range(n_colors)]

karton['features'] = []
karton['frames'] = []

for i, image in enumerate(karton['image']):
        
    siftimage = siftfastpy.Image(image.shape[1], image.shape[0])
    gray = cv.CreateMat(image.shape[0], image.shape[1], cv.CV_8UC1)
    cv.CvtColor(image, gray, cv.CV_BGR2GRAY)
    siftimage.SetData(np.asarray(gray))
    
    # compute keypoints and time how long it takes
    frames, desc = siftfastpy.GetKeypoints(siftimage)
    
    depth = np.zeros((frames.shape[0], 1))
    for j in range(frames.shape[0]):
        depth[j] = karton['depth'][i][math.floor(frames[j, 1]), math.floor(frames[j, 0])]
    
    # compute feature vector
    tmp = np.concatenate((frames[:, 2:4], desc, depth), axis=1).astype('float64')
    karton['features'].append(tmp)
    karton['frames'].append(frames)


flann = pyflann.FLANN()
# FIXME: currently without depth information because not aligned
params = flann.build_index(karton['features'][0], target_precision=0.95)

# plot the keypoints
for j in range(karton['frames'][0].shape[0]):
    cv.Rectangle(karton['image'][0],
                 (int(karton['frames'][0][j, 0]) - 1, int(karton['frames'][0][j, 1]) - 1),
                 (int(karton['frames'][0][j, 0]) + 1, int(karton['frames'][0][j, 1]) + 1),
                 color_tab[j])

for i in range(1,4):
    result, dists = flann.nn_index(karton['features'][i], n, checks=32)
    for j in range(dists.shape[0]):
        if dists[j,1] > thr * dists[j,0]:
            cv.Rectangle(karton['image'][i],
                         (int(karton['frames'][i][j, 0]) - 1, int(karton['frames'][i][j, 1]) - 1),
                         (int(karton['frames'][i][j, 0]) + 1, int(karton['frames'][i][j, 1]) + 1),
                         color_tab[result[j,0]])
            
for i, image in enumerate(karton['image']):
    cv.ShowImage('%d' % i, image)
    

cv.WaitKey(0)
