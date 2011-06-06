import cv
import math
import numpy as np
import pickle
import siftfastpy
import pyflann
import pylab as plt

'''
ideas of this test

paarweise die bilder anschauen (1-2, 2-3, 3-4)
alle sift feature die im groesseren gefunden werden gegen die im kleineren matchen
immer wenn es einen match gab (mit unseren 1.5 kriterium), dann berechne ich den quotienten der beiden depth/scale produkte
eine verteilung der quotienten plotten

'''

n = 2
thr = 1.5

karton = pickle.load(open('../out/karton.pickle'))

# go from most distant to closest picture
karton['image'].reverse()
karton['depth'].reverse()
karton['features'] = []
karton['frames'] = []
karton['ratios'] = [[],[],[]]

# compute feature vectors for all images
for i, image in enumerate(karton['image']):
        
    siftimage = siftfastpy.Image(image.shape[1], image.shape[0])
    gray = cv.CreateMat(image.shape[0], image.shape[1], cv.CV_8UC1)
    cv.CvtColor(image, gray, cv.CV_BGR2GRAY)
    siftimage.SetData(np.asarray(gray))
    
    # compute keypoints
    frames, desc = siftfastpy.GetKeypoints(siftimage)

    # compute feature vector    
    depth = np.zeros((frames.shape[0], 1))
    for j in range(frames.shape[0]):
        depth[j] = karton['depth'][i][math.floor(frames[j, 1]), math.floor(frames[j, 0])]
    tmp = np.concatenate((frames[:, 2:4], desc, depth), axis=1).astype('float64')
    
    karton['features'].append(tmp)
    karton['frames'].append(frames)

flann = pyflann.FLANN()

for i in range(len(karton['features'])-1):
    
    # build flann index against features from smaller picture
    flann.build_index(karton['features'][i], target_precision=0.95)

    result, dists = flann.nn_index(karton['features'][i+1], n, checks=32)
    
    # ismatch contains the indices in the testset for which a match is found
    ismatch = dists[:,1] > thr * dists[:,0]
    
    mean_dist1 = np.mean(karton['features'][i][karton['features'][i][:,-1]>0,-1])
    mean_dist2 = np.mean(karton['features'][i+1][karton['features'][i+1][:,-1]>0,-1])
                   
    # meta contains the index to object-ID mapping
    for j, res in enumerate(result):
        if ismatch[j]:
            prod1 = karton['features'][i][res[0],2] * mean_dist1
            prod2 = karton['features'][i+1][j,2] * mean_dist2
            karton['ratios'][i].append( prod1 / prod2)
    plt.figure()
    print len(karton['ratios'][i])
    plt.plot(karton['ratios'][i])
#plt.show()