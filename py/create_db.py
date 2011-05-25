'''
create a database of feature vectors from kinect images

the images (rgb and depth) are stored in pickle files. On this
images a feature vector is computed which consists of:
1-2: scale and orientation
3-130: sift descriptor
131: depth

it puts the feature vectors in a big table and also creates a list
which can be used to get object name or view angle for a certain feature vector

'''

import os
import glob
import pickle

import cv
import numpy as np
import siftfastpy

input_path = '../out/'
res = {'features': np.zeros((0, 131)), 'meta': []}
id = 0

f_list = glob.glob(input_path + '*.pickle')
print 'found the following files: '
print '\n'.join(['\t' + os.path.basename(f) for f in f_list])

for f in f_list:
    
    # load a pickled object (images)
    store = pickle.load(open(f))
    name = os.path.splitext(os.path.basename(f))[0]
    n_views = len(store['image'])
    
    # iterate over the views
    print 'working on: %s (%d views)' % (name, n_views)
    for view in range(n_views):
        
        # read in and convert to grayscale
        img = cv.fromarray(store['image'][view])
        gray = cv.CreateMat(img.rows, img.cols, cv.CV_8UC1)
        cv.CvtColor(img, gray, cv.CV_BGR2GRAY)
        
        # copy to siftfast structure and compute keypoints  
        siftimage = siftfastpy.Image(img.cols, img.rows)
        siftimage.SetData(np.asarray(gray))
        frames,desc = siftfastpy.GetKeypoints(siftimage)
        
        # get depth information and concatenate all to one feature vector
        n_keys = np.shape(frames)[0]
        depth = np.zeros((n_keys,1))
        for j in range(n_keys):
            depth[j] = store['depth'][view][int(frames[j,1]), int(frames[j,0])]
        tmp = np.concatenate((frames[:,2:4], desc, depth), axis=1)
        
        # add it to final data structure
        res['features'] = np.concatenate((res['features'], tmp))
        res['meta'] += [[id, name, view]] * n_keys
    id += 1

# store the results
print 'found %d features in %d files' % (len(res['features']), len(f_list))
out_file = os.path.join(input_path, 'pickled.db')
pickle.dump(res, open(out_file, 'w'))
print 'wrote the database to: %s' % out_file
