'''

This is a very simple recorder for pictures from a kinect camera.

It will be used to record images of an object which is located on 
a revolving table. This gives us for each object different views.
The recorder stores all images of an object in a pickle file which can
then be processed with sift in the create_db file

usage:
* draw a box around the object you want to record
* hit 'r'
* enter the name of the object and hit enter
'''

import cv
import OpenNIPythonWrapper as onipy
import numpy as np
import pickle
import time

# mouse callback
def on_mouse(event, x, y, flags, param):
    if param['draw']:
        param['pt2'] = (x, y)
    if event == cv.CV_EVENT_LBUTTONDOWN:
        param['pt1'] = (x, y)
        param['pt2'] = None
        param['draw'] = True
    if event == cv.CV_EVENT_LBUTTONUP:
        param['draw'] = False

print """program to record rbg and depth images of objects

this images can then be used to compute sift featues for a library.
Description of input Keys:
    n: New object to record
    s: Series of pictures will be recorded
    r: record a single picture
    w: write recorded images to disk

use the mouse to draw a rectangle about the object you want to record
"""

# settings
OPENNI_INITIALIZATION_FILE = "../config/BasicColorAndDepth.xml"
intervall = 1
t_max = 13

# initialization
KEY_ESC = 27
current_key = -1
data = {'pt1': None, 'pt2': None, 'draw': True}     # structure to communicate with mouse handler
store = {'image': [], 'depth': []}                  # this is how the data is saved in the end
record = False
t0 = None
obj_name = None
save_count = 0
t_last = 0
r_count = 0
cv.NamedWindow('image')
cv.SetMouseCallback('image', on_mouse, data)

# open the driver
g_context = onipy.OpenNIContext()
return_code = g_context.InitFromXmlFile(OPENNI_INITIALIZATION_FILE)

if return_code != onipy.XN_STATUS_OK:
    print "failed to initialize OpenNI"


try:

    # the image generators
    image_generator = onipy.OpenNIImageGenerator()
    g_context.FindExistingNode(onipy.XN_NODE_TYPE_IMAGE, image_generator)
    depth_generator = onipy.OpenNIDepthGenerator()
    g_context.FindExistingNode(onipy.XN_NODE_TYPE_DEPTH, depth_generator)
    width = depth_generator.XRes()
    height = depth_generator.YRes()

    # matrix headers and matrices for computation buffers
    current_image_frame = cv.CreateImageHeader(image_generator.Res(), cv.IPL_DEPTH_8U, 3)
    current_depth_frame = cv.CreateMatHeader(height, width, cv.CV_16UC1)


    while True:

        return_code = g_context.WaitAndUpdateAll()

        depth_data_raw = depth_generator.GetGrayscale16DepthMapRaw()
        cv.SetData(current_depth_frame, depth_data_raw)
        image_data_raw = image_generator.GetBGR24ImageMapRaw()
        cv.SetData(current_image_frame, image_data_raw)
        
        # draw the rectangle around the object
        if data['pt1'] != None and data['pt2'] != None:
            cv.Rectangle(current_image_frame, data['pt1'], data['pt2'], cv.Scalar(255, 255, 255))

        # finished recording
        if t0 != None and time.time() - t0 > t_max:
            print 'took %d images and store them in %s.pickle' % (len(store['image']), obj_name)
            pickle.dump(store, open('../out/%s.pickle' % obj_name, 'w'))
            record = False
            r_count = 0
            store = {'image': [], 'depth': []}
            t0 = None
                   
        # time to take a picture
        if record and time.time() - t_last > intervall:
            rect = cv.BoundingRect([data['pt1'], data['pt2']])
            sub = cv.GetSubRect(current_image_frame, rect)
            store['image'].append(np.asarray(sub))
            sub = cv.GetSubRect(current_depth_frame, rect)
            store['depth'].append(np.asarray(sub))
            t_last = time.time()
            r_count += 1
            print 'took picture at %f' % (t_last - t0) 

        # show the images
        cv.ShowImage( "image", current_image_frame )
        
        # wait for user input (keys)
        current_key = cv.WaitKey( 5 ) % 0x100
        if current_key == KEY_ESC:
            break
        elif current_key == ord('n'):
            obj_name = raw_input('please enter object name: ')
        elif current_key == ord('w'):
            print 'took %d images and store them in %s.pickle' % (len(store['image']), obj_name)
            pickle.dump(store, open('../out/%s.pickle' % obj_name, 'w'))
            store = {'image': [], 'depth': []}           
        elif current_key == ord('s'):
            print 'record series of pictures from rotating object'
            print 'and write it to disk'
            record = True
            t0 = time.time()
            t_last = t0
        elif current_key == ord('r'):
            print 'added an image to the list, write to disk with \'w\''
            rect = cv.BoundingRect([data['pt1'], data['pt2']])
            sub = cv.GetSubRect(current_image_frame, rect)
            store['image'].append(np.asarray(sub))
            sub = cv.GetSubRect(current_depth_frame, rect)
            store['depth'].append(np.asarray(sub))

finally:
    g_context.Shutdown()
    
