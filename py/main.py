import my_classes
__author__ = 'dedan'


import cv
import OpenNIPythonWrapper as onipy
import numpy as np
import pickle
from guppy import hpy
import time
import objgraph
import inspect, random


# for making random numbers
from my_classes import Obj
from my_classes import random_color

h = hpy()

# some constants
OPENNI_INITIALIZATION_FILE = "/home/dedan/Downloads/onipy/OpenNIPythonWrapper/OpenNIConfigurations/BasicColorAndDepth.xml"
KEY_ESC = 27
n_bins = 600
max_range = 6000
min_range = 800
hist_height = 64
k_width = 5
perc = 0.2
max_dist = 4000
n_colors = 100

# create a list of random colors
color_tab = [random_color() for i in range(n_colors)]

# open the driver
g_context = onipy.OpenNIContext()
return_code = g_context.InitFromXmlFile(OPENNI_INITIALIZATION_FILE)

if return_code != onipy.XN_STATUS_OK:
    print "failed to initialize OpenNI"


try:

    # initialization stuff
    save_count = 0
    objects = []
    current_key = -1
    image_generator = onipy.OpenNIImageGenerator()
    return_code = g_context.FindExistingNode(onipy.XN_NODE_TYPE_IMAGE, image_generator)
    current_image_frame = cv.CreateImageHeader(image_generator.Res(), cv.IPL_DEPTH_8U, 3)

    depth_generator = onipy.OpenNIDepthGenerator()
    return_code = g_context.FindExistingNode(onipy.XN_NODE_TYPE_DEPTH, depth_generator)
    width = depth_generator.XRes()
    height = depth_generator.YRes()
    current_depth_frame = cv.CreateMatHeader(height, width, cv.CV_16UC1)
    for_thresh = cv.CreateMat(height, width, cv.CV_32FC1)
    
    # create some matrices for drawing
    hist_img = cv.CreateMat(hist_height, width, cv.CV_8UC3)
    out = cv.CreateMat(height + hist_height, width, cv.CV_8UC3)
    contours = cv.CreateMat(height, width, cv.CV_8UC3)
    min_thresh = cv.CreateMat(height, width, cv.CV_8UC1)
    max_thresh = cv.CreateMat(height, width, cv.CV_8UC1)
    and_thresh = cv.CreateMat(height, width, cv.CV_8UC1)

    obj_draw = np.zeros((height, width))
    cont_draw = np.zeros((height, width))

    time_sum = 0
    loop_c = 0
    
    font = cv.InitFont(cv.CV_FONT_HERSHEY_PLAIN, 1, 1, 0, 1, 8) #Creates a font
    elem = cv.CreateStructuringElementEx(8, 8, 4, 4, cv.CV_SHAPE_RECT)
    storage = cv.CreateMemStorage(0)




    while True:

        return_code = g_context.WaitAndUpdateAll()

        # get the images
        depth_data_raw = depth_generator.GetGrayscale16DepthMapRaw()
        cv.SetData(current_depth_frame, depth_data_raw)
        image_data_raw = image_generator.GetBGR24ImageMapRaw()
        cv.SetData(current_image_frame, image_data_raw)
        
        t0 = time.time()
        
        # initialize matrices
        depth = np.asarray(current_depth_frame)
        cv.SetZero(hist_img)
        cv.SetZero(out)
        cv.SetZero(contours)
    
        # compute and smooth histogram
        hist, _ = np.histogram(depth, n_bins, range=(min_range, max_range), normed=False)
        hist = np.convolve(hist, np.ones(k_width) / k_width, 'same')
        max_hist = np.max(hist)

        # histogram clustering
        start = 0
        end = 0
        c = 1
        conts_list = []

        for i in range(max_dist /10):
            cur_value = hist[i]
            next_value = hist[i+1]
            
            # still walking up the hill
            if cur_value > hist[end]:
                end = i

            # arrived in a valley
            if ((cur_value < perc * hist[end]) & (1.1 * cur_value < next_value)):
                
                cv.Convert(current_depth_frame, for_thresh)


                cv.Threshold(for_thresh, min_thresh, start*10, 255, cv.CV_THRESH_BINARY)
                cv.Threshold(for_thresh, max_thresh, i*10, 255, cv.CV_THRESH_BINARY_INV)
                cv.And(min_thresh, max_thresh, and_thresh)
                

                # cut out a certain depth layer
#                mask = (depth > start * 10) & (depth < i * 10)               
#                mask = mask.astype(np.uint8) * 255
#                cv.Convert(depth, bla)
#                mask = cv.fromarray(mask)
                # erode the layer and find contours
                cv.Erode(and_thresh, and_thresh, elem)
#                conts = cv.FindContours(mask, storage, cv.CV_RETR_EXTERNAL, cv.CV_CHAIN_APPROX_SIMPLE)
                
                # collect all interesting contours in a list
#                while conts:
#                    if len(conts) > 50 and cv.ContourArea(conts) > 500:
#                        conts_list.append(list(conts))
#                    conts = conts.h_next()
#                del conts
                
                # prepare for search for next hill
                start = i
                end = i
                c = c + 1
                
            # draw current value in histogram
#            pts = [(i, hist_height),
#                   (i, hist_height),
#                   (i, int(hist_height-next_value*hist_height/max_hist)),
#                   (i, int(hist_height-cur_value*hist_height/max_hist))]
#            cv.FillConvexPoly(hist_img, pts, color_tab[c])


                
        # pickle the contours
        if current_key == ord('c'):
            with open('/home/dedan/obj_att/out/conts_%d.pickle' % save_count, 'w') as f:
                pickle.dump(conts_list, f)
                save_count = save_count +1
        
#        print len(conts_list)
        # iterate over tracked objects
#        for obj in objects:
#            found = False
#            
#            # when not seen n times, remove object
#            if obj.count < -3:
#                objects.remove(obj)
#                continue
#            
#            # draw the contour in an image for comparison
#            obj_draw[:] = 0
#            cv.FillPoly(obj_draw, [obj.cont], 1)
#            area = np.dot(np.ravel(obj_draw), np.ravel(obj_draw))
#            
#            # check for each object whether we see the contour again in this frame
#            for cont in conts_list:
#                cont_draw[:] = 0
#                cv.FillPoly(cont_draw, [cont], 1)
#                
#                # we found it
#                if np.dot(np.ravel(obj_draw), np.ravel(cont_draw)) > 0.5 * area:
#                    obj.cont = cont            # update the contour to track movements
#                    obj.count = obj.count + 1  # this helps new objects to recover from negative init
#                    found = True
#                    conts_list.remove(cont)     
#                    break
#            if not found:
#                if obj.count > 0:
#                    obj.count = 0               # once not seen -> immediately on delete list
#                else:
#                    obj.count = obj.count -1    # few times not seen -> it becomes only worse
#        
#        # new objects get a chance in the object list
#        for cont in conts_list: 
#            objects.append(Obj(cont))
#
#        # print the contours and a box around them
##        print len(objects), len(conts_list)
#        for obj in objects:
#            if obj.count < 0:
#                continue
#            cv.FillPoly(contours, [obj.cont], obj.color)
#            box = cv.MinAreaRect2(obj.cont)
#            b_points = [(int(x), int(y)) for x, y in cv.BoxPoints(box)]
#            for j in range(4):
#                cv.Line(contours, b_points[j], b_points[(j+1)%4], cv.Scalar(0,255,0))

#        for cont in conts_list:
#            cv.FillPoly(contours, [cont], my_classes.random_color())
#            box = cv.MinAreaRect2(cont)
#            b_points = [(int(x), int(y)) for x, y in cv.BoxPoints(box)]
#            for j in range(4):
#                cv.Line(contours, b_points[j], b_points[(j+1)%4], cv.Scalar(0,255,0))
            
                
        time_sum += time.time() - t0
        loop_c += 1
        
#        objgraph.show_growth()
        if (loop_c % 30) == 0: 
            objgraph.show_chain(objgraph.find_backref_chain(
                                                        random.choice(objgraph.by_type('tuple')),
                                                        inspect.ismodule),
                                                        filename='chain%d.png' % loop_c)

        
        
        

        
        # TODO: ueber frames mitteln (entweder nur im histogramm, vielleicht aber auch ueber ganze frames (mit discount factor)
        
                
                




        # output images
        outroi = (0, hist_height, width, height)
        a = cv.GetSubRect(out, outroi)
        cv.Copy(contours, a)
        histroi = (0, 0, width, hist_height)
        a = cv.GetSubRect(out, histroi)
        cv.Copy(hist_img, a)
        
        cv.PutText(out,"mean: %f" % (time_sum / loop_c), (0,height+hist_height-10),font, cv.Scalar(255,255,255)) #Draw the text


        # show the images
        cv.ShowImage( "Image Stream", current_image_frame )
        cv.ShowImage( "Depth Stream", current_depth_frame)
        cv.ShowImage("conts", out)

        # wait for user input (keys)
        current_key = cv.WaitKey( 5 ) % 0x100
        if current_key == KEY_ESC:
            print h.heap()
            break

finally:

    g_context.Shutdown()