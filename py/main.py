
import cv
import OpenNIPythonWrapper as onipy
import numpy as np
import pickle
import time
import siftfastpy
import Queue
import pylab as plt

from protoobj import Obj
from protoobj import random_color
from protoobj import SiftThread


# some constants
OPENNI_INITIALIZATION_FILE = "../config/BasicColorAndDepth.xml"
KEY_ESC = 27
n_bins = 600
max_range = 6000
min_range = 800
hist_height = 64
k_width = 5
perc = 0.2
max_dist = 4000
n_colors = 100
cont_length = 50
cont_area = 500
n_sift = 20
n_forget = -3
n_threads = 5

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
    stats = []
    current_key = -1
    timing = {'t_draw':0, 't_histo':0, 't_track':0, 't_pre':0}
    loop_c = 0
    font = cv.InitFont(cv.CV_FONT_HERSHEY_PLAIN, 1, 1, 0, 1, 8) #Creates a font
    elem = cv.CreateStructuringElementEx(8, 8, 4, 4, cv.CV_SHAPE_RECT)
    storage = cv.CreateMemStorage(0)
    
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
    for_thresh = cv.CreateMat(height, width, cv.CV_32FC1)
    min_thresh = cv.CreateMat(height, width, cv.CV_8UC1)
    max_thresh = cv.CreateMat(height, width, cv.CV_8UC1)
    and_thresh = cv.CreateMat(height, width, cv.CV_8UC1)
    gray = cv.CreateMat(height, width, cv.CV_8UC1)
    obj_draw = np.zeros((height, width))
    cont_draw = np.zeros((height, width))
    
    # create some matrices for drawing
    hist_img = cv.CreateMat(hist_height, width, cv.CV_8UC3)
    out = cv.CreateMat(height + hist_height, width, cv.CV_8UC3)
    contours = cv.CreateMat(height, width, cv.CV_8UC3)
    
    # start the sifting threadpool
    sift_pool = Queue.Queue(0)
    threads = []
    for i in range(n_threads):
        t = SiftThread(width, height, sift_pool, stats)
        t.setDaemon(True)
        t.start()
        threads.append(t)

    while True:

        return_code = g_context.WaitAndUpdateAll()

        # get the images
        # TODO: ueber frames mitteln (entweder nur im histogramm, vielleicht aber auch ueber ganze frames (mit discount factor)
        depth_data_raw = depth_generator.GetGrayscale16DepthMapRaw()
        cv.SetData(current_depth_frame, depth_data_raw)
        cv.Convert(current_depth_frame, for_thresh)
        image_data_raw = image_generator.GetBGR24ImageMapRaw()
        cv.SetData(current_image_frame, image_data_raw)
        
        # initialize matrices for drawing and start timing
        t0 = time.time()
        cv.SetZero(hist_img)
        cv.SetZero(out)
        cv.SetZero(contours)
        
        # compute and smooth histogram
        depth = np.asarray(current_depth_frame)
        hist, _ = np.histogram(depth, n_bins, range=(min_range, max_range), normed=False)
        hist = np.convolve(hist, np.ones(k_width) / k_width, 'same')
        max_hist = np.max(hist)
        timing['t_histo'] += time.time() - t0

        # histogram clustering
        start, end = 0, 0
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

                # cut out a certain depth layer                
                cv.Threshold(for_thresh, min_thresh, start*10, 255, cv.CV_THRESH_BINARY)
                cv.Threshold(for_thresh, max_thresh, i*10, 255, cv.CV_THRESH_BINARY_INV)
                cv.And(min_thresh, max_thresh, and_thresh)
                
                # erode the layer and find contours
                cv.Erode(and_thresh, and_thresh, elem)
                conts = cv.FindContours(and_thresh, storage, cv.CV_RETR_EXTERNAL, cv.CV_CHAIN_APPROX_SIMPLE)
                
                # collect all interesting contours in a list
                while conts:
                    if len(conts) > cont_length and cv.ContourArea(conts) > cont_area:
                        conts_list.append(list(conts))
                    conts = conts.h_next()
                
                # prepare for search for next hill
                start, end = i, i
                c += 1
                
               
            # draw current value in histogram
            pts = [(i, hist_height),
                   (i, hist_height),
                   (i, int(hist_height-next_value*hist_height/max_hist)),
                   (i, int(hist_height-cur_value*hist_height/max_hist))]
            cv.FillConvexPoly(hist_img, pts, color_tab[c])
          
        # time the histogram clustering  
        timing['t_pre'] += time.time() - t0

        # pickle the contours and save images
        if current_key == ord('s'):
            scaled = cv.CreateMat(height, width, cv.CV_8UC1)
            cv.ConvertScale(current_depth_frame, scaled, 0.05)
            cv.SaveImage('../out/image_%d.jpg' % save_count, current_image_frame)
            cv.SaveImage('../out/depth_%d.jpg' % save_count, scaled)

            with open('../out/objects_%d.pickle' % save_count, 'w') as f:
                pickle.dump(objects, f)
            save_count = save_count +1
        
        
        # iterate over tracked objects
        sift = False
        for obj in objects:
            found = False
            
            # if sift not done in this frame, object not yet labelled and already seen a few times
            if(obj.frames == None and obj.count > n_sift and sift == False):
                sift_pool.put((obj, current_image_frame))
                sift = True     # enough work added for this frame
                
            # when not seen n times, remove object
            if obj.count < n_forget:
                objects.remove(obj)
                continue
            
            # draw the contour in an image for comparison
            obj_draw[:] = 0
            cv.FillPoly(obj_draw, [obj.cont], 1)
            area = np.dot(np.ravel(obj_draw), np.ravel(obj_draw))
            
            # check for each object whether we see the contour again in this frame
            for cont in conts_list:
                cont_draw[:] = 0
                cv.FillPoly(cont_draw, [cont], 1)
                
                # we found it
                # TODO: maybe I can make this part parallel (but then I would have to compute all and cannot stop when found, hmm)
                if np.dot(np.ravel(obj_draw), np.ravel(cont_draw)) > 0.5 * area:
                    obj.cont = cont            # update the contour to track movements
                    obj.count = obj.count + 1  # this helps new objects to recover from negative init
                    found = True
                    conts_list.remove(cont)     
                    break

            if not found:
                if obj.count > 0:
                    obj.count = 0               # once not seen -> immediately on delete list
                else:
                    obj.count -= 1              # few times not seen -> it becomes only worse
        
        # new objects get a chance in the object list
        for cont in conts_list: 
            objects.append(Obj(cont))
        timing['t_track'] += time.time() - t0


        # draw the contours and a box around them
        # TODO: write in the image how many keypoints found
        for obj in objects:
            if obj.count < 0:
                continue
            cv.FillPoly(contours, [obj.cont], obj.color)
            if obj.frames != None:
                col = cv.Scalar(0,255,0)
            else:
                col = cv.Scalar(0,0,255)
            for j in range(4):
                cv.Line(contours, obj.box_points[j], obj.box_points[(j+1)%4], col)

        # output images
        # TODO: all images in one
        outroi = (0, hist_height, width, height)
        a = cv.GetSubRect(out, outroi)
        cv.Copy(contours, a)
        histroi = (0, 0, width, hist_height)
        a = cv.GetSubRect(out, histroi)
        cv.Copy(hist_img, a)
        loop_c +=1
        timing['t_draw'] += time.time() - t0
        
        
        i = 0
        for key, val in timing.iteritems():
            cv.PutText(out, 
                       "%s: %f" % (key, val / loop_c), 
                       (0, height+hist_height-(len(timing)*20)+(i*15)),
                       font, 
                       cv.Scalar(255,255,255))
            i += 1


        # show the images
        cv.ShowImage( "Image Stream", current_image_frame )
        cv.ShowImage( "Depth Stream", current_depth_frame)
        cv.ShowImage("conts", out)
        
        # wait for user input (keys)
        current_key = cv.WaitKey( 5 ) % 0x100
        if current_key == KEY_ESC:
            for thread in threads:
                thread.stop()
            break

finally:
    for key, val in timing.iteritems():
        print "%s: %f" % (key, val / loop_c)
    g_context.Shutdown()
    
    # finally plot how sift execution time depends on patch size
    plt.figure()
    plt.scatter([bla[0] for bla in stats], [bla[1] for bla in stats])
    plt.show()
