
from protoobj import Obj, Rect, SiftThread, random_color
import OpenNIPythonWrapper as onipy
import Queue
import cv
import datetime
import numpy as np
import os.path
import pickle
import pyflann
import pylab as plt
import siftfastpy
import sys
import time
import traceback



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
n_threads = 10
outpath = '../out/'

# compute a timestamp for output file
outpath += datetime.datetime.now().strftime('%d%m%y_%H%M%S')

# create a list of random colors
color_tab = [random_color() for i in range(n_colors)]

# open the driver
g_context = onipy.OpenNIContext()
return_code = g_context.InitFromXmlFile(OPENNI_INITIALIZATION_FILE)

if return_code != onipy.XN_STATUS_OK:
    print "failed to initialize OpenNI"


try:
    # initialization stuff
    print 'loading ..'    
    save_count = 0
    objects = []
    stats = []
    current_key = -1
    timing = {'t_draw': 0, 't_histo': 0, 't_track': 0, 't_pre': 0}
    loop_c = 0
    font = cv.InitFont(cv.CV_FONT_HERSHEY_PLAIN, 1, 1, 0, 1, 8)
    elem = cv.CreateStructuringElementEx(8, 8, 4, 4, cv.CV_SHAPE_RECT)
    storage = cv.CreateMemStorage(0)
    
    # the image generators
    image_generator = onipy.OpenNIImageGenerator()
    g_context.FindExistingNode(onipy.XN_NODE_TYPE_IMAGE, image_generator)
    depth_generator = onipy.OpenNIDepthGenerator()
    g_context.FindExistingNode(onipy.XN_NODE_TYPE_DEPTH, depth_generator)
    depth_generator.set_viewpoint(image_generator)
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

    print 'load object db and create flann index ..'
    db = pickle.load(open('../out/pickled.db'))
    flann = pyflann.FLANN()
    # FIXME: currently without depth information because not aligned
    params = flann.build_index(db['features'][:,:-1], target_precision=0.95)

    # start the sifting threadpool
    print 'starting the sift threads ..'
    sift_pool = Queue.Queue(0)
    threads = []
    for i in range(n_threads):
        t = SiftThread(width, height, sift_pool, stats, flann, db['meta'])
        t.setDaemon(True)
        t.start()
        threads.append(t)
    

    while True:

        return_code = g_context.WaitAndUpdateAll()

        # get the images
        # TODO: ueber frames mitteln (entweder nur im histogramm,
        # vielleicht aber auch ueber ganze frames (mit discount factor)
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

        for i in range(max_dist / 10):
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
                
            x_scale = 1
            # draw current value in histogram
            pts = [(int(i * x_scale), hist_height),
                   (int(i * x_scale + x_scale), hist_height),
                   (int(i * x_scale + x_scale), int(hist_height-next_value*hist_height/max_hist)),
                   (int(i * x_scale), int(hist_height-cur_value*hist_height/max_hist))]
            cv.FillConvexPoly(hist_img, pts, color_tab[c])
          
        # time the histogram clustering  
        timing['t_pre'] += time.time() - t0      
        
        # iterate over tracked objects
        sift = False
        for obj in objects:
            found = False
            
            # if sift not done in this frame, object not yet labelled and already seen a few times
            if((obj.frames == None and obj.count > n_sift and sift == False) or
              (obj.frames != None and obj.count % 30 == 0)):
                sift_pool.put((obj, current_image_frame))
                sift = True     # enough work added for this frame
                
            # when not seen n times, remove object
            if obj.count < n_forget:
                objects.remove(obj)
                continue
            
            # get bounding rectangle 1
            r1 = Rect(cv.BoundingRect(obj.cont))
            area = r1.width * r1.height
            
            # check for each object whether we see the contour again in this frame
            for cont in conts_list:
                
                r2 = Rect(cv.BoundingRect(cont))
                w = min(r1.x2, r2.x2) - max(r1.x1, r2.x1)
                h = min(r1.y2, r2.y2) - max(r1.y1, r2.y1)
                overlap = w * h
                
                # we found it
                if w > 0 and h > 0 and overlap > 0.5 * area:
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


        # draw the contours, a box around them and how many keypoints were found
        for obj in objects:
            if obj.count < 0:
                continue
            cv.FillPoly(contours, [obj.cont], obj.color)
            if obj.frames != None:
                col = cv.Scalar(0,255,0)
                
                # label with number of keypoints
                cv.PutText(contours, 
                           "k: %s" % obj.ids,
                           obj.box_points[0],
                           font, cv.Scalar(255,255,255))
                
                # plot the keypoints
                for i in range(obj.frames.shape[0]):
                    cv.Rectangle(contours,
                                 (int(obj.frames[i,0])-1, int(obj.frames[i,1])-1),
                                 (int(obj.frames[i,0])+1, int(obj.frames[i,1])+1),
                                 cv.Scalar(255,255,255))
            else:
                col = cv.Scalar(0,0,255)
            for j in range(4):
                cv.Line(contours, obj.box_points[j], obj.box_points[(j+1)%4], col)

        # output images
        outroi = (0, hist_height, width, height)
        a = cv.GetSubRect(out, outroi)
        cv.AddWeighted(contours, 0.5, current_image_frame, 0.5, 0, a)
        histroi = (0, 0, width, hist_height)
        a = cv.GetSubRect(out, histroi)
        cv.Copy(hist_img, a)
        loop_c +=1
        timing['t_draw'] += time.time() - t0
        
        # print execution time statistics
        i = 0
        for key, val in timing.iteritems():
            cv.PutText(out, 
                       "%s: %f" % (key, val / loop_c), 
                       (0, height+hist_height-(len(timing)*20)+(i*15)),
                       font, 
                       cv.Scalar(255,255,255))
            i += 1

        # show the images
        cv.ShowImage( "Depth Stream", contours)
        cv.ShowImage("conts", out)
        
        # wait for user input (keys)
        current_key = cv.WaitKey( 5 ) % 0x100
        if current_key == KEY_ESC:
            for thread in threads:
                thread.stop()
            break
        
        # pickle the contours and save images
        elif current_key == ord('s'):
            if not os.path.exists(outpath):
                os.mkdir(outpath)
            scaled = cv.CreateMat(height, width, cv.CV_8UC1)
            cv.ConvertScale(current_depth_frame, scaled, 0.05)
            cv.SaveImage(outpath + '/image_%d.jpg' % save_count, current_image_frame)
            cv.SaveImage(outpath + '/depth_%d.jpg' % save_count, scaled)
            cv.SaveImage(outpath + '/out_%d.jpg' % save_count, out)
            with open(outpath + '/objects_%d.pickle' % save_count, 'w') as f:
                pickle.dump(objects, f)
            save_count = save_count +1

except Exception as inst:
    print inst
    for tb in traceback.format_tb(sys.exc_info()[2]):
        print tb

finally:
    for key, val in timing.iteritems():
        print "%s: %f" % (key, val / loop_c)
    g_context.Shutdown()
    
    # finally plot how sift execution time depends on patch size
#    plt.figure()
#    plt.scatter([bla[0] for bla in stats], [bla[1] for bla in stats])
#    plt.show()
