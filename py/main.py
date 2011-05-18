__author__ = 'dedan'


import cv
import OpenNIPythonWrapper as onipy
import numpy as np
import pickle

# for making random numbers
import random

def random_color():
    """
    Return a random color
    """
    icolor = random.randint (0, 0xFFFFFF)
    return cv.Scalar (icolor & 0xff, (icolor >> 8) & 0xff, (icolor >> 16) & 0xff)

class Obj:
    def __init__(self, cont, color):
        self.cont = cont
        self.color = color
        self.inact_count = 0
        self.id = random.randint(1, 2**31)



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
    current_key = -1
    image_generator = onipy.OpenNIImageGenerator()
    return_code = g_context.FindExistingNode(onipy.XN_NODE_TYPE_IMAGE, image_generator)
    current_image_frame = cv.CreateImageHeader(image_generator.Res(), cv.IPL_DEPTH_8U, 3)

    depth_generator = onipy.OpenNIDepthGenerator()
    return_code = g_context.FindExistingNode(onipy.XN_NODE_TYPE_DEPTH, depth_generator)
    width = depth_generator.XRes()
    height = depth_generator.YRes()
    current_depth_frame = cv.CreateMatHeader(height, width, cv.CV_16UC1)
    
    # create some matrices for drawing
    hist_img = cv.CreateMat(hist_height, width, cv.CV_8UC3)
    out = cv.CreateMat(height + hist_height, width, cv.CV_8UC3)
    contours = cv.CreateMat(height, width, cv.CV_8UC3)



    while True:

        return_code = g_context.WaitAndUpdateAll()

        # get the images
        depth_data_raw = depth_generator.GetGrayscale16DepthMapRaw()
        cv.SetData(current_depth_frame, depth_data_raw)
        image_data_raw = image_generator.GetBGR24ImageMapRaw()
        cv.SetData(current_image_frame, image_data_raw)
        
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

        for i in range(max_dist / 10):
            cur_value = hist[i]
            next_value = hist[i+1]
            
            # still walking up the hill
            if cur_value > hist[end]:
                end = i

            # arrived in a valley
            if ((cur_value < perc * hist[end]) & (1.1 * cur_value < next_value)):

                # cut out a certain depth layer
                mask = (depth > start * 10) & (depth < i * 10)
                mask = mask.astype(np.uint8) * 255
                elem =  cv.CreateStructuringElementEx(8, 8, 4, 4, cv.CV_SHAPE_RECT)
                
                # erode the layer and find contours
                cv.Erode(mask, mask, elem)
                storage = cv.CreateMemStorage(0)
                conts = cv.FindContours(mask, storage, cv.CV_RETR_EXTERNAL, cv.CV_CHAIN_APPROX_SIMPLE)
                
                # collect all interesting contours in a list
                while conts:
                    if len(conts) > 30 and cv.ContourArea(conts) > 250:
                        conts_list.append(Obj(list(conts), color_tab[c]))
                    conts = conts.h_next()
                
                # prepare for search for next hill
                start = i
                end = i
                c = c + 1
                
            # draw current value in histogram
            pts = [(i, hist_height),
                   (i, hist_height),
                   (i, int(hist_height-next_value*hist_height/max_hist)),
                   (i, int(hist_height-cur_value*hist_height/max_hist))]
            cv.FillConvexPoly(hist_img, pts, color_tab[c])


                
        # pickle the contours
        if current_key == ord('c'):
            with open('/home/dedan/obj_att/out/conts_%d.pickle' % save_count, 'w') as f:
                pickle.dump(conts_list, f)
                save_count = save_count +1

        # print the contours and a box around them
        for obj in conts_list:
            cv.FillPoly(contours, [obj.cont], obj.color)
            box = cv.MinAreaRect2(obj.cont)
            b_points = [(int(x), int(y)) for x, y in cv.BoxPoints(box)]
            for j in range(4):
                cv.Line(contours, b_points[j], b_points[(j+1)%4], cv.Scalar(0,255,0))
                




        # output images
        outroi = (0, hist_height, width, height)
        a = cv.GetSubRect(out, outroi)
        cv.Copy(contours, a)
        histroi = (0, 0, width, hist_height)
        a = cv.GetSubRect(out, histroi)
        cv.Copy(hist_img, a)

        # show the images
        cv.ShowImage( "Image Stream", current_image_frame )
        cv.ShowImage( "Depth Stream", current_depth_frame)
        cv.ShowImage("conts", out)

        # wait for user input (keys)
        current_key = cv.WaitKey( 5 ) % 0x100
        if current_key == KEY_ESC:
            break

finally:

    g_context.Shutdown()
