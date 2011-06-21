

bugs
====

* when plotting the histogram clustering it is visible that the cluster from the big distant *pile* is not plotted as a cluster


experiments to do
=================

* depth-scale ratio
    * compute the relation between the depth value in a key-point and the scale of this point. We expect it to be constant. Look at the distribution of this relation for different camera distances.
    * match an object which is not in the database against the database. The result are false-positives only. Compare the depth-scale ratio of the false-positives to a distribution where the matched object is in the database.
* Whats the speed/quality improvement of computing the sift features only within the contour and not in the whole surrounding box.


how to build the model (database of objects)
============================================

* take images of a revolving object
* compute trajectory of sift feature of images
* compute mean (or median) of sift feature per trajectory 
* also compute mean depth value
    * maybe check for peaks (errors) in depth trajectory


object selection
================

* color histogram
    * look at homogeneity of color histogram. We expect objects to be more homogeneous than random patches of the image. The color histograms could be used in the validation of proto-objects.
    * the processed (SIFT) objects should also be annotated with the color histogram. Maybe this could later be used for a rough matching.

* compactness: how much of the object surrounding box is covered by the area surrounded by the contour. This could maybe be used to filter out big flat areas.


speed/quality improvements
==========================

* downsampling of depth image
* improve the SIFT-feature computation queue
    * new (not yet sifted) objects have highest priority
    * recompute sift features whenever there is time in order to keep track of changes in the image (maybe an object is turned). (priority class 2)
    * objects in the priority queue could be ordered by size, distance, saliency, etc.. (within priority class 2)
    * what was already sifted but nothing found stays in the list (because of possible rotation, etc) but with lowest priority class
* why are not all processors used when the thread-pool is used. Maybe switch to processes (!not so easy to implement because of no common memory between the processes!)
* maybe average over frames
    * either only the histogram
    * or maybe even the whole images
    * if we do this we could either take the last few images or even more with applying a discount factor
* try using the high-resolution images of the Kinect
    * is this already supported in the python bindings?
    * are the images still aligned?
    * does it become too slow with this images?
    * ==> use this images at least for the recorder

	
