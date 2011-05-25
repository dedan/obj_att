
import pyflann
import pickle
import numpy as np

db = pickle.load(open('../out/pickled.db'))

flann = pyflann.FLANN()
bla = flann.build_index(db['features'][:,:-1])
print bla

tmp = db['features'][15:17,:-1]
result, dists = flann.nn_index(tmp, 1, checks=bla['checks'])
print result
print dists