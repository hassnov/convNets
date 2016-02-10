import h5py
import cv2
import numpy
from matplotlib import pyplot as plt

f = h5py.File("/home/hasans/Downloads/dataset_stl_16000_32_1.h5", 'r')
d = f['/data']
print d.shape
im = numpy.reshape(d[:,2],[32,32,3])
img = cv2.imread('Mikolajczyk/graffiti/img1.ppm')[0:32, 0:32]
#x = im[:,:,0]
#im[:,:,0] = im[:,:,2]
#im[:,:,2] = x
#im = im*255                                                 
#plt.imshow(im), plt.show()
numpy.savetxt('numpy.txt', im,fmt='%f', delimiter=',')