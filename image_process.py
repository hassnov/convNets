import cv2
import itertools
#import numpy as np
from matplotlib import pyplot as plt

sift = cv2.xfeatures2d.SIFT_create()  # @UndefinedVariable

img1 = cv2.imread('Mikolajczyk/graffiti/img1.ppm')
#img1 = img1[300:600,300:600]
gray1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
kp = sift.detect(gray1,None) 
#img=cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
des = sift.compute(gray1, kp)

img2 = cv2.imread('Mikolajczyk/graffiti/img2.ppm')
img2 = img2[400:500,400:500]
gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
tkp = sift.detect(gray2,None)
#img2=cv2.drawKeypoints(gray,kp,img2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
tdes = sift.compute(gray2,tkp)

#distance
FLANN_INDEX_KDTREE = 1
flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
#search_params = dict(checks=50)
flann = cv2.flann.Index(des[1], flann_params)  # @UndefinedVariable
#flann = cv2.FlannBasedMatcher(flann_params,search_params)    
idx, dist = flann.knnSearch(tdes[1], 1, params={})
del flann

#sort distances
dist = dist[:,0]/2500.0
dist = dist.reshape(-1,).tolist()
idx = idx.reshape(-1).tolist()
indices = range(len(dist))
indices.sort(key=lambda i: dist[i])
dist = [dist[i] for i in indices]
idx = [idx[i] for i in indices]

#choose distances less than a threshold
ti = range(len(dist))
distance = 3
kp_final = []
tkp_final = []
for i, dis, j in itertools.izip(idx, dist, ti):
    if dis < distance:
        kp_final.append(kp[i])
        tkp_final.append(tkp[j])
    else:
        break

#img2 = cv2.drawMatches(img1, kp_final, img2, tkp_final, idx, img2,flags=2)    # @UndefinedVariable
gray1 = cv2.drawKeypoints(img1, kp_final, gray1)
gray2 = cv2.drawKeypoints(img2, tkp_final, gray2)

fig = plt.figure()
a=fig.add_subplot(1,2,1)
a.set_title('Img1')
plt.imshow(gray1)

b=fig.add_subplot(1,2,2)
b.set_title('Img2')
plt.imshow(gray2)


plt.show()

#cv2.imwrite('Mikolajczyk/graffiti/graffitisift_keypoints.jpg',img)