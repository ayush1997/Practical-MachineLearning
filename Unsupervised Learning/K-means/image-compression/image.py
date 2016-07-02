##############################################################################################
#                                                                                            #
#  A kmean image quantisation model which compresses the images but in not lossless          #
#                                                                                            #                
##############################################################################################


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import mahotas as mh

#read and flatten image

original_img = np.array(mh.imread('new.jpg'),dtype=np.float64)/255
original_dimensions = tuple(original_img.shape)
width,height,depth = tuple(original_img.shape)
image_flattened = np.reshape(original_img,(width*height,depth))


# print original_img
print original_img[0]
print len(original_img[0])
print original_dimensions
print width,height,depth

#randomly pcked 1000 colors
image_array_sample = shuffle(image_flattened,random_state=0)[:1000]


estimator = KMeans(n_clusters=64,random_state=0)
estimator.fit(image_array_sample)

cluster_assignments = estimator.predict(image_flattened)
# print cluster_assignments[:10000]

compressed_palette = estimator.cluster_centers_
compressed_img = np.zeros((width,height,compressed_palette.shape[1]))

# print len(compressed_palette)
# print compressed_img

label_idx = 0

for i in range(width):
    for j in range(height):
        compressed_img[i][j] =compressed_palette[cluster_assignments[label_idx]]
        label_idx+=1

plt.subplot(122)
plt.title('orignal image')
plt.imshow(original_img)
plt.axis('off')
plt.subplot(121)
plt.title('compressed')
plt.imshow(compressed_img)
plt.axis('off')
plt.show()
