#linear regression using covariance and variance

# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# import numpy as np
#
# X = [[6],[8],[10],[14],[18]]
# y = [[7],[9],[13],[17.5],[18]]
#
# plt.figure()
# plt.xlabel("diameter in inches")
# plt.ylabel("price in dollar")
# plt.plot(X,y,'k.')
#
# model = LinearRegression()
# model.fit(X,y)
#
# #mean
# print "prdicted is", model.predict(X)
# print "r squared is ", np.mean((model.predict(X)-y)**2)
#
# #variance
# print np.var(X,ddof=1)
#
# #covariance
# print np.cov([6,8,10,14,18],[7,9,13,17.5,18])[0][1]
# plt.axis([0,25,0,25])
# plt.grid(True)
# plt.show()


#multivariate and normal eq

# from numpy.linalg import  inv
# from numpy import dot, transpose
# X= [[1,6,2],[1,8,1],[1,10,0],[1,14,2],[1,18,0]]
# Y =[[7],[9],[13],[17.5],[18]]
#
# b = dot(inv(dot(transpose(X),X)),dot(transpose(X),Y))
# print b
# X_test = [[1,8,2],[1,9,0],[1,11,2],[1,16,2],[1,12,0]]
#
# pred_y = dot(X_test,b)
#
# print "predicted manually with normal equation"
# print pred_y
#
#
# from sklearn.linear_model import LinearRegression
#
# X_train= [[6,2],[8,1],[10,0],[14,2],[18,0]]
# Y_train =[[7],[9],[13],[17.5],[18]]
#
# model = LinearRegression()
# model.fit(X_train,Y_train)
#
#
# X_test = [[8,2],[9,0],[11,2],[16,2],[12,0]]
# Y_test = [[11],[8.5],[15],[18],[11]]
#
# print "prdicted using sklearn"
# prediction = model.predict(X_test)
# print prediction
#
# for i,pred in enumerate(prediction):
#     print "prdiction=",pred,"original=",Y_test[i]


#polynomila regression

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
#
# X_train = [[6],[8],[10],[14],[18]]
# Y_train = [[7],[9],[13],[17.5],[18]]
#
# X_test= [[6],[8],[11],[16]]
# Y_test =[[8],[12],[15],[18]]
#
# regressor =LinearRegression()
# regressor.fit(X_train,Y_train)
#
# xx = np.linspace(0,26,100)
# yy = regressor.predict(xx.reshape(xx.shape[0],1))
#
# plt.plot(xx,yy)


#pixel intesities

# from sklearn import datasets
# digits = datasets.load_digits()
# print digits
# print "Digigt:",digits.target[3]
# print digits.images[3]
# print "feature vector:", digits.images[3].reshape(-1,64)

# import numpy as nps
# from skimage.feature import corner_harris,corner_peaks
# from skimage.color import rgb2gray
# import matplotlib.pyplot as plt
# import skimage.io as io
# from skimage.exposure import equalize_hist
#
# def show_corners(corners,image):
#     fig = plt.figure()
#     plt.gray()
#     plt.imshow(image)
#     y_corner,x_corner = zip(*corners)
#     plt.plot(x_corner,y_corner,'or')
#     plt.xlim(0,image.shape[1])
#     plt.ylim(image.shape[0],0)
#     fig.set_size_inches(nps.array(fig.get_size_inches())*1.5)
#     plt.show()
#
# image  =io.imread('new.jpg')
# image = equalize_hist(rgb2gray(image))
# corners  =corner_peaks(corner_harris(image),min_distance=2)
# show_corners(corners,image)


#Bag of Words

# from sklearn.feature_extraction.text import CountVectorizer
# corpus=["UNC played Duke in basketball,Duke lost the basketball game,the game was fixed"]
#
# vectorizer =  CountVectorizer(stop_words="english")
# print vectorizer.fit_transform(corpus).todense()
# print vectorizer.vocabulary_






#k-means

# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn import metrics
# import matplotlib.pyplot as plt
# plt.subplot(3,2,1)
# x1 = np.array([1,2,3,1,5,6,5,5,6,7,8,9,7,9])
# x2 = np.array([1,3,2,2,8,6,7,6,7,1,2,1,1,3])
#
# X = np.array(zip(x1,x2)).reshape(len(x1),2)
# print X
#
# plt.xlim(0,10)
# plt.ylim(0,10)
# plt.title('instnces')
#
# plt.scatter(x1,x2)
#
# # plt.show()
#
# colors = ['b','g','r','c','m','y','k','b']
# markers = ['o','s','D','v','^','p','*','+']
# tests =[2,3,4,5,8]
#
# subplot_counter = 1
# for t in tests:
#     subplot_counter+=1
#     plt.subplot(3,2,subplot_counter)
#     kmeans_model = KMeans(n_clusters=t).fit(X)
#     for i,l in enumerate(kmeans_model.labels_):
#         plt.plot(x1[i],x2[i],color=colors[l],marker = markers[l],ls='None')
#         plt.xlim([0,10])
#         plt.ylim([0,10])
#         plt.title('k=%s, silhoutee=%0.3f' %(t,metrics.silhouetee_score(X,kmeans_model.labels_,metric='euclidian')))
#         plt.show()
#

#suft feature extraction

import mahotas as mh
from mahotas.features import surf

image = mh.imread('new.jpg',as_grey=True)
print 'The first SURF description:',surf.surf(image)[0]
print 'extracted SURF description',len(surf.surf(image)[0])
