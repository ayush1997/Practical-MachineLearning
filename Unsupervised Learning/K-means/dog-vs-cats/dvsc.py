##################################################################################################################
# An kmeans clustering model for classifying the images as dog or cat                                            #
#                                                                                                                #
# dataset source : www.kaggle.com/c/dogs-vs-cats/data                                                            #
###################################################################################################################


import numpy as np
import mahotas as mh
from mahotas.features import surf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.cluster import MiniBatchKMeans
import glob

all_instance_filenames=[]
all_instance_targets = []
#glob is Unix style pathname pattern expansion-regex for path name,return string for the path name matched.Results are in arbitrary order
#we are makng list of all the file names and target variables.Dog=0 and cat=1
for f in glob.glob('train/*.jpg'):
    target = 1 if 'cat' in f else 0
    all_instance_filenames.append(f)
    all_instance_targets.append(target)



all_instance_filenames = all_instance_filenames[:100]
all_instance_targets = all_instance_targets[:100]
# print all_instance_filenames
# print all_instance_targets


#surf features are made using mahotas
surf_features = []
counter = 0
for f in all_instance_filenames:
    print 'reading image:',f
    image =mh.imread(f,as_grey=True)
    surf_features.append(surf.surf(image)[:,5:])
    # original_dimensions = tuple(image.shape)
    # print original_dimensions
    # print image
    # print image[0]
    # print len(image[0])
    # print len(surf.surf(image)[:,5:][0])
    # print surf.surf(image)[0]
    # print surf.surf(image)

# print surf_features[3:]

#60% data for training
#testing and trainig data made
train_len = int(len(all_instance_filenames) * .60)

X_train_surf_features = np.concatenate(surf_features[:train_len])
X_test_surf_features = np.concatenate(surf_features[train_len:])
y_train = all_instance_targets[:train_len]
y_test = all_instance_targets[train_len:]

n_clusters = 300
print 'clustering',len(X_train_surf_features),'features'
estimator = MiniBatchKMeans(n_clusters=n_clusters)
estimator.fit_transform(X_train_surf_features)

X_train = []
for instance in surf_features[:train_len]:
    clusters = estimator.predict(instance)
    features = np.bincount(clusters)
    if len(features) < n_clusters:
        features = np.append(features,np.zeros((1,n_clusters-len(features))))
    X_train.append(features)

X_test = []
for instance in surf_features[train_len:]:
    clusters = estimator.predict(instance)
    features = np.bincount(clusters)
    if len(features) < n_clusters:
        features = np.append(features,np.zeros((1,n_clusters-len(features))))
    X_test.append(features)

clf = LogisticRegression(C=0.001,penalty='12')
clf.fit_transform(X_train,y_train)
predictions = clf.predict(X_test)

print "precision:",precision_score(y_test,predictions)
print "Recall:",recall_score(y_test,predictions)
print "accuracy",accuracy_score(y_test,predictions)
