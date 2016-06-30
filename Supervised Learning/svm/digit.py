import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
import matplotlib.cm as cm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report


digits = fetch_mldata('MNIST original',data_home='data/mnist').data

print digits[8000]
print len(digits[0])

# counter = 1
# for i in range(1,4):
#     for j in range(1,6):
#         plt.subplot(3,5,counter)
#         plt.imshow(digits[(i-1)*8000+j].reshape((28,28)),cmap=cm.Greys_r)
#         plt.axis('off')
#         counter +=1
# plt.show()

if __name__ == '__main__':
    data = fetch_mldata('MNIST original',data_home='data/mnist')
    X,y = data.data,data.target
    X = X/255.0*2 -1

    X_train,X_test,y_train,y_test = train_test_split(X,y)
    pipeline=Pipeline([
        ('clf',SVC(kernel='rbf',gamma=0.01,C=100))
    ])
    print X_train.shapeparameters = {
    'clf__gamma':(0.01,0.03,0.1,0.3,1),
    'clf__C':(0.1,0.3,1,3,10,30),
    }

    
