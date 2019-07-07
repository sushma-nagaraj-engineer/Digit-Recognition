from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn import preprocessing
import numpy as np
from collections import Counter
from keras.datasets import mnist
from sklearn import datasets

class SVM:
    def __init__(self):
        # load data
        (self.features_train, self.labels_train), (self.features_test, self.labels_test) = mnist.load_data()
        
    def transform_data(self):
        list_hog_fd = []
        for feature in self.features_train:
            fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1),
                     visualise=False)
            list_hog_fd.append(fd)
        hog_features = np.array(list_hog_fd, 'float64')

        # Normalize the features
        self.pp = preprocessing.StandardScaler().fit(hog_features)
        self.hog_features = self.pp.transform(hog_features)

        print("Count of digits in dataset", Counter(self.labels_train))

    def build_model(self):
        self.transform_data()
        # Create an linear SVM object
        self.clf = LinearSVC()
        # Perform the training
        self.clf.fit(self.hog_features, self.labels_train)
        # Save the classifier
        joblib.dump((self.clf, self.pp), "DigitClassifier.pkl", compress=3)

    def get_score(self):
        list_hog_fd_test = []
        for feature in self.features_test:
            fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1),
                     visualise=False)
            list_hog_fd_test.append(fd)
        hog_features_test = np.array(list_hog_fd_test, 'float64')

        pp = preprocessing.StandardScaler().fit(hog_features_test)
        hog_features_test = pp.transform(hog_features_test)
        score = self.clf.score(hog_features_test, self.labels_test, sample_weight=None)
        print("The score is: ")
        return score