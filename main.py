import numpy as np
import sys, os
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from generate_features import generate_features
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

classifiers = [svm.LinearSVC(), svm.SVC(), GaussianNB(), KNeighborsClassifier(), MLPClassifier(), RandomForestClassifier(), AdaBoostClassifier(), DecisionTreeClassifier()]

def main():
	train = [l.split('\t') for l in open('corpora/train/features.tsv', 'r').readlines()[1:]]
	train_data = np.array([[c for c in line[:-1]] for line in train]).astype(np.float)
	train_cls = np.array([line[-1].strip() for line in train])
	 
	test = [l.split('\t') for l in open('corpora/test/features.tsv', 'r').readlines()[1:]]
	test_data = np.array([[c for c in line[:-1]] for line in test]).astype(np.float)
	test_cls = np.array([line[-1].strip() for line in test])

	for classifier in classifiers:
		print classifier
		clf = classifier
		clf.fit(train_data, train_cls)
		expected = test_cls
		predicted = clf.predict(test_data)
		print(metrics.classification_report(expected, predicted))

if __name__ == '__main__':
	for i in xrange(1, 10):
		blocks = i*20
		print "[INFO] Generating model for {} blocks".format(blocks)
		generate_features('train', blocks)
		generate_features('test', blocks)
		main()
		print "="*30
