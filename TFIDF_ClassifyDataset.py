import io, json, os, re, shutil, string, sys, time, glob
import multiprocessing as MP
from multiprocessing import Pool

import matplotlib.pyplot as plt 
import seaborn as sns

import pandas as pd 
import numpy as np
_RANDOM_STATE =  108

import tensorflow as tf

# Importing from the Term Frequency- Inverse Density Frequency module from my 
# own build in class
sys.path.append('../TFIDF_Classifier/')
from TFIDF_classifier import TFIDF_Classifier

# Importing some preprocessing modules 
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from TFIDF_classifier import TFIDF_preprocess, vectorize_text, vectorize_text_MP

# Importing the TFIDF Vectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer

# Importing some different learning methods for comparison 
from sklearn.naive_bayes import MultinomialNB,CategoricalNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Importing some metrics 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix




if __name__ == "__main__": 
	
	AUTOTUNE = tf.data.AUTOTUNE

	fakeNews = pd.read_csv('fakeNews_dataset.csv', sep=',', header=0)
	fakeNews = fakeNews.dropna()

	# Retreiving the data 
	X = fakeNews.loc[:, 'text'].values[:20000]
	y = fakeNews['label'].values[:20000]

	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=_RANDOM_STATE)


	TFIDF_vectorizer = TfidfVectorizer(max_features = 4000)
	X_train = TFIDF_vectorizer.fit_transform(X_train).toarray()
	X_test = TFIDF_vectorizer.transform(X_test).toarray()

	FLAG = int(sys.argv[1])
	if FLAG == 1: 
		classifier = MultinomialNB()
		confMatrix_File = 'multiNB_confusion.png'
	if FLAG == 2: 
		classifier = DecisionTreeClassifier()
		confMatrix_File = 'dct_confusion.png'
	if FLAG == 3: 
		classifier = RandomForestClassifier(n_estimators=400)
		confMatrix_File = 'rfc_confusion.png'
	if FLAG == 4: 
		classifier = XGBClassifier(learning_rate=0.1, n_estimators=200)
		confMatrix_File = 'XGB_confusion.png'
	classifier.fit(X_train, y_train)

	y_pred = classifier.predict(X_test)
	y_pred_res = classifier.predict(X_test)
	# Classification metrics

	classification_report_un = classification_report(y_test, y_pred)
	print('\n Accuracy: ', accuracy_score(y_test, y_pred))
	print('\nClassification Report')
	print('======================================================')
	print('\n', classification_report_un)

	#Get the confusion matrix
	cf_matrix = confusion_matrix(y_test, y_pred)
	print(cf_matrix)
	confMat_fig = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')
	fig = confMat_fig.get_figure()
	fig.savefig(f'./ConfusionMatrices/{confMatrix_File}')





