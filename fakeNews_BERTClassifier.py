import pandas as pd 
import numpy as np
_RANDOM_STATE =  108

import multiprocessing as MP
from multiprocessing import Pool
import io, json, os, re, shutil, string, sys, time, glob

sys.path.append('../BERT_Classifier/')

# Importing some preprocessing modules
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from BERT_preprocessor import preprocessBERTClassifier

# Importing some backend tensorflow and Keras modules. 
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import TextVectorization

# Importing some metrics 
from sklearn.metrics import accuracy_score, classification_report

# Use Keras 2.
version_fn = getattr(tf.keras, "version", None)
if version_fn and version_fn().startswith("3."):
	# print("We are switching from using Keras 3.0 to Keras 2.0")
	import tf_keras as keras
else:
	keras = tf.keras
import keras_nlp
version_keras = getattr(keras, "version", None)
# print(f"Keras version: {version_keras}")


if __name__ == "__main__": 
	
	AUTOTUNE = tf.data.AUTOTUNE

	fakeNews = pd.read_csv('fakeNews_dataset.csv', sep=',', header=0)
	fakeNews = fakeNews.dropna()

	NLAYERS = 4
	# print(redditNews.head())

	#print(redditNews.columns)

	# Retreiving the data 
	X = fakeNews.loc[:, 'text'].values[:4000]
	y = fakeNews['label'].values[:4000]

	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=_RANDOM_STATE)

	
	
