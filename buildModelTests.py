import pandas as pd 
import numpy as np
_RANDOM_STATE =  108

import multiprocessing as MP
from multiprocessing import Pool
import io, json, os, re, shutil, string, sys, time, glob

sys.path.append('../Basic_Classifier/')

# Importing some preprocessing modules
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from Basic_classifier import Basic_Classifier, Basic_preprocess, vectorize_text, vectorize_text_MP
from LSTM_classifier import LSTM_Classifier


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

_NLAYERS = 0
_MAX_FEATURES = 2400
_SEQ_LENGTH = 128
_NNODES_INTRINSIC = 8

LSTMClassifier = LSTM_Classifier()
LSTMClassifier._MAX_FEATURES = _MAX_FEATURES
LSTMClassifier._SEQ_LENGTH = _SEQ_LENGTH
LSTMClassifier._NNODES_INTRINSIC = _NNODES_INTRINSIC

LSTMClassifier.add_InitLayer()

for l in range(_NLAYERS): LSTMClassifier.add_DenseLayers(l, NNODES=LSTMClassifier._NNODES_INTRINSIC)

LSTMClassifier.add_OutputLayer(NNODES=LSTMClassifier._NNODES_INTRINSIC)
LSTMClassifier.getNumLayers()
model = LSTMClassifier.summarizeModel()
# print(summary)
tf.keras.utils.plot_model(model)
tf.keras.utils.plot_model(model, to_file='LSTMClassifier.png', show_shapes=True)

BasicClassifier = Basic_Classifier()
BasicClassifier._MAX_FEATURES = _MAX_FEATURES
BasicClassifier._SEQ_LENGTH = _SEQ_LENGTH
BasicClassifier._NNODES_INTRINSIC = _NNODES_INTRINSIC

BasicClassifier.add_InitLayer()

for l in range(_NLAYERS): BasicClassifier.add_DenseLayers(l, NNODES=BasicClassifier._NNODES_INTRINSIC)

BasicClassifier.add_OutputLayer(NNODES=BasicClassifier._NNODES_INTRINSIC)
BasicClassifier.getNumLayers()
model = BasicClassifier.summarizeModel()


tf.keras.utils.plot_model(model, to_file='BasicClassifier.png',  show_shapes=True)

