import pandas as pd 
import numpy as np
_RANDOM_STATE =  108

import multiprocessing as MP
from multiprocessing import Pool
import io, json, os, re, shutil, string, sys, time, glob

sys.path.append('../Basic_Classifier_MultInOutput/')

# Importing some preprocessing modules
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE


# Importing some backend tensorflow and Keras modules. 
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import TextVectorization
import tensorflow.keras.layers as layers

# Importing some metrics 
from sklearn.metrics import accuracy_score, classification_report

from build_NeuralNetwork import Basic_Classifier_MultInOutput


FLAG = 2

if __name__ == "__main__": 
	
	AUTOTUNE = tf.data.AUTOTUNE
	NLAYERS = 0
	_MAX_FEATURES = 12800
	_SEQ_LENGTH = 256
	_NNODES_INTRINSIC = 64
	_EMBEDDING_DIMS = 64
	_IMG_DIM = 256
	_IMG_CHANNELS = 3

	# First step is to preprocess the data.  This includes 
	# simply stripping out the stopping words
	# then the next step is to vectorize/embed the text.  The goal of 
	# this step is to create vectors from the text, and embed them 
	# with numerical of values of somekind.  
	# TF-IDF does this simply by taking the inverted text frequency
	# of a given word 
	if FLAG == 0:
		theClassifier = Basic_Classifier()
		SAVDIR = './Basic_Classifier/ckpt'
		if not os.path.isdir(SAVDIR): os.makedirs(SAVDIR)
	elif FLAG == 1: 
		theClassifier = LSTM_Classifier()
		SAVDIR = './LSTM_Classifier/ckpt'
		if not os.path.isdir(SAVDIR): os.makedirs(SAVDIR)
	elif FLAG == 2: 
		theClassifier = Basic_Classifier_MultInOutput()
		SAVDIR = './Basic_Classifier_MultInOutput/ckpt'
		if not os.path.isdir(SAVDIR): os.makedirs(SAVDIR)	
	
	theClassifier._EMBEDDING_DIMS = _EMBEDDING_DIMS
	theClassifier._MAX_FEATURES = _MAX_FEATURES
	theClassifier._SEQ_LENGTH = _SEQ_LENGTH
	theClassifier._NNODES_INTRINSIC = _NNODES_INTRINSIC
	
	# Pre-processing: These models require a pre-processing step 
	# before the actual model building.  The reason I implement this 
	# is so that when building a model, it can be as flexible as possible
	# but a portion of this has to be made on the users end to make the 
	# data "visible" to the network.  Hence for example, the these networks
	# will require vectorized text data, not the actual text data itself
	# or the reduced images for the network to see (although you could add
	# multiple input layers for different image pre-processing techniques).  
	
	# We are now building the classifier.  This requires 
	# initLayer layer which includes the Input Layer and Embedding Layer
	INPUT_DICT = {
					'Text':  ['Text', _SEQ_LENGTH, _MAX_FEATURES, _EMBEDDING_DIMS, 'int64', 32], # --> 0
					'Title': ['Text', _SEQ_LENGTH, _MAX_FEATURES, _EMBEDDING_DIMS, 'int64', 32], # --> 1
					'Img1':   ['Image', [_IMG_DIM, _IMG_DIM, _IMG_CHANNELS], 'float64', 32], # --> 2
					'Img2':   ['Image', [_IMG_DIM, _IMG_DIM, _IMG_CHANNELS], 'float64', 32], # --> 3 
				}

	HIDDEN_DICT = {
					# Creating the LSTM layer for string data 
					'LSTMText1': [layers.Bidirectional(layers.LSTM(_EMBEDDING_DIMS, activation='tanh', return_sequences=True,)), [0, 0]], # --> 4
					'LSTMTitle1': [layers.Bidirectional(layers.LSTM(_EMBEDDING_DIMS, activation='tanh', return_sequences=True,)), [0, 1]], # --> 5
					'LSTMText2': [layers.Bidirectional(layers.LSTM(_NNODES_INTRINSIC, activation='tanh', return_sequences=True,)), [0, 4]], # --> 6
					'LSTMTitle2': [layers.Bidirectional(layers.LSTM(_NNODES_INTRINSIC, activation='tanh', return_sequences=True,)), [0, 5]], # --> 7
					'LSTMText3': [layers.Bidirectional(layers.LSTM(_NNODES_INTRINSIC, activation='tanh',)), [0, 6]], # --> 8
					'LSTMTitle3': [layers.Bidirectional(layers.LSTM(_NNODES_INTRINSIC, activation='tanh',)), [0, 7]], # --> 9
					'Concatenate_6_7': ['concatenate', [6, 7], 16, 'relu'],    # --> 10         
					
					# Convolving and pooling for IMG1 
					'IMG1_C2D_1': [layers.Conv2D(filters=32, kernel_size=(4,4)), [0, 2]],          # --> 11    
					'IMG1_P2D_1': [layers.MaxPooling2D(pool_size=(2, 2), padding='same'), [0, 11]], # --> 12
					'IMG1_C2D_2': [layers.Conv2D(filters=32, kernel_size=(4,4)), [0, 12]],          # --> 13   
					'IMG1_P2D_2': [layers.MaxPooling2D(pool_size=(2, 2), padding='same'), [0, 13]], # --> 14
					'IMG1_Flatten': [layers.Reshape((1, -1)), [0, 14]], # --> 15   

					# Convolving and pooling for IMG2 
					'IMG2_C2D_1': [layers.Conv2D(filters=32, kernel_size=(4,4)), [0, 3]],          # --> 16    
					'IMG2_P2D_1': [layers.MaxPooling2D(pool_size=(2, 2), padding='same'), [0, 16]], # --> 17
					'IMG2_C2D_2': [layers.Conv2D(filters=32, kernel_size=(4,4)), [0, 17]],          # --> 18   
					'IMG2_P2D_2': [layers.MaxPooling2D(pool_size=(2, 2), padding='same'), [0, 18]], # --> 19
					'IMG2_Flatten': [layers.Reshape((1, -1)), [0, 19]], # --> 20

					# Concat Lyaer 15 and 20 
					'Concatenate_15_20': ['concatenate', [15, 20], 16, 'relu'], # --> 21
					'Dense_21': [layers.Dense(64, activation='relu', name='Dense21'), [0, 21]],    # --> 22
					'Dense_22': [layers.Dense(64, activation='relu', name='Dense22'), [0, 22]],    # --> 23
					'Dense_23': [layers.Dense(64, activation='relu', name='Dense23'), [0, 23]],    # --> 24

				}
	# HIDDEN_DICT = {}

	# OUTPUT DICT parameters:
	# 1. Number of output dimensions
	# 2. Activation Function 
	# 3. Layer in Basic Layer to be connected to 
	OUTPUT_DICT = {
					'Output1': [layers.Dense(1, activation='sigmoid', name='Output1'), [0, 10]],
					'Output2': [layers.Dense(8, activation='softmax', name='Output2'), [0, 10]],
					'Output3': [layers.Dense(_SEQ_LENGTH, activation='relu', name='Output3'), [0, 24]],
					'Output4': [layers.Dense(_SEQ_LENGTH, activation='relu', name='Output4'), [0, 24]],
					'Output5': [layers.Dense(_SEQ_LENGTH, activation='relu', name='Output5'), [0, 15]],
					'Output6': [layers.Dense(_SEQ_LENGTH, activation='relu', name='Output6'), [0, 20]]
				}

	theClassifier.create_NeuralNetwork(INPUT_DICT, OUTPUT_DICT, HIDDEN_DICT)
	# theClassifier.create_NeuralNetwork(INPUT_DICT, OUTPUT_DICT, HIDDEN_DICT)
	model = theClassifier.summarizeModel()
	tf.keras.utils.plot_model(model, to_file=f'{SAVDIR}/multInOutput_BuildingClassifier.png',  \
							show_shapes=True, show_layer_names=True, show_layer_activations=True,)
	# print(summary)


