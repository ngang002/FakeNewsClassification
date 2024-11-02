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

FLAG = 1
if FLAG == 0: from Basic_classifier import Basic_Classifier, Basic_preprocess, vectorize_text, vectorize_text_MP
elif FLAG == 1: from LSTM_classifier import LSTM_Classifier, Basic_preprocess, vectorize_text, vectorize_text_MP

if __name__ == "__main__": 
	
	AUTOTUNE = tf.data.AUTOTUNE
	NLAYERS = 1
	_MAX_FEATURES = 128000
	_SEQ_LENGTH = 16
	_NNODES_INTRINSIC = 128
	_EMBEDDING_DIMS = 64
	
	# Retreiving the data and some preliminary 
	# cleanup stuff 
	fakeNews = pd.read_csv('fakeNews_dataset.csv', sep=',', header=0)
	fakeNews = fakeNews.dropna()

	
	X = fakeNews.loc[:, ['text']].values[:20000]
	y = fakeNews['label'].values[:20000]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=_RANDOM_STATE)
	
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
	
	theClassifier._EMBEDDING_DIMS = _EMBEDDING_DIMS
	theClassifier._MAX_FEATURES = _MAX_FEATURES
	theClassifier._SEQ_LENGTH = _SEQ_LENGTH
	theClassifier._NNODES_INTRINSIC = _NNODES_INTRINSIC
	
	
	# We must first vectorize all the data first, which consists
	# of taking our text data, and converting them into either floats 
	# or integer data
	vectorize_layer = theClassifier.create_VectorizationLayer()
	vectorize_layer.adapt(X_train)
	X_train_toVector = [[text, vectorize_layer, _] for _, text in enumerate(X_train)]
	X_test_toVector = [[text, vectorize_layer, _] for _, text in enumerate(X_test)]

	start_time = time.time()
	with Pool(processes=12) as p:
		output = p.map(vectorize_text_MP, X_train_toVector)
	print(f"Time to convert text to vectors: {time.time()-start_time}")
	X_train_vectorized = np.array(output)
	

	start_time = time.time()
	with Pool(processes=12) as p:
		output = p.map(vectorize_text_MP, X_test_toVector)
	print(f"Time to convert text to vectors: {time.time()-start_time}")
	X_test_vectorized = np.array(output)
	print(X_train_vectorized.shape, X_test_vectorized.shape)
	
	# We are now building the classifier.  This requires 
	# initLayer layer which includes the Input Layer and Embedding Layer
	theClassifier.add_InitLayer()

	# These current iterations simply add dense layers
	# which then include a random dropout of 0.2.  This 
	# can easiley be changed to include better, but 
	# more expensive layers like bi-directional LSTM
	# layers as well
	for l in range(NLAYERS):
		theClassifier.add_HiddenLayers(l, NNODES=theClassifier._NNODES_INTRINSIC)

	# This creates the output layers 
	theClassifier.add_OutputLayer(NNODES=theClassifier._NNODES_INTRINSIC)
	theClassifier.getNumLayers()
	model = theClassifier.summarizeModel()
	# print(summary)

	# creating the checkpoint 

	checkpoint_filepath = f'{SAVDIR}/checkpoint.weights.h5'
	model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    		filepath=checkpoint_filepath,
    		save_weights_only=True,
    		monitor='val_accuracy',
    		mode='max',
    		save_best_only=True)

	# creating a stopping point 
	early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

	theClassifier.set_optimizer(learning_rate=1E-5, epsilon=1E-5)
	model.compile(optimizer=theClassifier.optimizer, loss=theClassifier.loss, metrics=['accuracy'])    
	model.build(input_shape=(theClassifier._SEQ_LENGTH))
	# print(test_text, label)
	
	history = model.fit(x=X_train_vectorized, y=y_train, validation_split=0.2, batch_size=32,
					 epochs=50, callbacks=[model_checkpoint_callback, early_stopping])
	json.dump(history.history, open('./{}/classifier_history', 'w'))

	# best_model = keras.models.load_model(checkpoint_filepath) # The model (that are considered the best) can be loaded as -

	y_pred = model.predict(X_test_vectorized)[:,0]
	y_pred = [int(round(y)) for y in y_pred]

	# print(y_pred.shape, y_test.shape, y_pred_res.shape, y_test_res.shape)
	classification_report_un = classification_report(y_test, y_pred)

	print('\n Accuracy: ', accuracy_score(y_test, y_pred))
	print('\nClassification Report')
	print('======================================================')
	print('\n', classification_report_un)
	