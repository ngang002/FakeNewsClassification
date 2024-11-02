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

FLAG = 2
if FLAG == 0: from Basic_classifier import Basic_Classifier, Basic_preprocess, vectorize_text, vectorize_text_MP
elif FLAG == 1: from LSTM_classifier import LSTM_Classifier, Basic_preprocess, vectorize_text, vectorize_text_MP
elif FLAG == 2: from Basic_classifier_MultInOutput import Basic_Classifier_MultInOutput, Basic_preprocess, vectorize_text, vectorize_text_MP

if __name__ == "__main__": 
	
	AUTOTUNE = tf.data.AUTOTUNE
	NLAYERS = 0
	_MAX_FEATURES = 128000
	_SEQ_LENGTH = 256
	_NNODES_INTRINSIC = 64
	_EMBEDDING_DIMS = 64
	
	# Retreiving the data and some preliminary 
	# cleanup stuff 
	fakeNews = pd.read_csv('fakeNews_dataset.csv', sep=',', header=0)
	fakeNews = fakeNews.dropna()

	
	X = fakeNews.loc[:, ['text', 'title']].values[:20000]
	y = fakeNews['label'].values[:20000]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=_RANDOM_STATE)
	Xtext_train, Xtitle_train, Xtext_test, Xtitle_test = X_train[:,0], X_train[:,1], X_test[:,0], X_test[:,1]
	Xtextlen_train  = np.array([len(text.split()) for text in Xtext_train])
	Xtitlelen_train = np.array([len(text.split()) for text in Xtitle_train])
	Xtextlen_test   = np.array([len(text.split()) for text in Xtext_test])
	Xtitlelen_test  = np.array([len(text.split()) for text in Xtitle_test])
	print(Xtext_train.shape, Xtitle_train.shape, Xtext_test.shape, Xtitle_test.shape)

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
	
	
	# We must first vectorize all the data first, which consists
	# of taking our text data, and converting them into either floats 
	# or integer data
	vectorize_layer = theClassifier.create_VectorizationLayer()
	vectorize_layer.adapt(Xtext_train)
	
	Xtext_train_toVector = [[text, vectorize_layer, _] for _, text in enumerate(Xtext_train)]
	Xtext_test_toVector = [[text, vectorize_layer, _] for _, text in enumerate(Xtext_test)]

	Xtitle_train_toVector = [[text, vectorize_layer, _] for _, text in enumerate(Xtitle_train)]
	Xtitle_test_toVector = [[text, vectorize_layer, _] for _, text in enumerate(Xtitle_test)]


	start_time = time.time()
	with Pool(processes=12) as p:
		output_text = p.map(vectorize_text_MP, Xtext_train_toVector)
		output_title = p.map(vectorize_text_MP, Xtitle_train_toVector)
	print(f"Time to convert text to vectors: {time.time()-start_time}")
	Xtext_train_vectorized = np.array(output_text)
	Xtitle_train_vectorized = np.array(output_title)
	

	start_time = time.time()
	with Pool(processes=12) as p:
		output_text = p.map(vectorize_text_MP, Xtext_test_toVector)
		output_title = p.map(vectorize_text_MP, Xtitle_test_toVector)
	print(f"Time to convert text to vectors: {time.time()-start_time}")
	Xtext_test_vectorized = np.array(output_text)
	Xtitle_test_vectorized = np.array(output_title)
	print(Xtext_train_vectorized.shape, Xtext_test_vectorized.shape)
	print(Xtitle_train_vectorized.shape, Xtitle_test_vectorized.shape)
	
	# We are now building the classifier.  This requires 
	# initLayer layer which includes the Input Layer and Embedding Layer
	INPUT_DICT = {
					'Title': [_SEQ_LENGTH, 'int64', 64,  True], 
					'Text': [_SEQ_LENGTH, 'int64', 64, True], 
				}

	HIDDEN_DICT = {
					'Hidden1': [64, 'relu', -1],
					'Hidden2': [64, 'relu', -1],
				}


	# OUTPUT DICT parameters:
	# 1. Number of output dimensions
	# 2. Activation Function 
	# 3. Layer in Basic Layer to be connected to 
	OUTPUT_DICT = {
					'Output1': [1, 'sigmoid', -1],
				}

	theClassifier.create_NeuralNetwork(INPUT_DICT, OUTPUT_DICT, HIDDEN_DICT)
	model = theClassifier.summarizeModel()
	tf.keras.utils.plot_model(model, to_file=f'{SAVDIR}/multInOutput_Classifier.png',  \
							show_shapes=True, show_layer_names=True, show_layer_activations=True,)
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

	theClassifier.set_optimizer(learning_rate=5E-5)
	model.compile(optimizer=theClassifier.optimizer, loss=theClassifier.loss, metrics=['accuracy'])    
	model.build(input_shape=(theClassifier._SEQ_LENGTH, theClassifier._SEQ_LENGTH))
	model.summary()
	# print(test_text, label)

	history = model.fit(x=[Xtext_train_vectorized, Xtitle_train_vectorized], y=y_train, validation_split=0.2, batch_size=32,
					 epochs=50, callbacks=[model_checkpoint_callback, early_stopping])
	json.dump(history.history, open('./Basic_Classifier/basicClassifier_history', 'w'))

	# best_model = keras.models.load_model(checkpoint_filepath) # The model (that are considered the best) can be loaded as -

	y_pred = model.predict([Xtext_test_vectorized, Xtitle_test_vectorized])[:,0]
	y_pred = [int(round(y)) for y in y_pred]

	# print(y_pred.shape, y_test.shape, y_pred_res.shape, y_test_res.shape)
	classification_report_un = classification_report(y_test, y_pred)

	print('\n Accuracy: ', accuracy_score(y_test, y_pred))
	print('\nClassification Report')
	print('======================================================')
	print('\n', classification_report_un)
