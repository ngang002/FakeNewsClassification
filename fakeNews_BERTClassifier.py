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
elif FLAG == 2: 
	from cleanText import text_preprocessing
	from BERTpreprocessor import preprocessBERTClassifier
	from BERTBackbone import BERT_Classifier


if __name__ == "__main__": 

	AUTOTUNE = tf.data.AUTOTUNE
	_PRESET_FLAG = False
	_PRESET_MODEL = "bert_tiny_en_uncased_sst2"
	_NLAYERS = 0
	_MAX_FEATURES = 9600
	_SEQ_LENGTH = 4096
	_HIDDEN_NNODES = 32
	_INTERMEDIATE_NNODES = 32
	_EMBEDDING_DIMS = 32
	_NUM_SEGMENTS = 2
	_DROPOUT = 0.2
	

	theClassifier = BERT_Classifier(VOCAB_SIZE=_MAX_FEATURES, SEQ_LENGTH=_SEQ_LENGTH, NLAYERS=_NLAYERS, NUM_HEADS=4, \
											HIDDEN_DIM=_HIDDEN_NNODES, INTERMEDIATE_DIM=_INTERMEDIATE_NNODES, NUM_SEGMENTS=_NUM_SEGMENTS, \
											DROPOUT=_DROPOUT)

	print(BERT_Classifier.VOCAB_SIZE, BERT_Classifier.SEQ_LENGTH, BERT_Classifier.NLAYERS)
	# Retreiving the data and some preliminary 
	# cleanup stuff 
	fakeNews = pd.read_csv('fakeNews_dataset.csv', sep=',', header=0)
	fakeNews = fakeNews.dropna()

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
		theClassifier = BERT_Classifier()
		SAVDIR = './BERT_Classifier/ckpt'
		if not os.path.isdir(SAVDIR): os.makedirs(SAVDIR)

	X = fakeNews.loc[:, ['text']].values[:1000]
	print(np.shape(X))
	y = fakeNews['label'].values[:1000]
	
	start_time = time.time()
	with Pool(processes=12) as p: output = p.map(text_preprocessing, X)
	print(f"Time to convert text to vectors: {time.time()-start_time}")
	X_clean = np.array(output)

	X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=_RANDOM_STATE)
	y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
	y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)
	print(y_train.shape)
	print(y_test.shape)

	
	if not _PRESET_FLAG:
		# BERTpreprocessor_object, _VOCAB_SIZE = preprocessBERTClassifier(X_train, PREPROCESS_NAME=None, _KERAS_NLP_FLAG=True,
		# 					_VOCAB_SIZE= theClassifier.VOCAB_SIZE, _MAX_SEQ_LEN=theClassifier.SEQ_LENGTH, _CREATE_VOCAB_FILE=False, \
		# 					_VOCAB_FILE='BERT_Classifier/fakeDataset_vocabulary.txt', )
		BERTpreprocessor_object, _VOCAB_SIZE = preprocessBERTClassifier(X_train, PREPROCESS_NAME="bert_tiny_en_uncased_sst2", _KERAS_NLP_FLAG=True,
								_VOCAB_SIZE= theClassifier.VOCAB_SIZE, _MAX_SEQ_LEN=theClassifier.SEQ_LENGTH, _CREATE_VOCAB_FILE=False, \
								_VOCAB_FILE=None, )
		print(BERTpreprocessor_object)
		print(f"Size of returned vocabulary {_VOCAB_SIZE}")
		X_vectorized = BERTpreprocessor_object(X_train)
		print(X_vectorized.keys())


		BERTbackbone_object = theClassifier.buildBERTBackbone_fromBackbone()
		# MODEL = keras_nlp.models.BertMaskedLM(BACKBONE, preprocessor=BERTpreprocessor_object)
		MODEL = keras_nlp.models.BertClassifier(BERTbackbone_object, preprocessor=BERTpreprocessor_object, \
													num_classes=2, activation='sigmoid')
		

	elif _PRESET_FLAG:
		# Pretrained classifier.
		MODEL = keras_nlp.models.BertClassifier.from_preset(_PRESET_MODEL,
		    						num_classes=2, load_weights=True, activation='sigmoid')

	tf.keras.utils.plot_model(MODEL, to_file=f'{SAVDIR}/BERTClassifier.png',  show_shapes=True)
	
	checkpoint_filepath = f'{SAVDIR}/checkpoint.weights.h5'
	model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    		filepath=checkpoint_filepath,
    		save_weights_only=True,
    		monitor='val_accuracy',
    		mode='max',
    		save_best_only=True)

	# creating a stopping point 
	early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
	theClassifier.set_optimizer(learning_rate=1E-3, epsilon=1E-5)

	# Re-compile (e.g., with a new learning rate).
	MODEL.compile(
    loss=theClassifier._LOSS,
    optimizer=theClassifier._OPTIMIZER,
    jit_compile=True,
    metrics=["accuracy"],
	)
	MODEL.backbone.trainable = False
	MODEL.summary()

	MODEL.fit(x=X_train, y=y_train, validation_split=0.2, epochs=50)

	y_pred = MODEL.predict(X_test)[:,0]
	print(y_pred)
	y_pred = [int(round(y)) for y in y_pred]

	# print(y_pred.shape, y_test.shape, y_pred_res.shape, y_test_res.shape)
	classification_report_un = classification_report(y_test, y_pred)

	print('\n Accuracy: ', accuracy_score(y_test, y_pred))
	print('\nClassification Report')
	print('======================================================')
	print('\n', classification_report_un)



