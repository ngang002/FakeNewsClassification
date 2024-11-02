import numpy as np
import pandas as pd

import re
import string

import matplotlib.pyplot as plt

import tensorflow as tf 
from tensorflow.keras import Sequential
import tensorflow_text as tf_text
import tensorflow_hub as tf_hub

import keras
import keras_nlp
from keras import layers, losses
from tensorflow.keras import Input, Model

import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class TFIDF_Classifier(keras.Model):

	def __init__(self, _OPTIMIZER='Adam', _LOSS = keras.losses.BinaryCrossentropy(from_logits=True) \
								, _MAX_FEATURES=2000, _SEQ_LENGTH=512, _EMBEDDING_DIM=16, **kwargs):
		
		# Input stuff 
		self._NUM_CLASSES = 2
		self._MAX_FEATURES = _MAX_FEATURES
		self._SEQ_LENGTH = _SEQ_LENGTH
		self._EMBEDDING_DIM = _EMBEDDING_DIM

		self.optimizer = _OPTIMIZER
		self.loss = _LOSS


	def create_VectorizationLayer(self):

		return layers.TextVectorization(
			standardize=TFIDF_preprocess,
			max_tokens=self._MAX_FEATURES,
			output_mode='tf-idf',
			output_sequence_length=self._SEQ_LENGTH, 
			pad_to_max_tokens=True,
			name='Vectorize')


# Defining a Function to clean up the reviews 
@keras.saving.register_keras_serializable()
def TFIDF_preprocess(text):
	"""
	This applies NLP to retain only words (consisting of elements from the English alphabet)
	and retains words with some kind of meaning and removing "stopping words".  This also 
	strips away punctuation, and using the WordNetLemmatizer converts words with '-ing"'' or '-ly'
	into nouns instead to given more meaning 
	# I could probably multiprocess this too to speed things up 
	"""

	# Ensures that whatever I put into the text is a string 
	if type(text) != type(str): text = str(text)

	tf.strings.regex_replace(text, '[%s]' % re.escape(string.punctuation), '')

	main_words = re.sub('[^a-zA-Z]', ' ', text) # This only keeps the alphabet in a sequence of texts 
	main_words = (main_words.lower()).split()   # converts all the letters to lowercase letters  
	main_words = [w for w in main_words if not w in set(stopwords.words('english'))] # Remove stopwords (a/an, the, , pronouns,)

	lem = WordNetLemmatizer()
	main_words = [lem.lemmatize(w) for w in main_words if len(w) > 1] # Group different forms of the same word

	main_words = ' '.join(main_words)
	text = main_words

	return text


def vectorize_text(text, vecLayer):
	text = tf.expand_dims(text, -1)
	return vecLayer(text)

def vectorize_text_MP(multproc_wrapper): 
	text, vecLayer = multproc_wrapper[0], multproc_wrapper[1]
	return vecLayer(tf.expand_dims(text, -1))