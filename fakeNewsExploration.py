import numpy as np
import pandas as pd
_RANDOM_STATE = 108

import re
import string

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow_text as tf_text

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def getOccurrences(text):
	"""
	This applies NLP to retain only words (consisting of elements from the English alphabet)
	and retains words with some kind of meaning and removing "stopping words".  This also 
	strips away punctuation, and using the WordNetLemmatizer converts words with '-ing"'' or '-ly'
	into nouns instead to given more meaning 
	# I could probably multiprocess this too to speed things up 
	"""
	return len(re.findall('[#@*](\S+)', text)), len(re.findall('^http(\S+)', text))

fakeNews_dataset = pd.read_csv('fakeNews_Dataset.csv', sep=',', header=0)
print(f"Number of news articles: {len(fakeNews_dataset)}")
fakeNews_dataset = fakeNews_dataset.dropna()
print(f"Number of news articles: {len(fakeNews_dataset)}")
X = fakeNews_dataset.loc[:,'text']
X_title = fakeNews_dataset.loc[:,'title']
y = fakeNews_dataset['label']

labels, counts = np.unique(y, return_counts=True)
# [print(title) for title in X_title[:10000]]

X = X.values
y = y.values
X_len = np.array([len(text.split()) for text in X])
X_len_title = np.array([len(text.split()) for text in X_title])
# X_par = np.array([text.count('\n') for text in X])
print(len(X_len), len(X_len_title))

fig, ax = plt.subplots(figsize=(6,6))
ax.pie(counts, labels=['True News', 'Fake News'], autopct='%1.1f%%', colors=['lightskyblue', 'salmon'])
fig.savefig("./DataExploration_Plots/real_vs_fake.pdf")

X_occurrences = np.array([getOccurrences(text) for text in X])
# print(X_occurrences)

# print(set(stopwords.words('english')))
BINS = np.linspace(0, 600, 21)
hist_Fake, _ = np.histogram(X_len[y==0], bins = BINS, density=True)
hist_Real, _ = np.histogram(X_len[y==1], bins = BINS, density=True)
LENGTH_BINS = (BINS[:-1]+BINS[1:])/2.

BINS_TITLE = np.linspace(0, 40, 21)
histTitle_Real, _ = np.histogram(X_len_title[y==1], bins = BINS_TITLE, density=True)
histTitle_Fake, _ = np.histogram(X_len_title[y==0], bins = BINS_TITLE, density=True)
BINS_TITLE = (BINS_TITLE[:-1]+BINS_TITLE[1:])/2.


BINS = np.linspace(0, 100, 11)
newLine_Fake, _ = np.histogram(X_occurrences[y==0, 0], bins = BINS, density=True)
newLine_Real, _ = np.histogram(X_occurrences[y==1, 0], bins = BINS, density=True)
NEWLINE_BINS = (BINS[:-1]+BINS[1:])/2.

fig, ax = plt.subplots(figsize=(12,6), ncols=2)

ax[0].plot(LENGTH_BINS, hist_Fake, c='salmon', label='Fake Articles')
ax[0].plot(LENGTH_BINS, hist_Real, c='lightskyblue', label='Real Articles')
ax[1].plot(BINS_TITLE, histTitle_Fake, c='salmon', ls='--', label='Title Fake Articles')
ax[1].plot(BINS_TITLE, histTitle_Real, c='lightskyblue', ls='--', label='Title Real Articles')
ax[0].legend(loc='best')
# ax.plot(NEWLINE_BINS, newLine_Fake, c='r', ls='--',)
# ax.plot(NEWLINE_BINS, newLine_Real, c='b', ls='--',)
ax[0].set_xlabel('Number of Words in Article')
ax[0].set_ylabel('P(# of words)')

ax[1].set_xlabel('Number of Words in Title')
ax[1].set_ylabel('P(# of words)')

fig.savefig('./DataExploration_Plots/length_Of_articles.pdf')