# FakeNewsClassification


In today's world one of the most prevalent issues is the presence of Fake News across all platforms of media: television, social media, word-of-mouth, and internet.  Combatting fake news is a monumental task, especially in the age of the internet where one lie can set off a domino effect.  The goal of this Repository is to make an attempt at identifying fake news articles from articles that I scraped from online sources, including online news webpages (like the ) using the Beatiful Soup API.  However, due to the large volume of data necessary for this task I also supplemented my relatively small dataset (1,000 articles) with others gathered online from multiple sources to obtain a much larger ~70,000 dataset, that is gigabytes in size.  

The primary goal of this project was to implement NLP processes and ML/AI algorihtms to predict whether the news articles that I collected, which are often a part of my own media diet, are real news and fake news.  While this project may be small, it will demonstrate my ability to use NLP practices to solve real-world applicable issues, that are very pertinint today.  It then extended beyond my own media diet, as this was not a problem that only impacts me but society as a whole.  To this end, this repository is structured as follows: 

1. Data Exploration
2. Cleaning the Data (Stopwords, Lemmatization, Tokenization) 
3. Applied Machine Learning Algorithms
4. Results
5. Conclusions


## Data Exploration ##

As mentioned previously there are 70,000 articles in the set I was able to accumulate.  However, the data that I am using for this work is unfortunately too large to store on Github, so I will explain the structure here.  

These data are structured as: 
  1. Title
  2. Text
  3. Label (0 for fake, 1 for real) 



![In this figure ](DataExploration_Plots/real_vs_fake.png)

In this figure I show the number of real news vs fake news

![In this figure](DataExploration_Plots/length_Of_articles.png)



## Cleaning the Data (Stopwords, Lemmatization, Tokenization) ##

I then implemented a cleaning method that involved 3 major steps (all of these processes make use of the [NLTK Library](https://pythonspot.com/nltk-stop-words) ):
  1. Removing stopwords: Stop words like "a", "the", "in", etc.  These words generally are very high frequency words, and this frequency leads to very little meaning when this word is coupled with their "environments".  For example in this sentence, the words "for" and "the" hold very little meaning, and should we get rid of them, this sentence should largely be understood by the machine
  2. Lemmatization: Lemmatization is the process by which an NLP algorithm converts adverbs, adjectives, and other grammatical categories into "their" base worlds.  For example in the world "lovely" the base word is "love", indicative of a positive sentiment.  However, the suffix "-ly" simply denotes a different grammatical use of the word "love".  In short lemmatizing sorts words by grouping inflected or variant forms of the same word.  For the purposes of this work, I used the WordNetLemmatizer
  3. 
## Applied Machine Learning Algorithms ##
I first tested an term frequency–inverse document frequency (TF-IDF) vectorization methodology.  In this methodology the 
## Results ## 

## Conclusions and Future Work ##
