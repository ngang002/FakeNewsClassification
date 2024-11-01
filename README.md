# FakeNewsClassification


In today's world one of the most prevalent issues is the presence of Fake News across all platforms of media: television, social media, word-of-mouth, and internet.  Combatting fake news is a monumental task, especially in the age of the internet where one lie can set off a domino effect.  The goal of this Repository is to make an attempt at identifying fake news articles from articles that I scraped from online sources, including online news webpages (like the ) using the Beatiful Soup API.  However, due to the large volume of data necessary for this task I also supplemented my relatively small dataset (1,000 articles) with others gathered online from multiple sources to obtain a much larger ~70,000 dataset, that is gigabytes in size.  

The primary goal of this project was to observe the percentage of my media diet consisted of fake news articles (whether I read the article or not).  It then extended beyond my own media diet, as this was not a problem that only impacts me but society as a whole.  To this end, this repository is structured as follows: 

1. Data Exploration
2. Cleaning the Data (Stopwords, Lemmatization, Tokenization) 
3. Applied Machine Learning Algorithms
4. Results
5. Conclusions


## Data Exploration ##
![alt text](https://github.com/ngang002/FakeNewsClassification/origin/main/real_vs_fake.png?raw=true)


## Cleaning the Data (Stopwords, Lemmatization, Tokenization) ##

In this section I will talk about the process by which I cleaned the data.  As I mentioned previously I compiled the data into a format which included: 
  1. Title
  2. Text
  3. Classification
I then implemented a cleaning method that removed 

## Applied Machine Learning Algorithms ##
I first tested an term frequency–inverse document frequency (TF-IDF) vectorization methodology.  In this methodology the 
## Results ## 

## Conclusions and Future Work ##
