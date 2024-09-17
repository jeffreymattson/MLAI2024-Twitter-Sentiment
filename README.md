# MLAI2024-Twitter-Sentiment

## Using Text Classification to Detect Sentiment on Twitter 

**Jeffrey Mattson**

### Executive summary

**Project overview and goals:** The goal of this project is to detect the sentiment of a tweet on twitter.  We will start with preprocessing the data using by by tokenizing and removing stop words.  We will then vextorize the text using TF-IDF Vectorizer to convert it into numerical form that machine learning models can understand.  The we will continue by training various text classification models to predict a positive or negative sentiment of the tweet.  After that we will train different classifiers and avaluade based on Accuary, Precision, Recall, and F1-Score.


**Findings:** The best model for twitter sentiment prediction is the Support Vector Machine, with an accuracy score of 0.75, a recall of 0.84, and an F-1 of 0.79 based on the dataset we trained with.  Logistic Regression is a close second with nearly identical scores, varying only slightly in recall.  

### Research Question

The question this project addresses is to determine what is the best model for determining the sentimeent of a tweet on twitter

### Data Source

**Dataaset:** The dataset used in this project is available from Kaggle at https://www.kaggle.com/c/twitter-sentiment-analysis2/data?select=train.csv

It consists of 99,989 unique text comments each associated with a binary 'Sentiment' value.  

