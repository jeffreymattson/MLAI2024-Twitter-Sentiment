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

**Cleaning and preparation:** 

The data is randomly split into train and test sets to facilitate holdout cross validation, with a test size of 0.25.

**Most important features**

Following are the most important posotive and negative features after getting the feature names out from the TF-IDF Vectorizer.  Note that this is after stemming and lemmetization.

Top Positive Words:
 ['sweet' 'smile' 'awesom' 'congrat' 'glad' 'enjoy' 'great' 'congratul'
 'welcom' 'thank']
Top Negative Words:
 ['sad' 'poor' 'sadli' 'miss' 'suck' 'sorri' 'wish' 'cancel' 'cri' 'sick']
 
 
### Methodology

**Logistic Regression Model:** TF-IDF Vectorizer was used prior to instantiating a Logistic Regression model. GridSearchCV was also used for cross-validation with param_grid = {'C': [0.1, 1, 10]}

**Naive Bayes Model:** TF-IDF Vectorizer was used prior to instantiating a Naive Bayes model. GridSearchCV was also used for cross-validation with params = {  
'alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001)  
}

**Support Vector Classifier Model:** TF-IDF was used prior to instantiating the SVM Model. GridSearchCV is implemented with param_grid = {'C': [0.1, 1, 10]}

**Random Forest Model:** TF-IDF was used prior to instantiating the Random Forest Model.  GridSearchCV is implemented with param = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "criterion": ["gini", "entropy"]}

**XGBooost Model** TF-IDF was used prior to instantiating the XGBoost Model.  GridSearchCV is implemented with params = { 
    'max_depth': [3, 4, 5, 6, 8, 10, 12, 15],
    'learning_rate': [0.001, 0.01, 0.1, 0.20, 0.25, 0.30],
    "gamma":[0, 0.25, 0.5, 0.75,1],
    'n_estimators': [100, 500, 1000],
    "subsample":[0.9],
    "colsample_bytree":[0.5],
    "early_stopping_rounds": [10], 
}

### Model evaluation and results 

Model performance is visualized in a Confusion Matrix, Precision-Recall Curve, and Learning Curve.  Addionally, for each classifier a classification report was printed with the f1-score as the main score.  Logistic Regression and SVM both performed the best with an f1-score of .79.

### Threshold Tuning

An attempt was made to improve performance using threshold tuning.  This improved the f1-score of LogisticRegrawssion from .7878 to .7919.  Norminal but still an improvement.