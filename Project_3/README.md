# Sentiment Analysis
In this notebook, we will perform sentiment analysis on Vietnamese Shopee Food’s reviews in HCMC with three methods:
- Traditional Machine Learning using scikit-learn
- PySpark ML
- Deep Learning using Pytorch
- Transfer Learning using BERT

The output should solve two problems:
- Predict the sentiment of a review (scale: negative, neutral, positive)
- If the shop owner wants to know the sentiment of his shop, we will calculate the average sentiment of all reviews and give him the result. (Restaurant information, number of positive reviews and
Negatively attached to wordcloud and main keywords related, visualization devices and statistics...)
---
### Traditional Machine Learning
We will build and train multiple machine learning models using scikit-learn and pyspark to test, which is the best. We will use the following models:
- Logistic Regression
- Naive Bayes
- XGBoost
- Support Vector Machine
- Gradient boosting
- Ada Boost

### PySpark Machine Learning
We will build and train multiple machine learning models using pyspark to test, which is the best. We will use the following models:
- Logistic Regression
- Naive Bayes
- RandomForestClassifier
- Decision Tree


### Deep Learning
We will build and train a deep learning model using Pytorch. We will use the following models:
- RNN
- LSTM
- GRU

### Transfer Learning
We will use BERT for sentiment analysis.