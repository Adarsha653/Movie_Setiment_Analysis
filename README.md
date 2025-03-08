# What this project does?
 Asks users to enter movie review and then predicts whether the movie is positive or negative

# Steps performed to create this project
1. Downloaded IMDB movies dataset from gaggle containing 50,000 labelled movie review sentiments. Link: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
2. Pre-processed the data: removed stop-words, punctuations, numbers, lemmatised, tokenised texts and then performed TF-IDF vectorisation.
3. Used word cloud for EDA
4. Used logistic regression model to train the data and checked accuracy with accuracy_score, classification report and confusion matrix
5. Dumped model and TF-IDF with pickle
6. Created an app.py with flask, index.html and style.css to create a website for users to enter their sentiments and predict if they are positive or negative.

