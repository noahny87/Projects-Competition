# Import necessary libraries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import string

# Function to preprocess text
def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if word not in set(stopwords.words("english"))]
    text = ' '.join(text)
    return text

# Function to train Naive Bayes model
def train_naive_bayes(X_train, y_train):
    cv = CountVectorizer()
    X_train_transformed = cv.fit_transform(X_train).toarray()
    model = MultinomialNB()
    baggingmodel = BaggingClassifier(estimator=model, n_estimators=10)
    baggingmodel.fit(X_train_transformed, y_train)
    return baggingmodel, cv

# Function to train Logistic Regression model
def train_logistic_regression(X_train, y_train):
    lg_model = LogisticRegression(max_iter=1000)
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['saga'],
        'max_iter': [5000, 10000]
    }
    grid_search = GridSearchCV(lg_model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    model_best = grid_search.best_estimator_
    return model_best

# Function to evaluate model
def evaluate_model(model, cv, X_test, y_test):
    X_test_transformed = cv.transform(X_test).toarray()
    y_pred = model.predict(X_test_transformed)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Main function to run the script
def main():
    # Import data
    data = pd.read_csv("C:\\Users\\noahn\\OneDrive\\Data\\Training Data.csv")
    data['Review'] = data['Review'].apply(preprocess)
    X_train, X_test, y_train, y_test = train_test_split(data['Review'], data['Liked'], test_size=0.25, random_state=42)

    # Train and evaluate Naive Bayes model
    nb_model, cv = train_naive_bayes(X_train, y_train)
    evaluate_model(nb_model, cv, X_test, y_test)

    # Transform X_train for Logistic Regression model
    X_train_transformed = cv.transform(X_train).toarray()
    # Train and evaluate Logistic Regression model
    lr_model = train_logistic_regression(X_train_transformed, y_train)
    evaluate_model(lr_model, cv, X_test, y_test)

if __name__ == "__main__":
    main()