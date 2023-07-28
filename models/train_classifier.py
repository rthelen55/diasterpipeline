import sys
import numpy as np
import pandas as pd
import os
import re
from sqlalchemy import create_engine
import pickle

from scipy.stats import gmean
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

def load_data(database_filepath):
    """
    Load Data from the Database Function
    
    Arguments:
        database_filepath: Path to SQLite destination database
    Output:
        X: a dataframe containing features
        Y: a dataframe containing labels
        category_names: list of categories name
    """
    engine = create_engine("sqlite:///" + database_filepath)
    conn = engine.connect()
    df = pd.read_sql_table('Disaster_Response', conn)
    X = df['message']
    Y = df.drop(['id','message','original','genre'], axis = 1)
    category_names = Y.columns
    return X, Y, category_names
    
def tokenize(text):
    """
    Tokenize the text function
    
    Arguments:
        text: text that needs to be tokenized
    Output:
        tokens: List of tokens extracted from the provided text
    """

    url_place_holder = "urlplaceholder"
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    urls_found = re.findall(url_regex, text)
    
    for url in urls_found:
        text = text.replace(url, url_place_holder)
    
    text = re.sub(r'[^\w\s]','',text).lower()    
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens

def build_model():
    """
    Build Model

    Output:
        A Scikit ML model that processes text messages and applies a classifier.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    parameters = {
    'clf__estimator__learning_rate': [0.1, 0.2],
    'clf__estimator__n_estimators': [100, 200]
    }
    
    cv = GridSearchCV(estimator = pipeline, param_grid = parameters)
    model = cv
    return model
    
def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate Model

    This function applies a ML pipeline to a test set and prints out the model performance (accuracy and f1score)
    
    Arguments:
        model: A scikit ML model
        X_test: Test features
        Y_test: Test labels
        category_names: label names
    """

    Y_pred = model.predict(X_test)
    print(classification_report(Y_test.values, Y_pred, target_names = category_names))

def save_model(model, model_filepath):
    """
    Save Pipeline function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Arguments:
        model: GridSearchCV
        model_filepath: destination path to save .pkl file
    """
    with open(model_filepath, 'wb') as file:
          pickle.dump(model, file)

def main():
    """
    Train Classifier Main function
    
    This function applies the Machine Learning Pipeline:
        1) Extract data from SQLite db
        2) Train ML model on training set
        3) Estimate model performance on test set
        4) Save trained model as pickle file
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()