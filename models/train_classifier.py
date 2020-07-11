# import packages
import sys


import sqlite3
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import AdaBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.stem.porter import PorterStemmer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    """
    Load Data 
    Arguments:
    ----
        database_filepath: To load a database of disaster messages
    Output:
    ----
        X -> pd.Dataframe with the messages
        Y -> pd.Dataframe  with  a column for every categories
        category_names -> List with the names of categories
    """
    # load to database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('df',engine)

     # drop columns with null, new to avoid problems str
    df = df[~(df.isnull().any(axis=1))|((df.original.isnull())&~(df.offer.isnull()))]  
    
    # define features and label arrays
    X = df['message']
    Y = df.iloc[:,4:]
    # Y = df.iloc[:,4:].values
    #Y = df.drop(['original','genre','message','offer','request'], axis=1)
    Y= Y.astype(int)
    category_names = list(df.columns[4:])

    return X, Y, category_names



def tokenize(text):
    '''
    Tokenize function simple 
    Arguments:
    ----
        text: text messages
    Output:
    ----
        The tokens, after Tokenized the text, lower and lemmatized text
    '''
    print("tokenize")
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Model function minipal to avoid problem of time and space 
 
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(class_weight='balanced')))
    ])

    # define parameters for GridSearchCV
    parameters = {'clf__estimator__min_samples_leaf': [1, 5],
                  'clf__estimator__min_samples_split': [2, 10],
                  'clf__estimator__n_estimators': [10, 50]}

    # create gridsearch object
    #model = GridSearchCV (pipeline, param_grid= parameters, verbose =7 )    
    model = pipeline

    return model
    



def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test.iloc[:,1:].values, np.array([x[1:] for x in Y_pred]), target_names=category_names))
    #for i in range(len(category_names)):
        #print('Category: {} '.format(category_names[i]))
        #print(classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        
        

def save_model(model, model_filepath):
    """ 
    Saving model with pickle
    """
    pickle.dump(model, open(model_filepath, 'wb'))
    pass

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        #print(X.head(2))
        #print(Y.head(2))
        print(category_names)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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
