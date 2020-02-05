# import nltk and download punkt and wordnet
import nltk
nltk.download(['punkt', 'wordnet']);

import sys
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer



import pandas as pd
import numpy as np
import matplotlib as plt
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
import pickle


# 
def load_data(database_filepath):
    '''
    load_data() funtion to load Dataframe from SQLite database
    Input:
        database_filepath - database name and filepath
    output:
        X: message column in dF
        Y: categories columns in df
        category_names : categories names
    '''
    # create SQLite engine and read the database
    engine = create_engine('sqlite:///'+database_filepath)
    # read the table Disaster_Msg that was cleaned in process_data.py
    df = pd.read_sql_table("Disaster_Msg", engine)
    # assign the input column which contains the message to X
    X = df['message']
    # assign the categories columns to Y
    Y = df[df.columns[4:]]
    # get the categories names
    category_names = Y.columns.values
    return X, Y, category_names


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tokens.append(lemmatizer.lemmatize(tok).lower().strip())
    return clean_tokens


def build_model():
    '''
    build_model() build ML model using Pipeline
    Input:
        None
    output:
        Pipeline model
    '''
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, max_df= 1.0, ngram_range= (1, 2))),
        ('tfidf', TfidfTransformer(use_idf= True)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(min_samples_split= 6, n_estimators= 25)))
    ])
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate_model() to test our ML algorithm effecincy and print the f1_score, precision and recall results
    Input:
        model: ML model that was built in build_model()
        X_test: the sample was taking from X for testing reasons
        Y_test: the sample was taking from Y for testing reasons
        category_names: categories names
    Output:
        None
    Discription:
        This funtion will takes inputs and use the trained model to predict the testing samples to check the accuracy of the model.
        Then, will print the f1_score, precision and recall results
    '''
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    '''
    save_model() to save trained model as a pickle file
    Input:
        model: ML model that was trained
        model_filepath: the filepath where the trained model will be saved
    Output:
        None
    Discription:
        This funtion will takes the trained model and filepath to save the model as a pickle file in the model_filepath
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
        
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
