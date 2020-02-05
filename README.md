# Disaster Response Pipeline Project
This project will build a webapp for creating a machine learning pipeline to categorize real messages that were sent during disaster events so that you can send the messages to an appropriate disaster relief agency.
This project contains the following:

1. ETL Pipeline (data folder)
In a Python script, process_data.py, the following actions will be performed:
    - Loads the messages and categories datasets
    - Merges the two datasets
    - Cleans the data
    - Stores it in a SQLite database
    
2. ML Pipeline (models folder)
In a Python script, train_classifier.py, the following actions will be performed:
    - Loads data from the SQLite database
    - Splits the dataset into training and test sets
    - Builds a text processing and machine learning pipeline
    - Trains and tunes a model using GridSearchCV
    - Outputs results on the test set
    - Exports the final model as a pickle file
    
3. Flask Web App (app folder)
In a Python script, run.py, the following actions will be performed:
    - open an Flask app
    - Receive a messgae
    - categorize the message


Folder Hierarichy:
- app
 - template
    - master.html  # main page of web app
    - go.html  # classification result page of web app
 - run.py  # Flask file that runs app

- data
 - disaster_categories.csv  # data to process 
 - disaster_messages.csv  # data to process
 - process_data.py
 - InsertDatabaseName.db   # database to save clean data to

- models
 - train_classifier.py
 - classifier.pkl  # saved model 

- README.md


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
    - to run the webapp
        'python app/run.py'

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


