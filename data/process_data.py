import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    load_data() will take message and categories files filepath and return a Dataframe 
    Input:
        messages_filepath - filepath for the message csv file
        categories_filepath - filepath for the categories csv file
    output:
        df - merged dataframe by 'id' column
    '''
    # load messages csv file
    messages = pd.read_csv(messages_filepath)
    # load categories csv file
    categories = pd.read_csv(categories_filepath)
    # merge the two files into one dataframe using 'id' column
    df = pd.merge(messages, categories, on='id')
    # return the merged dataframe
    return df


def clean_data(df):
    '''
    clean_data() will take dataframe and return it after cleaning 
    Input:
        df - dataframe that required cleaning
    output:
        df - after cleaning
    Description:
        This function will take dataframe then start cleaning it and prepare it for training and ML algorithm.
        The cleaning process will follow the following steps:
            1- split category columns into a seperate column for each category then delete 'category' column
            2- drop duplicates
            3- drop columns that null values represents more than 25% of the data 
            4- drop columns that contain a single value
            5- drop abnormal values.
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe to extract a list of new column names for categories.
    row = categories.iloc[0]
    # use for loop to extract the column names 
    category_colnames = [x.split('-')[0] for x in row]
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # Convert category values to just numbers 0 or 1
        categories[column] = categories[column].str.split('-',expand=True)[1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    # Replace categories column in df with new category columns.
    # first, drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)
    # then, concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    # drop duplicates
    df.drop_duplicates(subset='id', keep='first', inplace=True)
    # Drop columns that null values represent more than 25% of the data and dropping it will not impact our data
    df.drop(columns=['original'], inplace=True)
    # Drop columns that cantain a single value
    df.drop(columns=['child_alone'], inplace=True)
    # Drop abnormal values in 'related' column and only keep 0 and 1
    df = df[df['related']!=2]
    return df


def save_data(df, database_filename):
    '''
    save_data() will take the cleaned dataframe and save it into a database
    Input:
        df - cleaned dataframe
        database_filename - filename of the database that database that the dataframe will be saved into 
    output:
        None
    '''
    # save the dataframe in SQLite
    # create the SQLite engine
    engine = create_engine('sqlite:///'+database_filename)
    # Write the Dataframe into Disaster_Msg table
    df.to_sql('Disaster_Msg', engine, index=False, if_exists='replace')
    return


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
