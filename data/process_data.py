# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load the files, read the messages and categories files and merge both dataframe .
    Return the dataframe with messages and the categories of each message.
    Arguments:
    ----
        messages_filepath: path to read “disaster_messages.csv” file
        categories_filepath: path to read “disaster_categories.csv” file
    Output:
    ----
        Merge dataframe of both files
    """

    # load message datasets
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages, categories, on='id')



    return df


def clean_data(df):
    """
    Process a Datatase of messages and categories: after clean and drop duplicates
    The messages are classified in categories in numerical format 
    Arguments:
    ----
        Dataframe from after execute load data
    Output:
    ----
        Dataframe of messages and categories clean    
    """

    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat = ';', expand = True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames =[]
    for r in row:
      category_colnames.append(r[:-2])

    categories.columns = category_colnames

    #convert to dummies
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    
    # convert column from string to numeric
    categories[column] = pd.to_numeric(categories[column])
    #categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    Save the clean dataframe into a sqlite database
    Arguments:
    ----
        Dataframe from after clean 
    Output:
    ----
        You can access to a database with a Database_filename    
    """
    
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('df', engine, index=False)
    pass


def main():
    """
    Runs load_data, clean_data and
    save_data and the result is a database of messages and categories data
    """
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