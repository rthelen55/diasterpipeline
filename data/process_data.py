import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, left_on='id', right_on='id', how='inner')
    return df
    
def clean_data(df):
    categories = df["categories"].str.split(';', expand = True)
    col_names = categories.iloc[0,:].apply(lambda x: x.split('-')[0])
    categories.columns = col_names
    
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
    
    df = df.drop('categories', axis = 1)
    df = pd.concat([df, categories], axis = 1)
    df.drop_duplicates(inplace = True)
    df.drop(['child_alone'], axis = 1, inplace = True) #all values are zero so not needed
    df['related'] = df['related'].apply(lambda x: 1 if x == 2 else x) # replace 2's with 1's
    
    return df
     
    


def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Disaster_Response', engine, index=False, if_exists = 'replace')


def main():
    
    messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
    
    if len(sys.argv) == 4:       

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        print(df.describe())
        
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