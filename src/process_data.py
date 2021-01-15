import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    ''' Load messages and categories dataset and merge both

    Params:
    messages_filepath (string): Filepath of messages dataset
        including dataset ref. and extention
    categories_filepath (string): Filepath of categories
        dataset including dataset ref. and extention

    Returns:
    data (dataframe): Dataframe merged with messagens and
        categories info
    '''

    # Load messages dataset
    messages = pd.read_csv(messages_filepath)

    # Load categories dataset
    categories = pd.read_csv(categories_filepath,
        delimiter=',')

    # Merge both datasets
    df = messages.merge(categories, how='left', on='id')

    return df

def clean_data(df):
    ''' Clean categories data in input dataframe. Splitting
    categories column, expanding them to other columns,
    renaming each one and cleanning value content. After,
    unnececery columns are dropped and duplicate lines 
    are removed.  

    Params:
    df (dataframe): Input dataframe with one column named
    'categories'

    Returns:
    data (dataframe): Output cleanned dataframe
    '''

    # Copy data
    data = df.copy()

    # Split categories in a unique row and expand throw 
    # columns
    categories = data.categories.str.split(';', expand=True)

    # Rename columns
    row = categories.head(1)
    get_title_func = lambda x: x[0].split('-')[0]
    category_colnames = row.apply(get_title_func).tolist()
    categories.columns = category_colnames

    for column in categories:
        # Remove infos from categories values
        categories[column] = categories[column].str.split('-').str[1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')

    # Drop unnecessary columns
    data.drop(['categories', 'id', 'original'], axis=1, inplace=True)
    
    # Merge cleaned categories and main dataframe
    data = pd.concat([data, categories], axis=1)

    # Drop duplicates
    data.drop_duplicates(inplace=True)

    return data

def save_data(df, database_path, table_name):
    ''' Save a dataframe in a sqlite db, creating a new table
    with table_name.

    Params:
    df (dataframe): Input dataframe
    database_path (string): Database path including database name
        and extension
    table_name (string): Table name which data will be inputed
    '''
    
    engine = create_engine('sqlite:///' + database_path)

    try:
        df.to_sql(table_name, engine, index=False, if_exists='replace')

    except(e):
        print(e)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath, 'messages')
        
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