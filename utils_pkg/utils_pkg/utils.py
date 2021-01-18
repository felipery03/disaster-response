import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import re

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

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

def tokenize(text):
    ''' Prep a text data casting to lowercase,
        remove punctuation, tokenizing,
        removing stop_words and lemmatinzing.

    Params:
    text (string): Input text data

    Returns:
    result (list): Processed tokens list 
    '''

    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    result = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    result = word_tokenize(result)
    result = [lemmatizer.lemmatize(w) for w in result if w not in stop_words]
    
    return result
