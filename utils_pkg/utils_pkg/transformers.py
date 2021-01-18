import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FilterColumns(BaseEstimator, TransformerMixin):
    ''' Transformer to filter columns in an input dataframe.
    
    '''
    def __init__(self, col_names=None):
        self.col_names = col_names
    def fit (self, X, y=None):
        return self
    def transform(self, X):
        return X[self.col_names]

class CountTokens(BaseEstimator, TransformerMixin):
    ''' Transformer that receives countvectorizer and returns 
    number of tokens per row.
    
    '''    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_len = X.sum(axis=1)
        
        return X_len

class GenreMessage(BaseEstimator, TransformerMixin):
    ''' Transformer that receives input dataframe, extract genre column
        and transform it in a hot-encoding.
    
    '''    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_genre_dummies = pd.get_dummies(X['genre']).values
        
        return X_genre_dummies