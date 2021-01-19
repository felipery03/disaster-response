import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FilterColumns(BaseEstimator, TransformerMixin):
    ''' Transformer to filter columns in an input dataframe.
    params:
    col_names(list): List with columns names to filter
    dim (int): If dim equals 1 returns a series
        else returns a dataframe
    '''
    def __init__(self, col_names=None, dim=1):
        self.col_names = col_names
        self.dim = dim

    def fit (self, X, y=None):
        return self

    def transform(self, X):
        if (self.dim == 1):
            # Return a series
            result = X[self.col_names]
        else:
            # Return a dataframe
            result = X[[self.col_names]]
        
        return result

class CountTokens(BaseEstimator, TransformerMixin):
    ''' Transformer that receives countvectorizer and returns 
    number of tokens per row.
    
    '''    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_len = X.sum(axis=1)
        
        return X_len