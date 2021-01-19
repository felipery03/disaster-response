import sys
import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
import pickle
import time
from utils_pkg.utils import tokenize, save_data
from utils_pkg.transformers import FilterColumns, CountTokens

def load_data(database_filepath, table_name):
    ''' Load data from sqlitedatabase.

    Params:
    database_filepath (string): Path with sqlite database
        including database name and extension
    table_name (string): Table name to be loaded

    Returns:
    X (Series): Series with message content
    Y (dataframe): Dataframe with targets
    category_names (list): List with all categories
    '''
    # Create engine
    engine = create_engine('sqlite:///' + database_filepath)
    
    # Load data
    df = pd.read_sql_table(table_name, engine)

    # Get only features
    X = df[['message', 'genre']].copy()

    # Get only targets
    Y = df[df.columns[4:]].copy()

    category_names = list(Y.columns)

    return (X, Y, category_names)
   
def build_model(tunning=False):
    ''' Setup pipeline model.

    Params:
    tunning (boolean): If True, the output will be a RandomSearchCV
    with different models and params
    Returns:
    pipeline (model): Pipeline preprocessing and model config steps
    '''
    pipeline = Pipeline([
    ('features', FeatureUnion([
        ('genre_feat', Pipeline([
            ('filter_genre', FilterColumns('genre', dim=2)),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))   
        ])),
        ('txt_feats', Pipeline([
            ('filter_msg', FilterColumns('message', dim=1)),
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('union_txt_feats', FeatureUnion([
                ('tfidf', TfidfTransformer()),
                ('count_tokens', CountTokens())        
            ]))
        ]))
    ])),
    ('clf', MultiOutputClassifier(LogisticRegression(class_weight='balanced', max_iter=1000), n_jobs=5))
     ])

    # Run RandomSearch
    if tunning:

        parameters = [
        {'clf': [MultiOutputClassifier(LogisticRegression(class_weight='balanced', max_iter=1500))],
        'clf__estimator__C': [0.1, 1.0, 10.0], 
        'features__txt_feats__vect__ngram_range': ((1, 1), (1, 2)),
        'features__txt_feats__vect__max_df': (0.5, 0.75, 1.0),
        'features__txt_feats__vect__max_features': (None, 5000, 10000),
        'features__txt_feats__union_txt_feats__tfidf__use_idf': (True, False),
        'features__transformer_weights': (
            {'genre_feat': 1, 'txt_feats': 0.5},
            {'genre_feat': 0.5, 'txt_feats': 1},
            {'genre_feat': 0.8, 'txt_feats': 1},
        )    
        },
        {    
        'clf': [MultiOutputClassifier(RandomForestClassifier(class_weight='balanced', random_state=0))],
        'clf__estimator__n_estimators': [10, 100, 250, 1000],
        'clf__estimator__max_depth':[5, 8, 10],
        'features__txt_feats__vect__ngram_range': ((1, 1), (1, 2)),
        'features__txt_feats__vect__max_df': (0.5, 0.75, 1.0),
        'features__txt_feats__vect__max_features': (None, 5000, 10000),
        'features__txt_feats__union_txt_feats__tfidf__use_idf': (True, False),
        'features__transformer_weights': (
            {'genre_feat': 1, 'txt_feats': 0.5},
            {'genre_feat': 0.5, 'txt_feats': 1},
            {'genre_feat': 0.8, 'txt_feats': 1},
        )
        }
        ]

        # Random Search configs
        model = RandomizedSearchCV(pipeline,
                            param_distributions=parameters,
                            cv=2,
                            random_state=0,
                            n_jobs=5,
                            n_iter=20,
                            verbose=3,
                            scoring = 'f1_weighted'
                            )
    
    else:
        model = pipeline

    return model

def get_results(y_true, y_pred):
    ''' Calculate precision, recall, f1-score and supports
        in class '1' for each predicted target.
    
    Params:
    y_true (array):
    y_pred (array):

    Returns:
    results (dict): Dict with key as target label and value
        as a list with precision, recall, f1-score and 
        supports respectively
    '''
    
    results = dict()
    
    for i in range(y_true.shape[1]):
        score = precision_recall_fscore_support(y_true.values[:, i], y_pred[:, i])
        precision = round(score[0][1], 2)
        recall = round(score[1][1], 2)
        f1 = round(score[2][1], 2)
        support = score[3][1]
                    
        results[y_true.columns[i]] = [precision, recall, f1, support]
    
    return results

def calc_weighted_metric(df, metric_col, vol_col):
    ''' Calculate mean of 'metric_col' weighted by 
        'vol_col'.

    Parameters:
    df (dataframe): Input dataframe
    metric_col (string): Column name in df with metric to
        be calculated
    vol_col (string): Column name in df with weights to
        be used in mean weighted.
    
    Returns:
    mean_w (float): Result of mean weighted calculation
    '''

    mean_w = sum(df[metric_col] * df[vol_col])/(df[vol_col].sum())

    return mean_w 

def evaluate_model(model, X_test, y_test, category_names):
    ''' Calculate metrics for test set as precision, recall, f1-score and
    support. Print values for each label and calculate f1 weighted mean for
    positive class.

    Params:
    model (Predictor): Model already fitted
    X_test (dataframe): Features for test data
    y_test (dataframe): Targets for test data
    category_names (list): List of target labels

    Outputs:
    results_df (dataframe): Dataframe with precision, recall, f1-score,
    and supports in positive class for each target

    '''
    # Predict to testset
    y_pred = model.predict(X_test)

    # Calculate metrics
    results = get_results(y_test, y_pred)
    results_df = pd.DataFrame(results, columns=category_names).transpose().reset_index()

    results_df.columns = ['category', 'precision', 'recall', 'f1', 'support']
    
    # Print precision, recall and f1-score
    for category in category_names:
        print('{}:\nprecision: {}\nrecall: {}\nf1: {}\nsupport: {}\n\n'.format(category, *results[category]))

    # Calculate mean weighted of f1-score
    f1_mean = round(calc_weighted_metric(results_df, 'f1', 'support'), 4)
    print(f'F1-score weighted mean: {f1_mean}')

    return results_df

def save_model(model, model_filepath):
    ''' Save model fitted in a pickle file.

    Params:
    model (model): Model fitted.
    model_filepath (string): Path to save pickle file
    '''
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath, 'messages')

        # Split data in train and test set
        X_train, X_test, Y_train, Y_test = train_test_split(X,
            Y,
            test_size=0.3,
            random_state=0)

        print('Building model...')
        model = build_model(tunning=False)
        
        print('Training model...')
        start = time.time()
        model.fit(X_train, Y_train.values)
        print(time.time() - start)
        
        print('Evaluating model...')
        results = evaluate_model(model, X_test, Y_test, category_names)

        print('Saving results...\n    DATABASE: {}'.format(database_filepath))
        save_data(results, database_filepath, 'results')

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