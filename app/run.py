import json
import plotly
import pandas as pd
import pickle
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter, Pie
from sqlalchemy import create_engine
app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/processed/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)
results = pd.read_sql_table('results', engine)

# load model
model = pickle.load(open("../models/model.pkl", 'rb'))

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # Get only targets data
    targets = df[df.columns[4:]].copy()

    # Plot1 - Genre messages
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    graph_one = []
    for idx in range(len(genre_counts)):
        graph_one.append(Bar(
                        x=[genre_names[idx]],
                        y=[genre_counts[idx]],
                    ))
    layout_one = dict(title = 'Distribution of Message Genres',
                    xaxis = dict(title = 'Genre'),
                    yaxis = dict(title = 'Count'),
                    showlegend = False
    )

    # Plot2 - Test f1-score x positive class distribution for each label
    distrib_df = round(targets.mean() * 100).reset_index()
    distrib_df.columns = ['category', 'distrib']

    results_merge_df = results.merge(distrib_df, how='inner', on='category')
    results_merge_df.f1 = results_merge_df.f1 * 100

    graph_two = []
    for _, tuple in results_merge_df[['category', 'f1', 'distrib']].iterrows():
        graph_two.append(Scatter(
        x = [tuple[1]],
        y = [tuple[2]],
        mode =  'markers',
        name = tuple[0]
        )) 
        
    layout_two = dict(title = 'F1-score x Percentage of postive class',
                xaxis = dict(title = 'F1-score test (%)'),
                yaxis = dict(title = 'Pct of positive class (%)')
    
    )    

    # Plot3 - Number of categories
           
    num_categories = (targets.sum(axis=1)).value_counts(normalize=True).sort_index()

    graph_three = []

    # Concatenate 8+ categories
    concat_8_more = pd.Series(num_categories[8:].sum(), index=['8+'])

    num_categories_compact = num_categories[:8].append(concat_8_more).reset_index()
    num_categories_compact.columns = ['qtd', 'pct']

    qts = num_categories_compact.qtd.astype('str')
    pcts = num_categories_compact.pct

    graph_three.append(Pie(
                values=pcts,
                labels=qts,
                sort=False
            ))

    layout_three = dict(title = 'Percentage of category quantities'
    )

    graphs = []
    
    graphs.append(dict(data=graph_one, layout=layout_one))
    graphs.append(dict(data=graph_two, layout=layout_two))
    graphs.append(dict(data=graph_three, layout=layout_three))

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query') 
    genre = request.args.get('genre')
    
    input_df = pd.DataFrame([[query,genre]], columns=['message', 'genre'])

    # use model to predict classification for query
    classification_labels = model.predict(input_df)[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
