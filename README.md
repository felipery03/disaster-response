# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python src/process_data.py data/raw/disaster_messages.csv data/raw/disaster_categories.csv data/processed/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python src/train_classifier.py data/processed/DisasterResponse.db models/model.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
