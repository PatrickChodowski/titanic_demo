from sklearn.pipeline import Pipeline
from joblib import load, dump
from preprocessing import preprocessing
import pandas as pd
import numpy as np
pd.set_option("display.max_columns", 100)

test_df = pd.read_csv('data/test.csv')
test_df2 = preprocessing(test_df)

model = load('model.pkl')
test_df2['prediction'] = model.predict(test_df2)

test_df2.reset_index(inplace=True)
test_df2[['passenger_id', 'prediction']].head()

