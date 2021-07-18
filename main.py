import pandas as pd
from preprocessing import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from joblib import load, dump

pd.set_option("display.max_columns", 100)
train_df = pd.read_csv('data/train.csv')

train_df2 = preprocessing(train_df)
X = train_df2.drop('survived', axis=1)
Y = train_df2['survived']


# split the data into train/test parts
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

pipe = Pipeline([('scaler', StandardScaler()),
                 ('normalizer', MinMaxScaler()),
                 ('lr', RandomForestClassifier())])

pipe.fit(x_train, y_train)

# 81.5% for LR, 75% DT, 81.5% RF
pipe.score(x_test, y_test)


pipe.predict(x_test)



dump(pipe, 'model.pkl')

