import pandas as pd
import numpy as np
import matplotlib
import scipy

TARGET_COL = 'Survived'
ID_COL = 'PassengerId'


def preprocessing(df: pd.DataFrame) -> pd.DataFrame:

    # target column class distribution check
    # train_df[TARGET_COL].value_counts() / train_df.shape[0]
    # 0    0.616162
    # 1    0.383838

    # Pclass is fake numeric - its really a categorical column, so I will transform it into dummy variables (0/1)
    # train_df['Pclass'].value_counts()
    #
    # # Convert all categoricals
    # train_df['Sex'].value_counts()
    df['Embarked'] = df['Embarked'].str.lower()

    df['sibsp_category'] = np.where(df['SibSp'] == 0, '0',
                                          np.where(df['SibSp'] == 1, '1',
                                                   np.where(df['SibSp'] >= 2, '2+', '')))

    df['parch_category'] = np.where(df['Parch'] == 0, '0',
                                          np.where(df['Parch'] == 1, '1',
                                                   np.where(df['Parch'] >= 2, '2+', '')))

    df['has_cabin'] = np.where(df['Cabin'].isnull(), 0, 1)
    df.drop(['SibSp', 'Parch', 'Ticket', 'Name', 'Cabin'], axis=1, inplace=True, errors='ignore')

    # impute age
    mean_age = df['Age'].mean()
    df['Age'] = df['Age'].fillna(mean_age)

    df2 = pd.get_dummies(df,
                         columns=['Pclass', 'Sex', 'sibsp_category', 'parch_category', 'Embarked'],
                         prefix=['pclass', 'sex', 'sibsp', 'parch', 'embarked'])

    df2.rename(columns={'PassengerId': 'passenger_id',
                        'Survived': 'survived',
                        'Age': 'age',
                        'Fare': 'fare'}, inplace=True)

    df2.set_index('passenger_id', inplace=True)
    df2.fillna(0, inplace=True)
    return df2
