import numpy as np
import pandas as pd #import pandas
from tpot import TPOTClassifier
from sklearn import preprocessing
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
  ########not working ###
del()  
#database
df = pd.read_csv('G:/new researches/breast cancer text paper/databases/DB4/DB4MOD.csv')

X = np.array(df.drop(['class'], 1))

y = np.array(df['class'])


# Look at the dataset again
print(f'Number of Rows: {df.shape[0]}')
print(f'Number of Columns: {df.shape[1]}')
print(df.head())
# define model evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=123)
tpot = TPOTClassifier(generations=10, population_size=50, cv=cv, scoring='accuracy', verbosity=2, random_state=1, n_jobs=-1)
tpot.fit(X, y)
tpot.export('bestmodel2.py')
# clf = TPOTClassifier(config_dict='TPOT NN', template='Selector-Transformer-PytorchLRClassifier',
#                      verbosity=2, population_size=10, generations=5)
# clf.fit(X_train, y_train)
# print(clf.score(X_test, y_test))
# clf.export('tpot_nn_demo_pipeline.py')