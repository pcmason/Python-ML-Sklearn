'''
Using 7 different regression methods to spot check how well each model solves
the linear regression problem of housing prices from the Boston Housing Dataset
'''

import pandas
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

#Download the dataset and convert it into a pandas df
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pandas.read_csv(url, delim_whitespace=True, names=names)
array = df.values
#Separate input & output data
x = array[:, 0:13]
y = array[:, 13]
#Use 10 fold CV test
kfold = model_selection.KFold(n_splits=10, random_state=11, shuffle=True)

#Now put all of the Regression methods into a dictionary
methods = {
    'LR': LinearRegression,
    'Ridge': Ridge,
    'Lasso': Lasso,
    'EN': ElasticNet,
    'KNN': KNeighborsRegressor,
    'CART': DecisionTreeRegressor,
    'SVM': SVR
}

#Go through the methods to create models & output their mean score
best_res = 0
best_name = ''
for method in methods:
    model = methods[method]()
    results = model_selection.cross_val_score(model, x, y, cv=kfold)
    print('%s: %.3f' % (method, results.mean()))
    if results.mean() >= best_res:
        best_res = results.mean()
        best_name = method

#Output which method performed the best
print('\nThe best performing algorithm is %s with an mean accuracy of %.3f' % (best_name, best_res))