'''
Example code using 3 different evaluations on a Linear Regression
model used on the Boston Housing Price Dataset.
'''
import pandas
from sklearn import model_selection
from sklearn.linear_model import LinearRegression

#Download the dataset and convert it into a pandas df
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pandas.read_csv(url, delim_whitespace=True, names=names)
array = df.values
#Separate input & output data
x = array[:, 0:13]
y = array[:, 13]
#Use 10 fold CV test
kfold = model_selection.KFold(n_splits=10, random_state=4, shuffle=True)
#Create Linear Regression model
model = LinearRegression()

#Create list for the 3 methods used to evaluate the accuracy of the model
methods = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']

#Now loop over each method and output their results
for method in methods:
    results = model_selection.cross_val_score(model, x, y, cv=kfold, scoring=method)
    print('%s: %.3f (%.3f)' % (method, results.mean(), results.std()))