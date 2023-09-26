'''Use 4 techniques to evaluate the Pima Indians Dataset with a Logistic Regression
    1) Train & Test Sets
    2) K-Fold Cross Validation
    3) Leave One Out Cross Validation
    4) Repeated Random Test-Train Splits
'''
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

#Load data into dataframe
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
array = data.values

#Split input & output data
x = array[:, 0:8]
y = array[:, 8]

#1) Evaluate using train & test split of 67 & 33 respectively
test_size = 0.33
seed = 10
#Create train & test splits from dataset
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=test_size, random_state=seed)
model = LogisticRegression(max_iter=1000)
#Train model on training data
model.fit(x_train, y_train)
#Test model on testing data
result = model.score(x_test, y_test)
#Output accuracy of model
print('Accuracy: %.3f%%' % (result * 100.0))

#2) Evaluate with K-Fold Cross Validation
num_instances = len(x)
#Use 10-fold cv
kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
#Test on Logistic Regression model
#modelTwo = LogisticRegression()
results = model_selection.cross_val_score(model, x, y, cv=kfold)
print('Accuracy: %3f%% (%3f%%)' % (results.mean() * 100.0, results.std() * 100.0))

#3) Evaluate with Leave-One-Out CV
loocv = model_selection.LeaveOneOut()
#modelThree = LogisticRegression()
looResults = model_selection.cross_val_score(model, x, y, cv=loocv)
print('Accuracy: %.3f%% (%.3f%%)' % (looResults.mean() * 100.0, looResults.std() * 100.0))

#4) Evaluate using shuffle split CV
kFold = model_selection.ShuffleSplit(n_splits=10, test_size=test_size, random_state=seed)
shufResults = model_selection.cross_val_score(model, x, y, cv=kFold)
print('Accuracy: %.3f%% (%.3f%%)' % (shufResults.mean() * 100.0, shufResults.std() * 100.0))
