'''
Example code using 5 different evaluation metrics on a
Logistic Regression model for the Pimas Indians Dataset.
'''

import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

#Download dataset and convert into pandas DF
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = pandas.read_csv(url, names=names)
array = df.values
#Separate input & output values
x = array[:, 0:8]
y = array[:, 8]
#Use 10 fold CV test
kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
model = LogisticRegression(solver='liblinear')

#Create a list for the 3 different scoring methods
methods = ['accuracy', 'neg_log_loss', 'roc_auc']

#Now go through each member of the list and output their results
for method in methods:
    results = model_selection.cross_val_score(model, x, y, cv=kfold, scoring=method)
    print('%s: %.3f (%.3f)' % (method, results.mean(), results.std()))

#For the final 2 metrics need to split the dataset into training and testing sets
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.33, random_state=7)
#Fit the model on the training data
model.fit(x_train, y_train)
#Get predictions for the model on the testing data
preds = model.predict(x_test)

#Output a confusion matrix for the predicted outcomes against the actual outcomes
print('\n', confusion_matrix(y_test, preds))

#Finally output the classification report for the predictions
print(classification_report(y_test, preds))
