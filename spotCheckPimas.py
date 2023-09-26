'''
Using 2 linear & 4 non-linear ML algorithms from sklearn to evaluate
their mean accuracy on the classification problem of early onset diabetes
from the Pimas Indians Dataset
'''
import pandas
from sklearn import model_selection
#Import all 6 of the methods
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

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

#Create a dictionary for the 6 methods to easily create models & output their results
methods = {'LR': LogisticRegression,
           'LDA': LinearDiscriminantAnalysis,
           'KNN': KNeighborsClassifier,
           'NB': GaussianNB,
           'CART': DecisionTreeClassifier,
           'SVM': SVC
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
