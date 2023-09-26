'''
Program that compares the mean and standard deviation of multiple sklearn
algorithms on the Pimas Indians Diabetes dataset and outputs a box & whisker
plot of the results
'''
import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


#Download dataset and convert into pandas DF
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = pandas.read_csv(url, names=names)
array = df.values
#Separate input & output values
x = array[:, 0:8]
y = array[:, 8]

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
funcs = []
scores = []
#Set the seed
seed = 7
for method in methods:
    kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
    model = methods[method]()
    funcs.append(method)
    results = model_selection.cross_val_score(model, x, y, cv=kfold)
    scores.append(results)
    print('%s: %.3f' % (method, results.mean()))
    if results.mean() >= best_res:
        best_res = results.mean()
        best_name = method

#Output which method performed the best
print('\nThe best performing algorithm is %s with an mean accuracy of %.3f' % (best_name, best_res))

#Box and whisker plot comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(scores)
ax.set_xticklabels(funcs)
plt.show()