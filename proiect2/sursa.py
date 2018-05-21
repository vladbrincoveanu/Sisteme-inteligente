import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
input_file = "/home/vlad/Desktop/Heart-Disease-Prediction-using-Machine-Leaning-master/test_2v.csv"


# comma delimited is the default
data = pd.read_csv(input_file, header = 0)
data = data.dropna()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data_1 = data.apply(le.fit_transform)
data_2 = data_1.loc[data_1['hypertension'] == 1] #1390
#x = data_1[data.columns.drop([[0,3,4]])]
#x = data_1.drop(data_1.columns[[0,3]],axis=1)
#y = data_1['hypertension']
data_3 = data_1[:2800];

result = pd.concat([data_2,data_3]);
#x = data_1[data.columns.drop([[0,3,4]])]
x = result.drop(result.columns[[0,3]],axis=1)
y = result['hypertension']

from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(x,
y,test_size=0.4,random_state=42)

from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
#clf = MLPClassifier(activation='logistic',solver='adam', alpha=1e-5, hidden_layer_sizes=(2,2), random_state=1, verbose=True, max_iter=400)
clf = LinearSVC(C=0.0000001)
clf.fit(X_train, y_train)

print clf.score(X_test,y_test)
y_predict=clf.predict(X_test)


# np.random.seed(0)
# for _ in range(6): 
# this_X = .1*np.random.normal(size=(2, 1)) + X_test
# regr.fit(this_X, y_train)
# plt.plot(y_test, y_predict) 
# plt.scatter(this_X, y, s=3)


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print classification_report(y_test,y_predict)
print confusion_matrix(y_test, y_predict)
#print (tn, fp, fn, tp)
